import { Injectable, signal, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { catchError } from 'rxjs/operators';
import {
  Message,
  AspectSentiment,
  PredictRequest,
  PredictResponse,
} from '../models/message.model';

@Injectable({
  providedIn: 'root',
})
export class SentimentAnalysisService {
  private http = inject(HttpClient);
  private apiUrl = '/api/predict';

  // Signals for state management
  private messagesSignal = signal<Message[]>([]);
  public messages = this.messagesSignal.asReadonly();

  public checkHealth(): Observable<any> {
    return this.http.get('/api/health');
  }

  constructor() {
    this.loadHistory();
  }

  public sendMessage(text: string, lang: 'vi' | 'en' = 'en'): void {
    this.addMessageSequence(text, text, lang, 'predict', null);
  }

  public addWelcomeMessage(): void {
    const welcome: Message = {
      id: crypto.randomUUID(),
      text: '👋 Sentiment Analysis Service is ready!\n\nType any text to analyse its sentiment (positive / negative / neutral) with aspect breakdown.\n\n📄 You can also upload a CSV file for batch predictions.',
      sender: 'bot',
      timestamp: new Date(),
      sentiment: null,
    };
    this.messagesSignal.update(msgs => [...msgs, welcome]);
  }

  public sendDocument(file: File, lang: 'vi' | 'en' = 'en'): void {
    const displayText = `📄 ${file.name}`;
    const fileFormData = new FormData();
    fileFormData.append('file', file);
    fileFormData.append('lang', lang);
    this.addMessageSequence(
      displayText,
      `Nội dung trích xuất từ file ${file.name}...`,
      lang,
      'upload/document',
      fileFormData,
    );
  }

  public sendAudio(blob: Blob, lang: 'vi' | 'en' = 'en'): void {
    const audioFormData = new FormData();
    audioFormData.append('file', blob);
    audioFormData.append('lang', lang);

    const userDisplayText = '🎤 [Audio Message]';
    this.addMessageSequence(
      userDisplayText,
      '[Audio Content]', 
      lang,
      'upload/audio',
      audioFormData,
    );
  }

  private addMessageSequence(
    userDisplayText: string,
    backendText: string,
    lang: 'vi' | 'en',
    endpoint: string,
    payload: any,
  ): void {
    const userMessage: Message = {
      id: crypto.randomUUID(),
      text: userDisplayText,
      sender: 'user',
      timestamp: new Date(),
      sentiment: null,
    };

    this.messagesSignal.update((msgs) => [...msgs, userMessage]);

    const botMessageId = crypto.randomUUID();
    const botLoadingMessage: Message = {
      id: botMessageId,
      text: 'Đang phân tích...',
      sender: 'bot',
      timestamp: new Date(),
      sentiment: 'LOADING',
    };

    this.messagesSignal.update((msgs) => [...msgs, botLoadingMessage]);

    let requestOb$: Observable<PredictResponse>;
    if (endpoint === 'predict') {
      requestOb$ = this.predict(backendText, lang);
    } else {
      // Gọi API Upload
      requestOb$ = this.http.post<PredictResponse>(`/api/${endpoint}`, payload);
    }

    requestOb$.subscribe({
      next: (res) => {
        this.updateBotMessage(botMessageId, {
          text: this.buildBotReply(res),
          sentiment: res.sentiment.toUpperCase() as any,
          confidenceScore: res.confidence,
          aspects: res.aspects,
          sarcasm_flag: res.sarcasm_flag,
          latency_ms: res.latency_ms,
        });
      },
      error: (err) => {
        console.error(`API error for ${endpoint}:`, err);
        this.updateBotMessage(botMessageId, {
          text: 'System error — could not reach the API. Please check your connection.',
          sentiment: 'ERROR',
        });
      },
    });
  }

  private predict(
    text: string,
    lang: 'vi' | 'en',
  ): Observable<PredictResponse> {
    return this.http.post<PredictResponse>(this.apiUrl, { text, lang } as PredictRequest);
  }

  private updateBotMessage(id: string, partial: Partial<Message>): void {
    this.messagesSignal.update((msgs) =>
      msgs.map((m) => (m.id === id ? { ...m, ...partial } : m)),
    );
    this.saveHistory();
  }

  public clearHistory(): void {
    this.messagesSignal.set([]);
    localStorage.removeItem('sentiment_chat_history');
  }

  private saveHistory(): void {
    localStorage.setItem(
      'sentiment_chat_history',
      JSON.stringify(this.messagesSignal()),
    );
  }

  private loadHistory(): void {
    const saved = localStorage.getItem('sentiment_chat_history');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        this.messagesSignal.set(parsed);
      } catch (e) {
        console.error('Failed to load history', e);
      }
    }
  }

  private buildBotReply(res: PredictResponse): string {
    const emoji = res.sentiment === 'positive' ? '😊' : res.sentiment === 'negative' ? '😞' : '😐';
    const label  = res.sentiment.charAt(0).toUpperCase() + res.sentiment.slice(1);
    const pct    = (res.confidence * 100).toFixed(1);

    let reply = `${emoji} Sentiment detected: **${label}** (confidence ${pct}%)`;

    if (res.sarcasm_flag) {
      reply += `\n⚠️ Sarcasm detected — result may be inverted.`;
    }

    if (res.aspects && res.aspects.length > 0) {
      reply += `\n\n📊 Aspect breakdown:`;
      res.aspects.forEach((a: AspectSentiment) => {
        const icon = a.sentiment === 'positive' ? '✅' : a.sentiment === 'negative' ? '❌' : '➖';
        reply += `\n  ${icon} ${a.aspect}: ${a.sentiment} (${(a.confidence * 100).toFixed(0)}%)`;
      });
    } else {
      reply += `\n\nℹ️ No specific aspects were detected above the threshold.`;
    }

    reply += `\n\n⏱ Processed in ${res.latency_ms.toFixed(0)} ms`;
    return reply;
  }
}
