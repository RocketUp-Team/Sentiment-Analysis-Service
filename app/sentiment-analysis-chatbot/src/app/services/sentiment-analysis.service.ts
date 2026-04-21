import { Injectable, signal, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { catchError } from 'rxjs/operators';
import {
  Message,
  AspectSentiment,
  PredictRequest,
  PredictResponse,
  ExplainRequest,
  ExplainResponse,
  BatchPredictResponse,
} from '../models/message.model';

@Injectable({
  providedIn: 'root',
})
export class SentimentAnalysisService {
  private http = inject(HttpClient);
  private apiUrl = '/api/predict';

  private messagesSignal = signal<Message[]>([]);
  public messages = this.messagesSignal.asReadonly();

  public checkHealth(): Observable<any> {
    return this.http.get('/api/health');
  }

  constructor() {
    this.loadHistory();
  }

  // ── Text analysis ────────────────────────────────────────────
  public sendMessage(text: string): void {
    this.addMessageSequence(text);
  }

  public addWelcomeMessage(): void {
    const welcome: Message = {
      id: crypto.randomUUID(),
      text: '👋 Sentiment Analysis Service is ready!\n\nType any text to analyse its sentiment (positive / negative / neutral) with aspect breakdown.\n\n📄 Use the Upload CSV button to run batch predictions.',
      sender: 'bot',
      timestamp: new Date(),
      sentiment: null,
    };
    this.messagesSignal.update(msgs => [...msgs, welcome]);
  }

  private addMessageSequence(text: string): void {
    const userMessage: Message = {
      id: crypto.randomUUID(),
      text,
      sender: 'user',
      timestamp: new Date(),
      sentiment: null,
    };
    this.messagesSignal.update(msgs => [...msgs, userMessage]);

    const botMessageId = crypto.randomUUID();
    const loadingMessage: Message = {
      id: botMessageId,
      text: 'Analysing...',
      sender: 'bot',
      timestamp: new Date(),
      sentiment: 'LOADING',
    };
    this.messagesSignal.update(msgs => [...msgs, loadingMessage]);

    this.predict(text).subscribe({
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
      error: () => {
        this.updateBotMessage(botMessageId, {
          text: 'System error — could not reach the API. Please check your connection.',
          sentiment: 'ERROR',
        });
      },
    });
  }

  private predict(text: string): Observable<PredictResponse> {
    return this.http.post<PredictResponse>(this.apiUrl, { text } as PredictRequest);
  }

  // ── Explain (SHAP) ───────────────────────────────────────────
  /**
   * Calls POST /api/explain for a bot message.
   * Looks up the user message that PRECEDES the bot message in the signal
   * to get the original text — no need to store sourceText anywhere.
   */
  public explainMessage(botMessageId: string): void {
    const msgs = this.messagesSignal();
    const botIdx = msgs.findIndex(m => m.id === botMessageId);

    // Walk backwards from the bot message to find the closest user message
    const userMsg = botIdx > 0
      ? msgs.slice(0, botIdx).reverse().find(m => m.sender === 'user')
      : null;

    if (!userMsg?.text?.trim()) {
      console.warn('[explain] No preceding user message found for', botMessageId);
      return;
    }

    // Mark loading
    this.messagesSignal.update(all =>
      all.map(m => m.id === botMessageId ? { ...m, explainLoading: true } : m)
    );

    this.callExplainApi(userMsg.text.trim()).subscribe({
      next: (res) => {
        this.messagesSignal.update(all =>
          all.map(m =>
            m.id === botMessageId
              ? { ...m, explainLoading: false, explainData: res }
              : m
          )
        );
        this.saveHistory();
      },
      error: (err) => {
        console.error('[explain] API error', err);
        this.messagesSignal.update(all =>
          all.map(m => m.id === botMessageId ? { ...m, explainLoading: false } : m)
        );
      },
    });
  }

  private callExplainApi(text: string): Observable<ExplainResponse> {
    return this.http.post<ExplainResponse>('/api/explain', { text } as ExplainRequest);
  }

  // ── Batch CSV ────────────────────────────────────────────────
  public batchPredict(file: File): Observable<BatchPredictResponse> {
    const form = new FormData();
    form.append('file', file);
    return this.http.post<BatchPredictResponse>('/api/batch_predict', form);
  }

  // ── Helpers ──────────────────────────────────────────────────
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

  private updateBotMessage(id: string, partial: Partial<Message>): void {
    this.messagesSignal.update(msgs =>
      msgs.map(m => (m.id === id ? { ...m, ...partial } : m))
    );
    this.saveHistory();
  }

  public clearHistory(): void {
    this.messagesSignal.set([]);
    localStorage.removeItem('sentiment_chat_history');
  }

  private saveHistory(): void {
    localStorage.setItem('sentiment_chat_history', JSON.stringify(this.messagesSignal()));
  }

  private loadHistory(): void {
    const saved = localStorage.getItem('sentiment_chat_history');
    if (saved) {
      try {
        this.messagesSignal.set(JSON.parse(saved));
      } catch {
        // ignore corrupt history
      }
    }
  }
}
