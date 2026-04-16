import { Injectable, signal, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of, throwError } from 'rxjs';
import { catchError, delay, map, tap } from 'rxjs/operators';
import { Message, PredictRequest, PredictResponse } from '../models/message.model';

@Injectable({
  providedIn: 'root'
})
export class SentimentAnalysisService {
  private http = inject(HttpClient);
  private apiUrl = 'http://localhost:8000/predict';

  // Signals for state management
  private messagesSignal = signal<Message[]>([]);
  public messages = this.messagesSignal.asReadonly();

  constructor() {
    this.loadHistory();
  }

  public sendMessage(text: string, lang: 'vi' | 'en' = 'vi'): void {
    this.addMessageSequence(text, text, lang, 'predict', null);
  }

  public sendDocument(file: File, lang: 'vi' | 'en' = 'vi'): void {
    const displayText = `📄 ${file.name}`;
    const fileFormData = new FormData();
    fileFormData.append('file', file);
    fileFormData.append('lang', lang);
    this.addMessageSequence(displayText, `Nội dung trích xuất từ file ${file.name}...`, lang, 'upload/document', fileFormData);
  }

  public sendAudio(blob: Blob, lang: 'vi' | 'en' = 'vi'): void {
    const audioFormData = new FormData();
    audioFormData.append('file', blob);
    audioFormData.append('lang', lang);
    
    // Giả định Text mà AI nghe được từ user
    const transcribedText = "🎤 [Giọng nói]: Chào hệ thống, tôi thấy dịch vụ của bạn thật sự rất tuyệt vời và đáng kinh ngạc!";
    this.addMessageSequence(transcribedText, transcribedText, lang, 'upload/audio', audioFormData);
  }

  private addMessageSequence(userDisplayText: string, backendText: string, lang: 'vi'|'en', endpoint: string, payload: any): void {
    const userMessage: Message = {
      id: crypto.randomUUID(),
      text: userDisplayText,
      sender: 'user',
      timestamp: new Date(),
      sentiment: null
    };

    this.messagesSignal.update(msgs => [...msgs, userMessage]);

    const botMessageId = crypto.randomUUID();
    const botLoadingMessage: Message = {
      id: botMessageId,
      text: 'Đang phân tích...',
      sender: 'bot',
      timestamp: new Date(),
      sentiment: 'LOADING'
    };

    this.messagesSignal.update(msgs => [...msgs, botLoadingMessage]);

    let requestOb$: Observable<PredictResponse>;
    if (endpoint === 'predict') {
      requestOb$ = this.predict(backendText, lang);
    } else {
      // Gọi API Upload
      requestOb$ = this.http.post<PredictResponse>(`http://localhost:8000/${endpoint}`, payload).pipe(
        catchError(err => {
          console.warn(`Backend connection failed for ${endpoint}, using mock fallback...`, err);
          return this.mockPredict(backendText);
        })
      );
    }

    requestOb$.subscribe({
      next: (res) => {
        this.updateBotMessage(botMessageId, {
          text: this.getSentimentText(res.sentiment),
          sentiment: res.sentiment.toUpperCase() as any,
          confidenceScore: res.confidence,
          latency_ms: res.latency_ms
        });
      },
      error: () => {
        this.updateBotMessage(botMessageId, {
          text: 'Rất tiếc, tôi không thể phân tích lúc này. Hãy thử lại sau!',
          sentiment: 'ERROR'
        });
      }
    });
  }

  private predict(text: string, lang: 'vi' | 'en'): Observable<PredictResponse> {
    return this.http.post<PredictResponse>(this.apiUrl, { text, lang } as PredictRequest).pipe(
      catchError(err => {
        console.warn('Backend connection failed, using mock fallback...', err);
        return this.mockPredict(text);
      })
    );
  }

  private mockPredict(text: string): Observable<PredictResponse> {
    const positiveWords = ['ngon', 'tốt', 'tuyệt', 'hay', 'good', 'great', 'awesome'];
    const negativeWords = ['tệ', 'dở', 'kém', 'xấu', 'bad', 'horrible', 'hate'];
    
    let sentiment: 'positive' | 'negative' | 'neutral' = 'neutral';
    const lowerText = text.toLowerCase();
    
    if (positiveWords.some(w => lowerText.includes(w))) sentiment = 'positive';
    else if (negativeWords.some(w => lowerText.includes(w))) sentiment = 'negative';

    return of({
      text,
      sentiment,
      confidence: 0.85 + Math.random() * 0.1,
      aspects: [],
      sarcasm_flag: false,
      latency_ms: 120
    }).pipe(delay(1000));
  }

  private updateBotMessage(id: string, partial: Partial<Message>): void {
    this.messagesSignal.update(msgs => msgs.map(m => 
      m.id === id ? { ...m, ...partial } : m
    ));
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
        const parsed = JSON.parse(saved);
        this.messagesSignal.set(parsed);
      } catch (e) {
        console.error('Failed to load history', e);
      }
    }
  }

  private getSentimentText(sentiment: string): string {
    switch (sentiment) {
      case 'positive': return 'Tuyệt vời! Tôi cảm nhận được năng lượng tích cực từ bạn. ✨';
      case 'negative': return 'Có vẻ như bạn đang gặp chuyện không vui. Tôi luôn ở đây lắng nghe. 🫂';
      default: return 'Tôi đã ghi nhận ý kiến của bạn một cách khách quan. Cảm ơn bạn! 📝';
    }
  }
}
