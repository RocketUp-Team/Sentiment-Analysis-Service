import {
  Component,
  Output,
  EventEmitter,
  Input,
  OnDestroy,
  ChangeDetectorRef,
  ElementRef,
  ViewChild,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import {
  LucideAngularModule,
  Paperclip,
  Smile,
  Mic,
  Send,
  Globe,
  Loader2,
  RefreshCw,
  AlertTriangle,
} from 'lucide-angular';

// ── Language definitions ────────────────────────────────────────────────────
interface LangOption {
  code: string;
  label: string;
  bcp47: string;
  flag: string;
}

const LANGUAGES: LangOption[] = [
  { code: 'auto', label: 'AUTO', bcp47: '',      flag: '🌐' },
  { code: 'vi',   label: 'VI',   bcp47: 'vi-VN', flag: '🇻🇳' },
  { code: 'en',   label: 'EN',   bcp47: 'en-US', flag: '🇺🇸' },
  { code: 'ja',   label: 'JA',   bcp47: 'ja-JP', flag: '🇯🇵' },
  { code: 'ko',   label: 'KO',   bcp47: 'ko-KR', flag: '🇰🇷' },
  { code: 'zh',   label: 'ZH',   bcp47: 'zh-CN', flag: '🇨🇳' },
  { code: 'fr',   label: 'FR',   bcp47: 'fr-FR', flag: '🇫🇷' },
  { code: 'de',   label: 'DE',   bcp47: 'de-DE', flag: '🇩🇪' },
  { code: 'es',   label: 'ES',   bcp47: 'es-ES', flag: '🇪🇸' },
];

// ── Script detection helpers ────────────────────────────────────────────────
function detectScript(text: string): string | null {
  if (!text) return null;
  const counts: Record<string, number> = { ko: 0, zh: 0, ja: 0, ar: 0, hi: 0, latin: 0 };
  for (const ch of text) {
    const cp = ch.codePointAt(0)!;
    if (cp >= 0xAC00 && cp <= 0xD7A3) { counts['ko']++; continue; }  // Hangul syllables
    if (cp >= 0x1100 && cp <= 0x11FF) { counts['ko']++; continue; }  // Hangul jamo
    if (cp >= 0x3040 && cp <= 0x30FF) { counts['ja']++; continue; }  // Hiragana / Katakana
    if (cp >= 0x4E00 && cp <= 0x9FFF) { counts['zh']++; continue; }  // CJK Unified
    if (cp >= 0x0600 && cp <= 0x06FF) { counts['ar']++; continue; }  // Arabic
    if (cp >= 0x0900 && cp <= 0x097F) { counts['hi']++; continue; }  // Devanagari
    if ((cp >= 0x0041 && cp <= 0x007A) || (cp >= 0x00C0 && cp <= 0x024F)) {
      counts['latin']++;
    }
  }
  const dominant = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  return dominant[1] > 0 ? dominant[0] : null;
}

const SCRIPT_TO_BCP47: Record<string, string> = {
  ko: 'ko-KR', zh: 'zh-CN', ja: 'ja-JP',
  ar: 'ar-SA', hi: 'hi-IN', latin: '',
};

// ── Component ────────────────────────────────────────────────────────────────
@Component({
  selector: 'app-chat-input',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideAngularModule],
  styles: [`
    /* ── Mic ripple animations ── */
    @keyframes mic-ripple {
      0%   { transform: scale(1); opacity: 0.55; }
      100% { transform: scale(2.4); opacity: 0; }
    }
    .mic-ripple      { animation: mic-ripple 1s ease-out infinite; }
    .mic-ripple-slow { animation: mic-ripple 1s ease-out 0.42s infinite; }

    /* ── Spinner ── */
    @keyframes _spin { to { transform: rotate(360deg); } }
    .spin { animation: _spin 0.75s linear infinite; }

    /* ── Pulse border ring on recording button ── */
    @keyframes pulse-ring {
      0%   { box-shadow: 0 0 0 0 rgba(239,68,68,0.7); }
      70%  { box-shadow: 0 0 0 14px rgba(239,68,68,0); }
      100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
    }
    .mic-pulse { animation: pulse-ring 1s ease-out infinite; }

    /* ── Fixed lang dropdown ── */
    .lang-dropdown {
      position: fixed;
      z-index: 9999;
      background: #fff;
      border-radius: 14px;
      box-shadow: 0 12px 40px rgba(0,0,0,0.18);
      border: 1px solid rgba(99,102,241,0.18);
      overflow: hidden;
      min-width: 164px;
    }
    :host-context(.dark) .lang-dropdown {
      background: #0f172a;
      border-color: rgba(255,255,255,0.08);
    }
    .lang-item {
      display: flex;
      align-items: center;
      gap: 0.55rem;
      padding: 0.55rem 1rem;
      font-size: 0.8rem;
      font-weight: 700;
      cursor: pointer;
      transition: background 0.12s;
      white-space: nowrap;
      color: inherit;
    }
    .lang-item:hover { background: rgba(99,102,241,0.10); color: #6366f1; }
    .lang-item.active { background: rgba(99,102,241,0.14); color: #6366f1; }

    /* ── Mismatch warning toast ── */
    .lang-warn {
      position: absolute;
      bottom: calc(100% + 6px);
      left: 0;
      right: 0;
      background: #fef3c7;
      border: 1px solid #fcd34d;
      color: #92400e;
      border-radius: 12px;
      padding: 8px 12px;
      font-size: 0.72rem;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 8px;
      z-index: 9999;
    }
    :host-context(.dark) .lang-warn {
      background: #422006;
      border-color: #92400e;
      color: #fcd34d;
    }
  `],
  template: `
    <!-- Wrapper: relative so the warning toast is anchored here -->
    <div class="relative flex items-center gap-4 w-full transition-all duration-500">

      <!-- ── Script mismatch warning ─────────────────────────── -->
      <div *ngIf="scriptWarning" class="lang-warn">
        <lucide-angular name="alert-triangle" [size]="14" style="flex-shrink:0"></lucide-angular>
        <span>{{ scriptWarning }}</span>
        <button
          *ngIf="suggestedLang"
          class="ml-auto px-2 py-0.5 rounded-lg bg-amber-500 text-white text-[10px] font-black uppercase tracking-wider flex items-center gap-1"
          (click)="retryWithSuggested()"
        >
          <lucide-angular name="refresh-cw" [size]="10"></lucide-angular>
          Thử lại
        </button>
        <button
          class="px-2 py-0.5 rounded-lg bg-black/10 dark:bg-white/10 text-[10px] font-black"
          (click)="scriptWarning = ''; suggestedLang = null"
        >✕</button>
      </div>

      <!-- ── Premium Glass Input Card ────────────────────────── -->
      <div
        class="flex-1 glass-panel !bg-white/40 dark:!bg-slate-900/40 rounded-3xl flex items-center px-3 md:px-6 gap-2 md:gap-4 border border-white/20 dark:border-white/5 focus-within:ring-2 focus-within:ring-indigo-500/30 focus-within:bg-white dark:focus-within:bg-slate-900 transition-all duration-300 relative"
      >

        <!-- Paperclip -->
        <button
          class="text-slate-400 hover:text-indigo-500 transition-all duration-300 active:scale-95 shrink-0"
          (click)="onOpenBatch.emit()"
          [disabled]="disabled || isRecording || isProcessing"
          title="Upload CSV for batch analysis"
        >
          <lucide-angular name="paperclip" [size]="22"></lucide-angular>
        </button>

        <!-- Language button — uses fixed dropdown to escape backdrop-blur stacking ctx -->
        <button
          #langBtn
          class="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-black/5 dark:bg-white/5 hover:bg-indigo-500/10 text-slate-500 dark:text-slate-400 transition-all active:scale-90 border border-transparent hover:border-indigo-500/20 shrink-0"
          (click)="toggleLangPicker($event)"
          [disabled]="disabled || isRecording || isProcessing"
          title="Switch language"
        >
          <span class="text-base leading-none">{{ currentLang.flag }}</span>
          <span class="text-[10px] font-black uppercase tracking-widest">{{ currentLang.label }}</span>
        </button>

        <!-- Recording / Processing state -->
        <div
          *ngIf="isRecording || isProcessing"
          class="flex-1 flex items-center gap-3 font-medium text-[15px] overflow-hidden"
          [class.text-red-500]="isRecording"
          [class.text-indigo-500]="isProcessing && !isRecording"
        >
          <ng-container *ngIf="isRecording">
            <div class="w-2.5 h-2.5 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)] animate-pulse shrink-0"></div>
            <span class="tracking-tight truncate">{{ interimText || 'Đang ghi âm...' }}</span>
            <span class="text-[10px] font-black bg-red-500/10 px-2 py-0.5 rounded-full shrink-0">{{ recordingTime }}s</span>
          </ng-container>
          <ng-container *ngIf="isProcessing && !isRecording">
            <div class="spin shrink-0 text-indigo-500">
              <lucide-angular name="loader-2" [size]="15"></lucide-angular>
            </div>
            <span class="tracking-tight">Đang xử lý...</span>
          </ng-container>
        </div>

        <!-- Normal text input -->
        <input
          *ngIf="!isRecording && !isProcessing"
          type="text"
          [(ngModel)]="text"
          (keyup.enter)="send()"
          [disabled]="disabled"
          placeholder="Type text to analyse sentiment..."
          class="flex-1 bg-transparent border-none focus:ring-0 text-[15px] md:text-[15.5px] py-4 md:py-5 outline-none dark:text-white placeholder:text-slate-400 font-medium min-w-0"
        />

        <button
          class="text-slate-400 hover:text-indigo-500 transition-all duration-300 active:scale-95 shrink-0"
          *ngIf="!isRecording && !isProcessing"
          [disabled]="disabled"
        >
          <lucide-angular name="smile" [size]="22"></lucide-angular>
        </button>
      </div>

      <!-- ── Mic / Send Button ─────────────────────────────────── -->
      <div class="relative flex items-center justify-center shrink-0">
        <div *ngIf="isRecording" class="absolute inset-0 bg-red-500/25 rounded-full mic-ripple pointer-events-none"></div>
        <div *ngIf="isRecording" class="absolute inset-0 bg-red-500/12 rounded-full mic-ripple-slow pointer-events-none"></div>

        <button
          id="mic-send-btn"
          (mousedown)="onMouseDown($event)"
          (mouseup)="onMouseUp($event)"
          (mouseleave)="onMouseLeave($event)"
          (touchstart)="onTouchStart($event)"
          (touchend)="onTouchEnd($event)"
          (touchcancel)="onTouchCancel($event)"
          (click)="handleClick($event)"
          [disabled]="disabled || isProcessing"
          [ngClass]="btnClass()"
          class="relative flex items-center justify-center text-white rounded-[24px] md:rounded-3xl transition-all duration-300 transform active:scale-95 shadow-xl select-none z-10"
          [title]="micTitle()"
        >
          <div *ngIf="hasSendText() && !isRecording" class="flex items-center gap-2.5">
            <span class="text-xs font-black uppercase tracking-[0.2em] hidden md:inline">Send</span>
            <lucide-angular name="send" [size]="20"></lucide-angular>
          </div>
          <lucide-angular *ngIf="!hasSendText() && !isRecording && !isProcessing" name="mic" [size]="24"></lucide-angular>
          <lucide-angular *ngIf="isRecording" name="mic" [size]="28"></lucide-angular>
          <div *ngIf="isProcessing && !isRecording" class="spin">
            <lucide-angular name="loader-2" [size]="24"></lucide-angular>
          </div>
        </button>

        <!-- Hint label -->
        <div
          *ngIf="!isRecording && !isProcessing && !hasSendText()"
          class="absolute -bottom-5 left-1/2 -translate-x-1/2 whitespace-nowrap text-[9px] font-bold tracking-widest text-slate-400 pointer-events-none uppercase"
        >Hold to speak</div>
      </div>
    </div>

    <!-- ── Fixed Lang Dropdown (portal-like, escapes stacking ctx) ── -->
    <ng-container *ngIf="showLangPicker">
      <!-- Invisible full-screen backdrop for close-on-outside-click -->
      <div
        class="fixed inset-0"
        style="z-index:9998"
        (click)="showLangPicker = false"
      ></div>
      <div
        class="lang-dropdown"
        [style.top.px]="dropdownY"
        [style.left.px]="dropdownX"
      >
        <div
          *ngFor="let l of langOptions"
          class="lang-item"
          [class.active]="l.code === currentLang.code"
          (click)="selectLang(l); $event.stopPropagation()"
        >
          <span>{{ l.flag }}</span>
          <span>{{ l.label }}</span>
          <span *ngIf="l.bcp47" class="text-[10px] text-slate-400 font-normal ml-auto">{{ l.bcp47 }}</span>
          <span *ngIf="!l.bcp47" class="text-[10px] text-slate-400 font-normal ml-auto">browser</span>
        </div>
      </div>
    </ng-container>
  `,
})
export class ChatInputComponent implements OnDestroy {
  @Input() disabled = false;
  @Output() onSendText  = new EventEmitter<{ text: string; lang: string }>();
  @Output() onSendAudio = new EventEmitter<{ audio: Blob; lang: string }>();
  @Output() onOpenBatch = new EventEmitter<void>();

  @ViewChild('langBtn', { read: ElementRef }) langBtnRef!: ElementRef;

  // ── UI state ──────────────────────────────────────────────────────────────
  text           = '';
  langOptions    = LANGUAGES;
  currentLang    = LANGUAGES[0];    // default: AUTO
  showLangPicker = false;
  dropdownX      = 0;
  dropdownY      = 0;

  isRecording  = false;
  isProcessing = false;
  recordingTime = 0;
  interimText  = '';

  scriptWarning : string       = '';
  suggestedLang : LangOption | null = null;

  // ── Private state ─────────────────────────────────────────────────────────
  private recognition      : any     = null;
  private recognitionActive          = false;
  private autoSend                   = false;
  private recordingInterval : any    = null;
  private isPointerDown              = false;
  private lastFinalText              = '';

  constructor(private cdr: ChangeDetectorRef) {
    this.initSpeechRecognition();
  }

  // ── Language picker ───────────────────────────────────────────────────────

  toggleLangPicker(e: MouseEvent) {
    e.stopPropagation();
    if (this.showLangPicker) { this.showLangPicker = false; return; }

    // Calculate fixed position from button rect
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    this.dropdownX = rect.left;
    // Show ABOVE the button; we'll estimate 280px for height, then adjust
    this.dropdownY = rect.top - 288;
    if (this.dropdownY < 8) this.dropdownY = rect.bottom + 6; // flip to below if no room

    this.showLangPicker = true;
  }

  selectLang(l: LangOption) {
    this.currentLang = l;
    this.showLangPicker = false;
    this.scriptWarning = '';
    this.suggestedLang = null;
    if (this.recognition) {
      this.recognition.lang = this.resolvedBcp47;
    }
    this.cdr.detectChanges();
  }

  /** Resolve effective BCP-47 for recognition */
  private get resolvedBcp47(): string {
    if (this.currentLang.code === 'auto') {
      // Use browser's UI language as best-effort auto
      return navigator.language || 'en-US';
    }
    return this.currentLang.bcp47;
  }

  // ── Speech Recognition ─────────────────────────────────────────────────────

  private initSpeechRecognition() {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SR) return;

    this.recognition = new SR();
    this.recognition.continuous      = true;
    this.recognition.interimResults  = true;
    this.recognition.maxAlternatives = 1;

    this.recognition.onstart = () => {
      this.recognitionActive = true;
      this.cdr.detectChanges();
    };

    this.recognition.onresult = (event: any) => {
      let interim = '';
      let final   = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) { final   += t; }
        else                          { interim += t; }
      }
      this.interimText = interim;
      if (final) {
        this.lastFinalText = final;
        this.text          = (this.text + ' ' + final).trim();
        this.interimText   = '';
      }
      this.cdr.detectChanges();
    };

    this.recognition.onerror = (event: any) => {
      if (event.error === 'aborted') this.autoSend = false;
      this.recognitionActive = false;
      this.finishRecording();
    };

    this.recognition.onend = () => {
      this.recognitionActive = false;
      this.isProcessing      = false;
      this.finishRecording();
    };
  }

  private startRecording() {
    if (!this.recognition) {
      alert('Trình duyệt không hỗ trợ Web Speech API. Vui lòng dùng Chrome hoặc Edge.');
      return;
    }
    if (this.recognitionActive) return;

    try {
      this.text          = '';
      this.interimText   = '';
      this.lastFinalText = '';
      this.autoSend      = false;
      this.scriptWarning = '';
      this.suggestedLang = null;

      this.recognition.lang = this.resolvedBcp47;
      this.recognition.start();

      this.isRecording  = true;
      this.isProcessing = false;
      this.recordingTime = 0;
      this.recordingInterval = setInterval(() => {
        this.recordingTime++;
        this.cdr.detectChanges();
      }, 1000);

      this.cdr.detectChanges();
    } catch (err) {
      console.error('[SR] start error:', err);
    }
  }

  private stopRecordingAndSend() {
    if (!this.isRecording) return;
    this.autoSend = true;
    this._stopRecognition(false);
  }

  private cancelRecording() {
    if (!this.isRecording) return;
    this.autoSend = false;
    this.text = '';
    this._stopRecognition(true);
  }

  private _stopRecognition(abort: boolean) {
    clearInterval(this.recordingInterval);
    this.recordingInterval = null;
    this.isRecording  = false;
    this.isProcessing = true;
    this.cdr.detectChanges();
    try {
      abort ? this.recognition?.abort() : this.recognition?.stop();
    } catch {
      this.isProcessing = false;
      this.finishRecording();
    }
  }

  private finishRecording() {
    this.isRecording  = false;
    this.isProcessing = false;
    clearInterval(this.recordingInterval);
    this.recordingInterval = null;
    this.interimText   = '';

    const doSend = this.autoSend;
    this.autoSend = false;

    // ── Script mismatch check ───────────────────────────────────
    if (this.text.trim()) {
      const script   = detectScript(this.text);
      const usedLang = this.currentLang.code;
      const mismatch = this.detectMismatch(script, usedLang);
      if (mismatch) {
        this.scriptWarning = mismatch.warning;
        this.suggestedLang = mismatch.suggest;
      }
    }

    this.cdr.detectChanges();

    if (doSend && this.text.trim()) {
      setTimeout(() => this.send(), 0);
    }
  }

  /**
   * Returns a warning + suggestion if the transcribed script clearly doesn't
   * match the recognition language that was used.
   */
  private detectMismatch(script: string | null, usedLang: string): { warning: string; suggest: LangOption | null } | null {
    if (!script || script === 'latin') return null; // Latin is ambiguous
    if (usedLang === 'auto') return null;           // AUTO mode: no mismatch check

    const scriptToLang: Record<string, string> = {
      ko: 'ko', zh: 'zh', ja: 'ja', ar: 'ar', hi: 'hi',
    };
    const expectedLang = scriptToLang[script];
    if (!expectedLang) return null;
    if (expectedLang === usedLang) return null;

    const suggest = LANGUAGES.find(l => l.code === expectedLang) ?? null;
    const warning = `⚠️ Phát hiện chữ ${script.toUpperCase()} nhưng đang dùng ${usedLang.toUpperCase()}. ${suggest ? `Thử lại với ${suggest.flag} ${suggest.label}?` : ''}`;
    return { warning, suggest };
  }

  /** Re-record with the suggested language after mismatch warning */
  retryWithSuggested() {
    if (!this.suggestedLang) return;
    this.currentLang   = this.suggestedLang;
    this.scriptWarning = '';
    this.suggestedLang = null;
    if (this.recognition) this.recognition.lang = this.resolvedBcp47;
    this.cdr.detectChanges();
    // Small delay to let UI update then auto-start recording
    setTimeout(() => this.startRecording(), 100);
  }

  ngOnDestroy() {
    clearInterval(this.recordingInterval);
    try { this.recognition?.abort(); } catch { /* ignore */ }
  }

  // ── Pointer & Touch handlers ───────────────────────────────────────────────

  onMouseDown(e: MouseEvent) {
    if (this.hasSendText() || this.disabled || this.isProcessing) return;
    e.preventDefault();
    this.isPointerDown = true;
    this.startRecording();
  }

  onMouseUp(e: MouseEvent) {
    if (!this.isPointerDown) return;
    this.isPointerDown = false;
    if (this.isRecording) this.stopRecordingAndSend();
  }

  onMouseLeave(e: MouseEvent) {
    if (!this.isPointerDown) return;   // only cancel if button was held
    this.isPointerDown = false;
    if (this.isRecording) this.cancelRecording();
  }

  onTouchStart(e: TouchEvent) {
    if (this.hasSendText() || this.disabled || this.isProcessing) return;
    e.preventDefault();
    this.isPointerDown = true;
    this.startRecording();
  }

  onTouchEnd(e: TouchEvent) {
    if (!this.isPointerDown) return;
    this.isPointerDown = false;
    if (this.isRecording) this.stopRecordingAndSend();
  }

  onTouchCancel(e: TouchEvent) {
    if (!this.isPointerDown) return;
    this.isPointerDown = false;
    if (this.isRecording) this.cancelRecording();
  }

  handleClick(e: MouseEvent) {
    if (this.hasSendText() && !this.isRecording && !this.isProcessing) this.send();
  }

  // ── Send ──────────────────────────────────────────────────────────────────

  send() {
    if (this.disabled) return;
    const trimmed = this.text.trim();
    if (trimmed) {
      this.text        = '';   // clear FIRST so UI updates before emit side-effects
      this.interimText = '';
      this.cdr.detectChanges(); // force DOM update immediately
      this.onSendText.emit({ text: trimmed, lang: this.currentLang.code });
    }
  }

  // ── Template helpers ──────────────────────────────────────────────────────

  hasSendText(): boolean { return !!this.text.trim(); }

  micTitle(): string {
    if (this.isRecording)    return 'Thả để gửi / Release to send';
    if (this.hasSendText())  return 'Send message';
    return `Giữ để nói (${this.currentLang.label}) / Hold to speak`;
  }

  btnClass(): string {
    if (this.isRecording)  return 'bg-red-500 shadow-red-500/50 text-white w-14 h-14 scale-110 mic-pulse';
    if (this.isProcessing) return 'bg-indigo-400 shadow-indigo-400/30 text-white w-12 h-12 md:w-14 md:h-14 opacity-75 cursor-wait';
    if (this.hasSendText()) return 'bg-gradient-to-br from-indigo-500 to-violet-600 shadow-indigo-500/30 text-white w-12 h-12 md:h-14 md:w-auto px-6 md:px-8';
    return 'bg-gradient-to-br from-indigo-500 to-violet-600 shadow-indigo-500/30 text-white w-12 h-12 md:w-14 md:h-14';
  }
}
