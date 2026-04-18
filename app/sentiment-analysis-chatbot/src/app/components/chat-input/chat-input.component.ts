import { Component, Output, EventEmitter, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import {
  LucideAngularModule,
  Paperclip,
  Smile,
  Mic,
  Send,
  Globe,
  X
} from 'lucide-angular';

@Component({
  selector: 'app-chat-input',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideAngularModule],
  template: `
    <div class="flex items-center gap-4 w-full transition-all duration-500">
      <!-- Premium Glass Input Card -->
      <div
        class="flex-1 glass-panel !bg-white/40 dark:!bg-slate-900/40 rounded-3xl flex items-center px-3 md:px-6 gap-2 md:gap-4 border border-white/20 dark:border-white/5 focus-within:ring-2 focus-within:ring-indigo-500/30 focus-within:bg-white dark:focus-within:bg-slate-900 transition-all duration-300 relative overflow-hidden"
      >
        <!-- Paperclip: opens batch CSV upload modal in parent -->
        <button
          class="text-slate-400 hover:text-indigo-500 transition-all duration-300 active:scale-95"
          (click)="onOpenBatch.emit()"
          [disabled]="disabled || isRecording"
          title="Upload CSV for batch analysis"
        >
          <lucide-angular name="paperclip" [size]="22"></lucide-angular>
        </button>

        <!-- Language Switcher -->
        <button
          class="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-black/5 dark:bg-white/5 hover:bg-indigo-500/10 text-slate-500 dark:text-slate-400 transition-all active:scale-90 border border-transparent hover:border-indigo-500/20"
          (click)="toggleLang()"
          [disabled]="disabled || isRecording"
        >
          <lucide-angular name="globe" [size]="14"></lucide-angular>
          <span class="text-[10px] font-black uppercase tracking-widest">{{ lang }}</span>
        </button>



        <!-- Recording Pulse -->
        <div
          *ngIf="isRecording"
          class="flex-1 flex items-center gap-3 text-red-500 dark:text-red-400 font-medium text-[15.5px]"
        >
          <div
            class="w-2.5 h-2.5 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)] animate-pulse"
          ></div>
          <span class="animate-pulse tracking-tight">Đang ghi âm...</span>
          <span class="text-[10px] font-black bg-red-500/10 px-2 py-0.5 rounded-full">{{ recordingTime }}s</span>
        </div>

        <!-- Normal Input -->
        <input
          *ngIf="!isRecording"
          type="text"
          [(ngModel)]="text"
          (keyup.enter)="send()"
          [disabled]="disabled"
          placeholder="Type text to analyse sentiment..."
          class="flex-1 bg-transparent border-none focus:ring-0 text-[15px] md:text-[15.5px] py-4 md:py-5 outline-none dark:text-white placeholder:text-slate-400 font-medium min-w-0"
        />

        <button
          class="text-slate-400 hover:text-indigo-500 transition-all duration-300 active:scale-95"
          *ngIf="!isRecording"
          [disabled]="disabled"
        >
          <lucide-angular name="smile" [size]="22"></lucide-angular>
        </button>
      </div>

      <!-- Action Button with Ripple -->
      <div class="relative flex items-center justify-center">
        <!-- Ripple Effect Background -->
        <div *ngIf="isRecording" class="absolute inset-0 bg-red-500/20 rounded-full animate-ripple pointer-events-none"></div>
        <div *ngIf="isRecording" class="absolute inset-0 bg-red-500/10 rounded-full animate-ripple [animation-delay:0.5s] pointer-events-none"></div>

        <button
          (mousedown)="startHold($event)"
          (mouseup)="stopHold($event)"
          (mouseleave)="cancelHold($event)"
          (touchstart)="startHold($event)"
          (touchend)="stopHold($event)"
          (touchcancel)="cancelHold($event)"
          (click)="handleClick($event)"
          [disabled]="disabled"
          [ngClass]="
            isRecording
              ? 'bg-red-500 shadow-red-500/50 text-white w-14 h-14 scale-110'
              : 'bg-gradient-to-br from-indigo-500 to-violet-600 shadow-indigo-500/30 text-white ' +
                (text.trim()
                  ? 'w-12 h-12 md:h-14 md:w-auto px-6 md:px-8'
                  : 'w-12 h-12 md:w-14 md:h-14')
          "
          class="relative flex items-center justify-center text-white rounded-[24px] md:rounded-3xl transition-all duration-500 transform active:scale-95 shadow-xl select-none z-10"
        >
          <lucide-angular
            *ngIf="!isRecording && !text.trim()"
            name="mic"
            [size]="24"
          ></lucide-angular>
          <lucide-angular
            *ngIf="isRecording"
            name="mic"
            [size]="28"
          ></lucide-angular>
          <div
            *ngIf="text.trim() && !isRecording"
            class="flex items-center gap-2.5"
          >
            <span class="text-xs font-black uppercase tracking-[0.2em] hidden md:inline">Send</span>
            <lucide-angular name="send" [size]="20"></lucide-angular>
          </div>
        </button>
      </div>
    </div>
  `,
})
export class ChatInputComponent {
  @Input() disabled = false;
  @Output() onSendText = new EventEmitter<{ text: string; lang: 'vi' | 'en' }>();
  @Output() onSendAudio = new EventEmitter<{ audio: Blob; lang: 'vi' | 'en' }>();
  @Output() onOpenBatch = new EventEmitter<void>();  // paperclip → open batch modal

  text = '';
  lang: 'vi' | 'en' = 'en';

  // no selectedFile — file handling moved to batch-upload component

  // Audio state
  isRecording = false;
  recordingTime = 0;
  recordingInterval: any;
  recognition: any;
  transcribedText = '';
  shouldAutoSend = false;

  constructor() {
    this.setupSpeechRecognition();
  }

  setupSpeechRecognition() {
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      this.recognition = new SpeechRecognition();
      this.recognition.continuous = true;
      this.recognition.interimResults = true;
      this.recognition.lang = this.lang === 'vi' ? 'vi-VN' : 'en-US';

      this.recognition.onresult = (event: any) => {
        let interimTranscript = '';
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          } else {
            interimTranscript += event.results[i][0].transcript;
          }
        }

        this.transcribedText = finalTranscript || interimTranscript;
        if (this.transcribedText) {
          this.text = this.transcribedText;
        }
      };

      this.recognition.onerror = (event: any) => {
        console.error('Speech recognition error', event.error);
        this.cleanupRecordingState();
      };

      this.recognition.onend = () => {
        this.cleanupRecordingState();

        // Auto send after recognition fully completes
        if (this.shouldAutoSend && this.text.trim()) {
          this.send();
        }
        this.shouldAutoSend = false;
      };
    }
  }

  isSendMode() {
    return this.text.trim();
  }

  handleClick(e?: Event) {
    if (this.isSendMode()) {
      this.send();
    }
  }

  startHold(e?: Event) {
    if (this.disabled) return;
    if (this.isSendMode()) return;
    this.startRecording();
  }

  stopHold(e?: Event) {
    if (this.isSendMode()) return;
    if (this.isRecording) {
      this.shouldAutoSend = true;
      this.recognition?.stop(); // Ngân ghi âm và đợi onend
      this.cleanupRecordingState();
    }
  }

  cancelHold(e?: Event) {
    if (this.isSendMode()) return;
    if (this.isRecording) {
      this.shouldAutoSend = false; // Bị hủy thì không gửi
      this.recognition?.abort(); // Hủy ngay
      this.cleanupRecordingState();
      this.text = ''; // Clear text if cancelled
    }
  }

  startRecording() {
    if (!this.recognition) {
      alert(
        'Trình duyệt của bạn không hỗ trợ nhận diện giọng nói (Web Speech API). Vui lòng dùng Google Chrome.',
      );
      return;
    }

    try {
      this.transcribedText = '';
      this.text = '';
      this.shouldAutoSend = false;
      this.recognition.lang = this.lang === 'vi' ? 'vi-VN' : 'en-US';
      this.recognition.start();

      this.isRecording = true;
      this.recordingTime = 0;
      this.recordingInterval = setInterval(() => this.recordingTime++, 1000);
    } catch (err) {
      console.error('Microphone API error:', err);
    }
  }

  cleanupRecordingState() {
    if (this.isRecording) {
      this.isRecording = false;
      clearInterval(this.recordingInterval);
    }
  }

  send() {
    if (this.disabled) return;
    if (this.text.trim()) {
      this.onSendText.emit({ text: this.text.trim(), lang: this.lang });
      this.text = '';
    }
  }

  toggleLang() {
    this.lang = this.lang === 'vi' ? 'en' : 'vi';
    if (this.recognition) {
      this.recognition.lang = this.lang === 'vi' ? 'vi-VN' : 'en-US';
    }
  }
}
