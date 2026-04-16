import { Component, Output, EventEmitter, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule, Paperclip, Smile, Mic, Send } from 'lucide-angular';

@Component({
  selector: 'app-chat-input',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideAngularModule],
  template: `
    <div class="flex items-center gap-4 w-full transition-all duration-500">
      
      <!-- Premium Glass Input Card -->
      <div class="flex-1 glass-panel !bg-white/40 dark:!bg-slate-900/40 rounded-3xl flex items-center px-3 md:px-6 gap-2 md:gap-4 border border-white/20 dark:border-white/5 focus-within:ring-2 focus-within:ring-indigo-500/30 focus-within:bg-white dark:focus-within:bg-slate-900 transition-all duration-300">
         
         <button class="text-slate-400 hover:text-indigo-500 transition-all duration-300 active:scale-95" (click)="fileInput.click()" [disabled]="disabled || isRecording">
            <lucide-angular name="paperclip" [size]="22"></lucide-angular>
         </button>
         <input type="file" #fileInput hidden (change)="onFileSelected($event)" accept=".pdf,.doc,.docx" [disabled]="disabled || isRecording">
         
         <!-- File Badge -->
         <div *ngIf="selectedFile" class="flex-1 flex items-center bg-indigo-50/50 dark:bg-indigo-900/40 px-3 py-2 rounded-xl my-2">
            <div class="flex-1 truncate text-sm text-indigo-700 dark:text-indigo-300 font-medium whitespace-nowrap overflow-hidden text-ellipsis mr-2">
               📄 {{ selectedFile.name }}
            </div>
            <button (click)="removeFile()" class="text-slate-400 hover:text-red-500 transition-colors">
               <lucide-angular name="x" [size]="16"></lucide-angular>
            </button>
         </div>

         <!-- Recording Pulse -->
         <div *ngIf="isRecording" class="flex-1 flex items-center gap-3 text-red-500 dark:text-red-400 font-medium text-[15.5px]">
            <div class="w-2.5 h-2.5 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)] animate-pulse"></div>
            Đang ghi âm... {{ recordingTime }}s
         </div>

         <!-- Normal Input -->
         <input
           *ngIf="!selectedFile && !isRecording"
           type="text"
           [(ngModel)]="text"
           (keydown.enter)="send()"
           [disabled]="disabled"
           placeholder="Deep analyze sentiment..."
           class="flex-1 bg-transparent border-none focus:ring-0 text-[15px] md:text-[15.5px] py-4 md:py-5 outline-none dark:text-white placeholder:text-slate-400 font-medium min-w-0"
         />

         <button class="text-slate-400 hover:text-indigo-500 transition-all duration-300 active:scale-95" *ngIf="!isRecording" [disabled]="disabled">
            <lucide-angular name="smile" [size]="22"></lucide-angular>
         </button>
      </div>

      <!-- Action Button with Gradient & Depth -->
      <button
        (mousedown)="startHold($event)"
        (mouseup)="stopHold($event)"
        (mouseleave)="cancelHold($event)"
        (touchstart)="startHold($event)"
        (touchend)="stopHold($event)"
        (touchcancel)="cancelHold($event)"
        (click)="handleClick($event)"
        [disabled]="disabled"
        [ngClass]="isRecording ? 'bg-red-500 dark:bg-red-500 shadow-red-500/50 text-white w-12 h-12 md:w-14 md:h-14 scale-110' : ('bg-gradient-to-br from-indigo-500 to-violet-600 shadow-indigo-500/30 text-white ' + ((text.trim() || selectedFile) ? 'w-12 h-12 md:h-14 md:w-auto px-4 md:px-7' : 'w-12 h-12 md:w-14 md:h-14'))"
        class="flex items-center justify-center text-white rounded-[24px] md:rounded-3xl transition-all duration-300 transform active:scale-95 shadow-xl select-none shrink-0"
      >
        <lucide-angular *ngIf="!text.trim() && !selectedFile && !isRecording" name="mic" [size]="24"></lucide-angular>
        <lucide-angular *ngIf="isRecording" name="mic" [size]="24" class="animate-pulse"></lucide-angular>
        <div *ngIf="(text.trim() || selectedFile) && !isRecording" class="flex items-center gap-2.5">
            <span class="text-xs font-black uppercase tracking-[0.2em] hidden md:inline">Send</span>
            <lucide-angular name="send" [size]="20" class="ml-0.5"></lucide-angular>
        </div>
      </button>
    </div>
  `,
})
export class ChatInputComponent {
  @Input() disabled = false;
  @Output() onSendText = new EventEmitter<{ text: string, lang: 'vi' | 'en' }>();
  @Output() onSendFile = new EventEmitter<{ file: File, lang: 'vi' | 'en' }>();
  @Output() onSendAudio = new EventEmitter<{ audio: Blob, lang: 'vi' | 'en' }>();

  text = '';
  lang: 'vi' | 'en' = 'vi';
  
  selectedFile: File | null = null;
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
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
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

  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
      this.text = ''; 
      input.value = ''; 
    }
  }

  removeFile() {
    this.selectedFile = null;
  }

  isSendMode() {
    return this.text.trim() || this.selectedFile;
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
      alert('Trình duyệt của bạn không hỗ trợ nhận diện giọng nói (Web Speech API). Vui lòng dùng Google Chrome.');
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
    
    if (this.selectedFile) {
      this.onSendFile.emit({ file: this.selectedFile, lang: this.lang });
      this.selectedFile = null;
    } else if (this.text.trim()) {
      this.onSendText.emit({ text: this.text.trim(), lang: this.lang });
      this.text = '';
    }
  }
}
