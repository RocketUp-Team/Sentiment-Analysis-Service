import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Message, AspectSentiment } from '../../models/message.model';
import { LucideAngularModule, CheckCheck, Loader2, AlertTriangle, Clock, BarChart2, AlertCircle } from 'lucide-angular';

@Component({
  selector: 'app-message-bubble',
  standalone: true,
  imports: [CommonModule, LucideAngularModule],
  template: `
    <div [ngClass]="['flex w-full mb-3 group', isUser ? 'justify-end' : 'justify-start']">
      <div [ngClass]="[
        'message-bubble relative group hover:scale-[1.01] transition-all duration-300',
        isUser ? 'user-bubble' : 'bot-bubble'
      ]">

        <!-- Message Content: render bold **text** -->
        <div class="space-y-3">
          <p class="pr-10 leading-relaxed text-[15.5px] font-medium tracking-normal whitespace-pre-wrap"
             [innerHTML]="formattedText">
          </p>

          <!-- ── Bot result badges ── -->
          <div *ngIf="!isUser && message.sentiment && message.sentiment !== 'LOADING' && message.sentiment !== 'ERROR'"
               class="pt-3 mt-1 border-t border-black/[0.04] dark:border-white/[0.04] space-y-3">

            <!-- Row 1: sentiment pill + confidence + latency -->
            <div class="flex flex-wrap items-center gap-2">
              <!-- Sentiment pill -->
              <div [ngClass]="['flex items-center gap-2 px-3 py-1 rounded-full border', getSentimentBadgeClass()]">
                <div [ngClass]="getSentimentGlow()" class="w-1.5 h-1.5 rounded-full shadow-lg"></div>
                <span class="text-[9px] font-black uppercase tracking-[0.15em]">{{ message.sentiment }}</span>
              </div>

              <!-- Confidence -->
              <div *ngIf="message.confidenceScore"
                   class="px-2.5 py-1 rounded-full bg-black/5 dark:bg-white/5 border border-black/5 dark:border-white/5">
                <span class="text-[9px] font-bold text-slate-500 uppercase tracking-widest">
                  Confidence: {{ message.confidenceScore * 100 | number:'1.1-1' }}%
                </span>
              </div>

              <!-- Latency -->
              <div *ngIf="message.latency_ms"
                   class="flex items-center gap-1 px-2.5 py-1 rounded-full bg-black/5 dark:bg-white/5 border border-black/5 dark:border-white/5">
                <lucide-angular name="clock" [size]="10" class="text-slate-400"></lucide-angular>
                <span class="text-[9px] font-bold text-slate-500 uppercase tracking-widest">
                  {{ message.latency_ms | number:'1.0-0' }} ms
                </span>
              </div>

              <!-- Sarcasm flag -->
              <div *ngIf="message.sarcasm_flag"
                   class="flex items-center gap-1 px-2.5 py-1 rounded-full bg-amber-500/10 border border-amber-500/20">
                <lucide-angular name="alert-triangle" [size]="10" class="text-amber-500"></lucide-angular>
                <span class="text-[9px] font-black text-amber-600 uppercase tracking-widest">Sarcasm</span>
              </div>
            </div>

            <!-- Row 2: Aspect badges -->
            <div *ngIf="message.aspects && message.aspects.length > 0" class="space-y-1.5">
              <p class="text-[9px] font-black uppercase tracking-[0.2em] text-slate-400 flex items-center gap-1">
                <lucide-angular name="bar-chart-2" [size]="10"></lucide-angular>
                Aspect Breakdown
              </p>
              <div class="flex flex-wrap gap-2">
                <div *ngFor="let asp of message.aspects"
                     [ngClass]="['flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-[10px] font-bold', getAspectClass(asp.sentiment)]">
                  <span class="text-[8px]">{{ getAspectIcon(asp.sentiment) }}</span>
                  <span class="capitalize">{{ asp.aspect }}</span>
                  <span class="opacity-60">{{ asp.confidence * 100 | number:'1.0-0' }}%</span>
                </div>
              </div>
            </div>

            <!-- No aspects note -->
            <div *ngIf="!message.aspects || message.aspects.length === 0"
                 class="flex items-center gap-1.5">
              <lucide-angular name="alert-circle" [size]="11" class="text-slate-400"></lucide-angular>
              <span class="text-[10px] text-slate-400 italic">No aspects detected above threshold.</span>
            </div>
          </div>
        </div>

        <!-- Loading animation -->
        <div *ngIf="message.sentiment === 'LOADING'" class="mt-4 flex items-center gap-3">
          <div class="flex gap-1">
            <span class="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
            <span class="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
            <span class="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-bounce"></span>
          </div>
          <span class="text-[9px] font-black uppercase tracking-[0.2em] text-indigo-500 italic">Neural Processing...</span>
        </div>

        <!-- Error state -->
        <div *ngIf="message.sentiment === 'ERROR'" class="mt-2 flex items-center gap-2">
          <lucide-angular name="alert-circle" [size]="14" class="text-rose-400 shrink-0"></lucide-angular>
          <span class="text-[10px] text-rose-400 font-bold">API Error</span>
        </div>

        <!-- Timestamp -->
        <div class="absolute bottom-2.5 right-3.5 flex items-center gap-2 select-none opacity-0 group-hover:opacity-100 transition-all duration-300 translate-y-1 group-hover:translate-y-0">
          <span class="text-[9px] font-black tabular-nums tracking-widest uppercase opacity-40">
            {{ message.timestamp | date:'HH:mm' }}
          </span>
          <lucide-angular *ngIf="isUser" name="check-check" [size]="14" class="text-indigo-300 dark:text-indigo-400"></lucide-angular>
        </div>
      </div>
    </div>
  `,
})
export class MessageBubbleComponent {
  @Input({ required: true }) message!: Message;

  get isUser(): boolean {
    return this.message.sender === 'user';
  }

  /** Render **bold** markdown fragments as <strong> */
  get formattedText(): string {
    return (this.message.text ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  }

  getSentimentBadgeClass(): string {
    switch (this.message.sentiment) {
      case 'POSITIVE': return 'bg-emerald-500/10 text-emerald-600 border-emerald-500/20';
      case 'NEGATIVE': return 'bg-rose-500/10 text-rose-600 border-rose-500/20';
      case 'NEUTRAL':  return 'bg-amber-500/10 text-amber-600 border-amber-500/20';
      default:         return 'bg-slate-500/10 text-slate-600 border-slate-500/20';
    }
  }

  getSentimentGlow(): string {
    switch (this.message.sentiment) {
      case 'POSITIVE': return 'bg-emerald-500 shadow-[0_0_12px_rgba(16,185,129,0.5)]';
      case 'NEGATIVE': return 'bg-rose-500 shadow-[0_0_12px_rgba(244,63,94,0.5)]';
      case 'NEUTRAL':  return 'bg-amber-500 shadow-[0_0_12px_rgba(245,158,11,0.5)]';
      default:         return 'bg-slate-400';
    }
  }

  getAspectClass(sentiment: string): string {
    switch (sentiment) {
      case 'positive': return 'bg-emerald-500/10 text-emerald-700 border-emerald-500/20';
      case 'negative': return 'bg-rose-500/10 text-rose-700 border-rose-500/20';
      default:         return 'bg-slate-500/10 text-slate-600 border-slate-500/20';
    }
  }

  getAspectIcon(sentiment: string): string {
    switch (sentiment) {
      case 'positive': return '✅';
      case 'negative': return '❌';
      default:         return '➖';
    }
  }
}
