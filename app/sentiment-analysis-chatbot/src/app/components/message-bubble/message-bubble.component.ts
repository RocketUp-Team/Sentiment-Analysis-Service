import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Message } from '../../models/message.model';
import { LucideAngularModule, CheckCheck, Smile, Frown, Meh, Loader2, AlertCircle } from 'lucide-angular';

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
        <!-- Message Content -->
        <div class="space-y-3">
          <p class="pr-10 leading-relaxed text-[15.5px] font-medium tracking-normal whitespace-pre-wrap">
            {{ message.text }}
          </p>
          
          <!-- Premium Bot Insights Section -->
          <div *ngIf="!isUser && message.sentiment && message.sentiment !== 'LOADING'" 
               class="flex flex-wrap items-center gap-2.5 pt-3 mt-1 border-t border-black/[0.03] dark:border-white/[0.03]">
             
             <div [ngClass]="['flex items-center gap-2 px-3 py-1 rounded-full border', getSentimentBadgeClass()]">
                <div [ngClass]="getSentimentGlow()" class="w-1.5 h-1.5 rounded-full shadow-lg"></div>
                <span class="text-[9px] font-black uppercase tracking-[0.15em] opacity-90">
                  {{ message.sentiment }}
                </span>
             </div>

             <div *ngIf="message.confidenceScore" class="px-2.5 py-1 rounded-full bg-black/5 dark:bg-white/5 border border-black/5 dark:border-white/5">
                <span class="text-[9px] font-bold text-slate-500 uppercase tracking-widest">
                  Accuracy: {{ message.confidenceScore * 100 | number:'1.0-0' }}%
                </span>
             </div>
          </div>
        </div>

        <!-- Bot Processing State -->
        <div *ngIf="message.sentiment === 'LOADING'" class="mt-4 flex items-center gap-3">
           <div class="flex gap-1">
              <span class="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
              <span class="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
              <span class="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-bounce"></span>
           </div>
           <span class="text-[9px] font-black uppercase tracking-[0.2em] text-indigo-500 italic">Neural Processing...</span>
        </div>

        <!-- Bottom Meta Layer -->
        <div class="absolute bottom-2.5 right-3.5 flex items-center gap-2 select-none opacity-0 group-hover:opacity-100 transition-all duration-300 translate-y-1 group-hover:translate-y-0">
          <span class="text-[9px] font-black tabular-nums tracking-widest uppercase opacity-40">
            {{ message.timestamp | date:'HH:mm' }}
          </span>
          <lucide-angular 
            *ngIf="isUser" 
            name="check-check" 
            [size]="14" 
            class="text-indigo-300 dark:text-indigo-400">
          </lucide-angular>
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

  getSentimentBadgeClass(): string {
    switch (this.message.sentiment) {
      case 'POSITIVE': return 'bg-emerald-500/10 text-emerald-600 border-emerald-500/20';
      case 'NEGATIVE': return 'bg-rose-500/10 text-rose-600 border-rose-500/20';
      case 'NEUTRAL': return 'bg-amber-500/10 text-amber-600 border-amber-500/20';
      default: return 'bg-slate-500/10 text-slate-600 border-slate-500/20';
    }
  }

  getSentimentGlow(): string {
     switch (this.message.sentiment) {
      case 'POSITIVE': return 'bg-emerald-500 shadow-[0_0_12px_rgba(16,185,129,0.5)]';
      case 'NEGATIVE': return 'bg-rose-500 shadow-[0_0_12px_rgba(244,63,94,0.5)]';
      case 'NEUTRAL': return 'bg-amber-500 shadow-[0_0_12px_rgba(245,158,11,0.5)]';
      default: return 'bg-slate-400';
    }
  }
}

