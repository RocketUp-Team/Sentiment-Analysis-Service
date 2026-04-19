import { Component, Input, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Message, AspectSentiment, ExplainResponse } from '../../models/message.model';
import { SentimentAnalysisService } from '../../services/sentiment-analysis.service';
import {
  LucideAngularModule,
  CheckCheck, Loader2, AlertTriangle, Clock, BarChart2, AlertCircle, Zap, ChevronDown, ChevronUp,
} from 'lucide-angular';

@Component({
  selector: 'app-message-bubble',
  standalone: true,
  imports: [CommonModule, LucideAngularModule],
  styles: [`
    /* ── SHAP bar chart ── */
    .shap-bar {
      height: 8px;
      border-radius: 99px;
      transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
      min-width: 2px;
    }
    .shap-pos { background: linear-gradient(90deg, #10b981, #34d399); }
    .shap-neg { background: linear-gradient(90deg, #f43f5e, #fb7185); }

    /* ── Explain panel slide-in ── */
    .explain-panel {
      animation: slideDown 0.25s cubic-bezier(0.4,0,0.2,1);
    }
    @keyframes slideDown {
      from { opacity: 0; transform: translateY(-6px); }
      to   { opacity: 1; transform: translateY(0); }
    }
  `],
  template: `
    <div [ngClass]="['flex w-full mb-3 group', isUser ? 'justify-end' : 'justify-start']">
      <div [ngClass]="[
        'message-bubble relative group hover:scale-[1.01] transition-all duration-300',
        isUser ? 'user-bubble' : 'bot-bubble'
      ]">

        <!-- Message Content -->
        <div class="space-y-3">
          <p class="pr-10 leading-relaxed text-[15.5px] font-medium tracking-normal whitespace-pre-wrap"
             [innerHTML]="formattedText">
          </p>

          <!-- ── Bot result badges ── -->
          <div *ngIf="!isUser && message.sentiment && message.sentiment !== 'LOADING' && message.sentiment !== 'ERROR'"
               class="pt-3 mt-1 border-t border-black/[0.04] dark:border-white/[0.04] space-y-3">

            <!-- Row 1: sentiment pill + confidence + latency -->
            <div class="flex flex-wrap items-center gap-2">
              <div [ngClass]="['flex items-center gap-2 px-3 py-1 rounded-full border', getSentimentBadgeClass()]">
                <div [ngClass]="getSentimentGlow()" class="w-1.5 h-1.5 rounded-full shadow-lg"></div>
                <span class="text-[9px] font-black uppercase tracking-[0.15em]">{{ message.sentiment }}</span>
              </div>

              <div *ngIf="message.confidenceScore"
                   class="px-2.5 py-1 rounded-full bg-black/5 dark:bg-white/5 border border-black/5 dark:border-white/5">
                <span class="text-[9px] font-bold text-slate-500 uppercase tracking-widest">
                  Confidence: {{ message.confidenceScore * 100 | number:'1.1-1' }}%
                </span>
              </div>

              <div *ngIf="message.latency_ms"
                   class="flex items-center gap-1 px-2.5 py-1 rounded-full bg-black/5 dark:bg-white/5 border border-black/5 dark:border-white/5">
                <lucide-angular name="clock" [size]="10" class="text-slate-400"></lucide-angular>
                <span class="text-[9px] font-bold text-slate-500 uppercase tracking-widest">
                  {{ message.latency_ms | number:'1.0-0' }} ms
                </span>
              </div>

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

            <div *ngIf="!message.aspects || message.aspects.length === 0"
                 class="flex items-center gap-1.5">
              <lucide-angular name="alert-circle" [size]="11" class="text-slate-400"></lucide-angular>
              <span class="text-[10px] text-slate-400 italic">No aspects detected above threshold.</span>
            </div>

            <!-- ── Explain Button ── -->
            <div class="flex items-center gap-2 pt-1">
              <button
                (click)="toggleExplain()"
                [disabled]="message.explainLoading"
                class="flex items-center gap-1.5 px-3 py-1 rounded-full text-[9px] font-black uppercase tracking-widest transition-all active:scale-95 border"
                [ngClass]="message.explainData
                  ? 'bg-violet-500/10 text-violet-600 border-violet-500/20 hover:bg-violet-500/20'
                  : 'bg-indigo-500/8 text-indigo-600 dark:text-indigo-400 border-indigo-500/15 hover:bg-indigo-500/15'"
              >
                <ng-container *ngIf="!message.explainLoading">
                  <lucide-angular name="zap" [size]="9"></lucide-angular>
                  <span>{{ message.explainData ? (showExplain ? 'Hide' : 'Show') + ' Explain' : 'Explain' }}</span>
                  <lucide-angular *ngIf="message.explainData && showExplain" name="chevron-up" [size]="9"></lucide-angular>
                  <lucide-angular *ngIf="message.explainData && !showExplain" name="chevron-down" [size]="9"></lucide-angular>
                </ng-container>
                <ng-container *ngIf="message.explainLoading">
                  <lucide-angular name="loader-2" [size]="9" class="animate-spin"></lucide-angular>
                  <span>Explaining...</span>
                </ng-container>
              </button>
            </div>

            <!-- ── SHAP Explanation Panel ── -->
            <div *ngIf="message.explainData && showExplain"
                 class="explain-panel mt-2 p-4 rounded-2xl bg-black/[0.03] dark:bg-white/[0.03] border border-black/[0.05] dark:border-white/[0.05] space-y-3">

              <div class="flex items-center justify-between">
                <p class="text-[9px] font-black uppercase tracking-[0.2em] text-indigo-500 flex items-center gap-1">
                  <lucide-angular name="zap" [size]="10"></lucide-angular>
                  SHAP Token Influence
                </p>
                <span class="text-[9px] text-slate-400 font-bold">
                  Base: {{ message.explainData.base_value | number:'1.3-3' }}
                  · {{ message.explainData.latency_ms | number:'1.0-0' }} ms
                </span>
              </div>

              <!-- Token bars -->
              <div class="space-y-1.5">
                <div *ngFor="let item of shapItems(message.explainData)" class="flex items-center gap-2">
                  <!-- Token label -->
                  <span class="text-[10px] font-mono font-bold text-slate-600 dark:text-slate-300 w-20 truncate shrink-0 text-right"
                        [title]="item.token">{{ item.token }}</span>

                  <!-- Bar (centered pivot at 0) -->
                  <div class="flex-1 flex items-center h-5 relative">
                    <!-- Zero line -->
                    <div class="absolute left-1/2 w-px h-4 bg-black/10 dark:bg-white/10"></div>

                    <!-- Positive bar (right of center) -->
                    <div *ngIf="item.value >= 0"
                         class="shap-bar shap-pos absolute"
                         [style.left.%]="50"
                         [style.width.%]="item.widthPct">
                    </div>
                    <!-- Negative bar (left of center, flipped) -->
                    <div *ngIf="item.value < 0"
                         class="shap-bar shap-neg absolute"
                         [style.right.%]="50"
                         [style.width.%]="item.widthPct">
                    </div>
                  </div>

                  <!-- Value label -->
                  <span class="text-[9px] font-black font-mono w-14 shrink-0"
                        [class.text-emerald-600]="item.value >= 0"
                        [class.text-rose-500]="item.value < 0">
                    {{ item.value >= 0 ? '+' : '' }}{{ item.value | number:'1.3-3' }}
                  </span>
                </div>
              </div>

              <!-- Legend -->
              <div class="flex items-center gap-4 pt-1">
                <div class="flex items-center gap-1.5">
                  <div class="w-3 h-2 rounded-full bg-gradient-to-r from-emerald-500 to-emerald-400"></div>
                  <span class="text-[9px] font-bold text-slate-400">Pushes positive</span>
                </div>
                <div class="flex items-center gap-1.5">
                  <div class="w-3 h-2 rounded-full bg-gradient-to-r from-rose-500 to-rose-400"></div>
                  <span class="text-[9px] font-bold text-slate-400">Pushes negative</span>
                </div>
              </div>
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

  private sentimentService = inject(SentimentAnalysisService);

  /** Controls whether the SHAP panel is expanded */
  showExplain = false;

  get isUser(): boolean { return this.message.sender === 'user'; }

  /** Render **bold** fragments as <strong> */
  get formattedText(): string {
    return (this.message.text ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  }

  // ── Explain button handler ───────────────────────────────────────────────

  toggleExplain() {
    if (this.message.explainData) {
      this.showExplain = !this.showExplain;
      return;
    }
    this.showExplain = true;
    // Service looks up the preceding user message automatically
    this.sentimentService.explainMessage(this.message.id);
  }

  // ── SHAP visualization helpers ───────────────────────────────────────────

  /**
   * Returns tokens sorted by absolute SHAP value (highest first),
   * with widthPct normalised to 0-45 (leaves room for pivot + label).
   */
  shapItems(data: ExplainResponse): Array<{ token: string; value: number; widthPct: number }> {
    const maxAbs = Math.max(...data.shap_values.map(Math.abs), 1e-9);
    return data.tokens
      .map((token, i) => ({
        token,
        value: data.shap_values[i],
        widthPct: (Math.abs(data.shap_values[i]) / maxAbs) * 45,
      }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  }

  // ── Badge helpers ────────────────────────────────────────────────────────

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
