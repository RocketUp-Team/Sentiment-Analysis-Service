import { Component, Output, EventEmitter, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SentimentAnalysisService } from '../../services/sentiment-analysis.service';
import { BatchPredictResponse, BatchItemResult } from '../../models/message.model';
import { LucideAngularModule, Upload, X, FileText, CheckCircle, XCircle, AlertCircle, Download, Loader2, BarChart2, ChevronDown, ChevronUp } from 'lucide-angular';

@Component({
  selector: 'app-batch-upload',
  standalone: true,
  imports: [CommonModule, LucideAngularModule],
  template: `
    <!-- Overlay backdrop -->
    <div class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in"
         (click)="onClose.emit()">

      <!-- Panel -->
      <div class="relative w-full max-w-4xl max-h-[90vh] flex flex-col rounded-[2rem] bg-white dark:bg-slate-900 shadow-2xl shadow-black/40 overflow-hidden animate-in slide-in-from-bottom-4 duration-300"
           (click)="$event.stopPropagation()">

        <!-- Header -->
        <div class="flex items-center justify-between px-8 py-6 border-b border-black/5 dark:border-white/5">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-2xl bg-indigo-500 flex items-center justify-center text-white shadow-lg shadow-indigo-500/30">
              <lucide-angular name="file-text" [size]="20"></lucide-angular>
            </div>
            <div>
              <h2 class="text-lg font-black tracking-tight dark:text-white">Batch CSV Predict</h2>
              <p class="text-[11px] text-slate-500 font-medium">Upload a CSV with a <code class="bg-black/5 dark:bg-white/10 px-1 rounded">text</code> column · max 500 rows</p>
            </div>
          </div>
          <button (click)="onClose.emit()"
                  class="p-2 text-slate-400 hover:text-rose-500 hover:bg-rose-500/10 rounded-2xl transition-all">
            <lucide-angular name="x" [size]="22"></lucide-angular>
          </button>
        </div>

        <!-- Body -->
        <div class="flex-1 overflow-y-auto px-8 py-6 space-y-6">

          <!-- Drop zone -->
          <div *ngIf="!result() && !loading()"
               class="relative rounded-3xl border-2 border-dashed border-indigo-300 dark:border-indigo-700 bg-indigo-500/5 hover:bg-indigo-500/10 transition-all group"
               (dragover)="$event.preventDefault()"
               (drop)="onDrop($event)">

            <!--
              Full-cover transparent input — directly clickable by the user.
              NO programmatic .click(), NO label[for], NO hidden attr,
              NO pointer-events:none.  This works on macOS/Chrome/Safari.
            -->
            <input
              type="file"
              (change)="onFileSelect($event)"
              class="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
            />

            <div class="relative z-0 flex flex-col items-center justify-center py-14 gap-4 pointer-events-none">
              <div class="w-16 h-16 rounded-3xl bg-indigo-500/10 flex items-center justify-center group-hover:scale-110 transition-transform">
                <lucide-angular name="upload" [size]="28" class="text-indigo-500"></lucide-angular>
              </div>
              <div class="text-center">
                <p class="text-base font-bold dark:text-white">Drop CSV here or <span class="text-indigo-500">browse</span></p>
                <p class="text-xs text-slate-500 mt-1">Must contain a <strong>text</strong> column · max 500 rows · UTF-8</p>
              </div>
              <div *ngIf="selectedFile()" class="flex items-center gap-2 px-4 py-2 bg-indigo-500/10 rounded-2xl border border-indigo-500/20">
                <lucide-angular name="file-text" [size]="16" class="text-indigo-500"></lucide-angular>
                <span class="text-sm font-bold text-indigo-600 dark:text-indigo-400">{{ selectedFile()!.name }}</span>
                <span class="text-xs text-slate-500">({{ fileSizeLabel() }})</span>
              </div>
            </div>
          </div>


          <!-- Loading state -->
          <div *ngIf="loading()" class="flex flex-col items-center justify-center py-16 gap-6">
            <div class="w-20 h-20 rounded-[2rem] bg-indigo-500/10 flex items-center justify-center">
              <lucide-angular name="loader-2" [size]="36" class="text-indigo-500 animate-spin"></lucide-angular>
            </div>
            <div class="text-center">
              <p class="font-bold dark:text-white">Processing CSV...</p>
              <p class="text-sm text-slate-500 mt-1">Running predictions on each row. This may take a moment.</p>
            </div>
          </div>

          <!-- Results -->
          <div *ngIf="result()">
            <!-- Summary cards -->
            <div class="grid grid-cols-4 gap-3 mb-6">
              <div class="rounded-2xl bg-slate-50 dark:bg-slate-800 p-4 text-center">
                <p class="text-2xl font-black dark:text-white">{{ result()!.total_items }}</p>
                <p class="text-[10px] font-bold text-slate-500 uppercase tracking-widest mt-1">Total Rows</p>
              </div>
              <div class="rounded-2xl bg-emerald-50 dark:bg-emerald-900/30 p-4 text-center">
                <p class="text-2xl font-black text-emerald-600">{{ result()!.processed_items }}</p>
                <p class="text-[10px] font-bold text-emerald-600 uppercase tracking-widest mt-1">Processed</p>
              </div>
              <div class="rounded-2xl bg-rose-50 dark:bg-rose-900/30 p-4 text-center">
                <p class="text-2xl font-black text-rose-500">{{ result()!.failed_items }}</p>
                <p class="text-[10px] font-bold text-rose-500 uppercase tracking-widest mt-1">Failed</p>
              </div>
              <div class="rounded-2xl bg-indigo-50 dark:bg-indigo-900/30 p-4 text-center">
                <p class="text-2xl font-black text-indigo-600">{{ result()!.latency_ms.toFixed(0) }}<span class="text-sm font-bold">ms</span></p>
                <p class="text-[10px] font-bold text-indigo-600 uppercase tracking-widest mt-1">Total time</p>
              </div>
            </div>

            <!-- Sentiment distribution -->
            <div class="grid grid-cols-3 gap-3 mb-6">
              <div *ngFor="let s of sentimentSummary()" [ngClass]="s.barClass"
                   class="rounded-2xl p-4 border">
                <div class="flex items-center justify-between mb-2">
                  <span [ngClass]="s.labelClass" class="text-[11px] font-black uppercase tracking-widest">{{ s.label }}</span>
                  <span [ngClass]="s.labelClass" class="font-black text-xl">{{ s.count }}</span>
                </div>
                <div class="w-full h-1.5 bg-black/10 dark:bg-white/10 rounded-full overflow-hidden">
                  <div [ngClass]="s.fillClass" class="h-full rounded-full transition-all"
                       [style.width]="s.pct + '%'"></div>
                </div>
                <p [ngClass]="s.labelClass" class="text-[10px] font-bold mt-1 opacity-70">{{ s.pct.toFixed(1) }}% of total</p>
              </div>
            </div>

            <!-- Results table -->
            <div class="rounded-3xl border border-black/5 dark:border-white/5 overflow-hidden">
              <div class="flex items-center justify-between px-5 py-3 bg-black/2 dark:bg-white/2 border-b border-black/5 dark:border-white/5">
                <p class="text-[11px] font-black uppercase tracking-widest text-slate-500 flex items-center gap-1.5">
                  <lucide-angular name="bar-chart-2" [size]="12"></lucide-angular>
                  Results ({{ result()!.results.length }} rows)
                </p>
                <button (click)="downloadCSV()"
                        class="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-indigo-500 text-white text-[11px] font-black uppercase tracking-widest hover:bg-indigo-600 active:scale-95 transition-all">
                  <lucide-angular name="download" [size]="12"></lucide-angular>
                  Export CSV
                </button>
              </div>
              <div class="overflow-x-auto max-h-[320px] overflow-y-auto">
                <table class="w-full text-sm">
                  <thead class="sticky top-0 bg-white dark:bg-slate-900 border-b border-black/5 dark:border-white/5">
                    <tr>
                      <th class="text-left px-4 py-2.5 text-[10px] font-black uppercase tracking-widest text-slate-500 w-12">#</th>
                      <th class="text-left px-4 py-2.5 text-[10px] font-black uppercase tracking-widest text-slate-500">Text</th>
                      <th class="text-left px-4 py-2.5 text-[10px] font-black uppercase tracking-widest text-slate-500 w-28">Sentiment</th>
                      <th class="text-left px-4 py-2.5 text-[10px] font-black uppercase tracking-widest text-slate-500 w-24">Confidence</th>
                      <th class="text-left px-4 py-2.5 text-[10px] font-black uppercase tracking-widest text-slate-500 w-12">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr *ngFor="let item of result()!.results"
                        class="border-b border-black/[0.03] dark:border-white/[0.03] hover:bg-black/[0.02] dark:hover:bg-white/[0.02] transition-colors">
                      <td class="px-4 py-3 text-xs text-slate-400 font-mono">{{ item.row }}</td>
                      <td class="px-4 py-3 text-xs text-slate-700 dark:text-slate-300 max-w-[280px] truncate">{{ item.text }}</td>
                      <td class="px-4 py-3">
                        <span [ngClass]="getSentimentClass(item.sentiment)"
                              class="px-2.5 py-0.5 rounded-full text-[10px] font-black uppercase tracking-widest border">
                          {{ item.sentiment }}
                        </span>
                      </td>
                      <td class="px-4 py-3 text-xs font-bold text-slate-600 dark:text-slate-400">
                        {{ (item.confidence * 100).toFixed(1) }}%
                      </td>
                      <td class="px-4 py-3">
                        <lucide-angular *ngIf="!item.error" name="check-circle" [size]="16" class="text-emerald-500"></lucide-angular>
                        <lucide-angular *ngIf="item.error" name="x-circle" [size]="16" class="text-rose-500"></lucide-angular>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <!-- Error -->
          <div *ngIf="errorMsg()" class="flex items-center gap-3 p-4 rounded-2xl bg-rose-50 dark:bg-rose-900/20 border border-rose-200 dark:border-rose-800">
            <lucide-angular name="alert-circle" [size]="20" class="text-rose-500 shrink-0"></lucide-angular>
            <p class="text-sm text-rose-600 dark:text-rose-400 font-medium">{{ errorMsg() }}</p>
          </div>
        </div>

        <!-- Footer -->
        <div class="px-8 py-4 border-t border-black/5 dark:border-white/5 flex items-center justify-between">
          <button *ngIf="result()" (click)="reset()"
                  class="px-5 py-2.5 rounded-2xl bg-black/5 dark:bg-white/5 text-slate-600 dark:text-slate-400 text-sm font-bold hover:bg-black/10 transition-all active:scale-95">
            Upload Another
          </button>
          <span *ngIf="!result()" class="text-xs text-slate-400">Supported: .csv · UTF-8</span>
          <button *ngIf="selectedFile() && !result() && !loading()" (click)="runBatch()"
                  class="px-6 py-2.5 rounded-2xl bg-indigo-500 text-white font-bold text-sm hover:bg-indigo-600 active:scale-95 transition-all shadow-lg shadow-indigo-500/30">
            Run Batch Analysis
          </button>
        </div>
      </div>
    </div>
  `,
})
export class BatchUploadComponent {
  @Output() onClose = new EventEmitter<void>();

  private service = inject(SentimentAnalysisService);

  selectedFile  = signal<File | null>(null);
  loading       = signal(false);
  result        = signal<BatchPredictResponse | null>(null);
  errorMsg      = signal<string>('');

  fileSizeLabel(): string {
    const f = this.selectedFile();
    if (!f) return '';
    return f.size > 1024 * 1024
      ? (f.size / 1024 / 1024).toFixed(1) + ' MB'
      : (f.size / 1024).toFixed(0) + ' KB';
  }

  onFileSelect(e: Event): void {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.csv')) {
      this.errorMsg.set(`"${file.name}" is not a CSV file. Please select a .csv file.`);
      (e.target as HTMLInputElement).value = '';
      return;
    }
    this.errorMsg.set('');
    this.selectedFile.set(file);
  }

  onDrop(e: DragEvent): void {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    if (file) this.selectedFile.set(file);
  }

  runBatch(): void {
    const file = this.selectedFile();
    if (!file) return;
    this.loading.set(true);
    this.errorMsg.set('');
    this.service.batchPredict(file).subscribe({
      next: (res) => {
        this.result.set(res);
        this.loading.set(false);
      },
      error: (err) => {
        this.errorMsg.set(err?.error?.detail ?? 'API error — please try again.');
        this.loading.set(false);
      },
    });
  }

  reset(): void {
    this.selectedFile.set(null);
    this.result.set(null);
    this.errorMsg.set('');
  }

  sentimentSummary() {
    const res = this.result();
    if (!res) return [];
    const total = res.total_items || 1;
    const count = (s: string) => res.results.filter(r => r.sentiment === s).length;
    return [
      { label: 'Positive', count: count('positive'), pct: (count('positive') / total) * 100,
        barClass: 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800',
        labelClass: 'text-emerald-600', fillClass: 'bg-emerald-500' },
      { label: 'Neutral',  count: count('neutral'),  pct: (count('neutral')  / total) * 100,
        barClass: 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800',
        labelClass: 'text-amber-600', fillClass: 'bg-amber-500' },
      { label: 'Negative', count: count('negative'), pct: (count('negative') / total) * 100,
        barClass: 'bg-rose-50 dark:bg-rose-900/20 border-rose-200 dark:border-rose-800',
        labelClass: 'text-rose-500', fillClass: 'bg-rose-500' },
    ];
  }

  getSentimentClass(s: string): string {
    switch (s) {
      case 'positive': return 'bg-emerald-500/10 text-emerald-700 border-emerald-500/20';
      case 'negative': return 'bg-rose-500/10 text-rose-700 border-rose-500/20';
      default:         return 'bg-amber-500/10 text-amber-700 border-amber-500/20';
    }
  }

  downloadCSV(): void {
    const res = this.result();
    if (!res) return;
    const header = 'row,text,sentiment,confidence,aspects,error';
    const rows = res.results.map(r =>
      [
        r.row,
        `"${r.text.replace(/"/g, '""')}"`,
        r.sentiment,
        (r.confidence * 100).toFixed(2) + '%',
        `"${r.aspects.map(a => `${a.aspect}:${a.sentiment}`).join('; ')}"`,
        r.error ?? '',
      ].join(',')
    );
    const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'batch_results.csv'; a.click();
    URL.revokeObjectURL(url);
  }
}
