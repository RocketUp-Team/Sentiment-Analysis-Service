import { Component, inject, viewChild, ElementRef, effect, signal, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SentimentAnalysisService } from './services/sentiment-analysis.service';
import { ThemeService } from './services/theme.service';
import { MessageBubbleComponent } from './components/message-bubble/message-bubble.component';
import { ChatInputComponent } from './components/chat-input/chat-input.component';
import { LucideAngularModule, Search, Menu, MoreVertical, Moon, Sun, Trash2, BrainCircuit, Sparkles, User, Settings, Archive, Star, Bookmark, X, Loader2, Zap } from 'lucide-angular';
import { trigger, transition, style, animate, state } from '@angular/animations';
import { interval, takeWhile, switchMap, catchError, of, Subscription } from 'rxjs';

interface ChatItem {
  id: string;
  name: string;
  avatar: string;
  lastMsg: string;
  time: string;
  unread?: number;
  isBot?: boolean;
  status: 'online' | 'typing' | 'offline';
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule, 
    MessageBubbleComponent, 
    ChatInputComponent, 
    LucideAngularModule
  ],
  template: `
    <div class="flex h-screen w-screen overflow-hidden bg-white dark:bg-[#020617] transition-all duration-700 relative">
      
      <!-- Premium Loading Overlay -->
      <div *ngIf="!isBackendReady()" 
           class="absolute inset-0 z-[100] flex flex-col items-center justify-center bg-white dark:bg-[#020617] transition-all duration-1000">
          <div class="chat-bg-pattern !opacity-20"></div>
          
          <div class="relative w-32 h-32 md:w-48 md:h-48 mb-8">
             <div class="absolute inset-0 rounded-[2.5rem] bg-indigo-500/20 animate-ping"></div>
             <div class="absolute inset-4 rounded-[2rem] bg-indigo-500/40 animate-pulse"></div>
             <div class="absolute inset-0 flex items-center justify-center">
                <div class="w-16 h-16 md:w-24 md:h-24 rounded-[2rem] bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center text-white shadow-2xl shadow-indigo-500/40 animate-float">
                   <lucide-angular name="brain-circuit" class="w-8 h-8 md:w-12 md:h-12"></lucide-angular>
                </div>
             </div>
          </div>

          <div class="text-center space-y-4 px-6 max-w-md z-10">
             <h2 class="text-2xl md:text-3xl font-black tracking-tighter dark:text-white">
                Preparing <span class="text-indigo-500">Neural Engine</span>
             </h2>
             <p class="text-slate-500 dark:text-slate-400 font-medium text-sm md:text-base leading-relaxed">
                Hệ thống đang tải mô hình AI (~2GB). <br class="hidden md:block"> Vui lòng đợi trong giây lát...
             </p>
             
             <div class="flex items-center justify-center gap-3 pt-4">
                <div class="flex items-center gap-2 px-4 py-2 bg-black/5 dark:bg-white/5 rounded-2xl border border-black/5 dark:border-white/5 animate-pulse">
                   <lucide-angular name="loader-2" class="w-4 h-4 text-indigo-500 animate-spin"></lucide-angular>
                   <span class="text-[10px] font-black uppercase tracking-widest text-slate-500">Loading Deep Weights</span>
                </div>
             </div>
          </div>

          <div class="absolute bottom-12 text-center">
              <p class="text-[10px] font-black uppercase tracking-[0.3em] text-slate-400 group flex items-center gap-2">
                 <lucide-angular name="zap" class="w-3 h-3 text-amber-500"></lucide-angular>
                 Optimized on AI-Dev-Ops
              </p>
          </div>
      </div>

      <!-- Premium Mesh Background -->
      <div class="chat-bg-pattern"></div>
      
      <!-- Mobile Sidebar Overlay Backdrop -->
      <div *ngIf="isSidebarMobileOpen()" (click)="toggleSidebarMobile()" class="fixed inset-0 bg-black/50 dark:bg-black/80 z-40 md:hidden backdrop-blur-sm animate-in fade-in transition-all"></div>

      <!-- LEFT SIDEBAR -->
      <aside [ngClass]="isSidebarMobileOpen() ? 'translate-x-0' : '-translate-x-full md:translate-x-0'" class="fixed md:relative inset-y-0 left-0 w-[280px] md:w-[320px] lg:w-[400px] flex flex-col border-r border-black/5 dark:border-white/5 z-50 glass-tg transition-transform duration-300 bg-white/95 dark:bg-slate-950/95 md:bg-transparent backdrop-blur-3xl shadow-2xl md:shadow-none">
        <div class="p-6 flex flex-col gap-6">
          <div class="flex items-center gap-4">
            <button (click)="toggleMenu()" class="p-2.5 hover:bg-black/5 dark:hover:bg-white/5 rounded-2xl transition-all">
               <lucide-angular name="menu" [size]="24" class="text-slate-500"></lucide-angular>
            </button>
            <h1 class="text-xl font-black tracking-tighter dark:text-white">Sentiment <span class="text-indigo-500">Pro</span></h1>
          </div>
          
          <div class="space-y-4">
            <div class="bg-indigo-500/5 border border-indigo-500/10 p-5 rounded-[2rem] space-y-3">
              <div class="flex items-center gap-3">
                <div class="w-10 h-10 rounded-2xl bg-indigo-500 flex items-center justify-center text-white shadow-lg shadow-indigo-500/20">
                  <lucide-angular name="brain-circuit" [size]="20"></lucide-angular>
                </div>
                <div>
                  <h3 class="text-xs font-black uppercase tracking-widest text-indigo-500">Analysis Engine</h3>
                  <p class="text-[13px] font-bold dark:text-white">{{ isBackendReady() ? 'Active & Ready' : 'Initializing...' }}</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="flex-1 px-6 space-y-6">
          <div class="space-y-2">
            <p class="text-[10px] font-black uppercase tracking-[0.2em] text-slate-400 px-1">Session Info</p>
            <div class="glass-panel p-4 rounded-3xl border border-black/5 dark:border-white/5">
              <div class="flex items-center justify-between mb-3">
                 <span class="text-xs font-bold text-slate-500">Total Analysis</span>
                 <span class="text-xs font-black text-indigo-500">{{ messages().length }}</span>
              </div>
              <div class="w-full bg-black/5 dark:bg-white/5 h-1.5 rounded-full overflow-hidden">
                 <div class="bg-indigo-500 h-full w-[65%] rounded-full shadow-[0_0_8px_rgba(99,102,241,0.5)]"></div>
              </div>
            </div>
          </div>
        </div>

        <div class="p-6 mt-auto border-t border-black/5 dark:border-white/5">
           <div class="flex items-center gap-4 p-3 hover:bg-black/5 dark:hover:bg-white/5 rounded-[2rem] transition-all cursor-pointer group">
              <div class="w-12 h-12 rounded-[1.25rem] bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center text-white font-black shadow-lg group-hover:rotate-6 transition-transform">
                 AI
              </div>
              <div class="flex-1 min-w-0">
                 <h4 class="text-sm font-black dark:text-white truncate">Administrator</h4>
                 <p class="text-[10px] font-bold text-emerald-500 uppercase tracking-widest">System {{ isBackendReady() ? 'Online' : 'Offline' }}</p>
              </div>
              <lucide-angular name="settings" [size]="18" class="text-slate-400 group-hover:rotate-90 transition-transform duration-500"></lucide-angular>
           </div>
        </div>

        <div *ngIf="isMenuOpen()" class="settings-overlay animate-in fade-in slide-in-from-left duration-500">
           <div class="flex items-center justify-between mb-12">
              <h2 class="text-2xl font-black tracking-tighter dark:text-white">Settings</h2>
              <button (click)="toggleMenu()" class="p-3 text-slate-400 hover:text-indigo-500 transition-colors">
                <lucide-angular name="x" [size]="24"></lucide-angular>
              </button>
           </div>
           <nav class="space-y-4">
              <button (click)="clearHistory()" class="w-full flex items-center gap-6 p-5 hover:bg-red-500/10 rounded-3xl text-red-500 transition-all active:scale-95 group">
                 <div class="p-3 rounded-2xl bg-red-500/10 group-hover:bg-red-500 group-hover:text-white transition-all">
                    <lucide-angular name="trash-2" [size]="22"></lucide-angular>
                 </div>
                 <span class="font-bold text-lg tracking-tight">Clear All Data</span>
              </button>
           </nav>
        </div>
      </aside>

      <!-- MAIN CONTENT -->
      <section class="flex-1 flex flex-col relative overflow-hidden">
        
        <div class="absolute top-[10%] left-[20%] w-[400px] h-[400px] bg-indigo-500/5 rounded-full blur-[120px] -z-10 animate-float"></div>
        <div class="absolute bottom-[20%] right-[10%] w-[300px] h-[300px] bg-violet-500/5 rounded-full blur-[100px] -z-10 animate-float" style="animation-delay: -2s"></div>

        <header class="h-[74px] glass-tg flex items-center justify-between px-4 md:px-8 z-20 shrink-0 border-b border-black/5 dark:border-white/5">
           <div class="flex items-center gap-2 md:gap-5 cursor-pointer group">
              <button (click)="toggleSidebarMobile()" class="md:hidden p-2 -ml-2 hover:bg-black/5 dark:hover:bg-white/5 rounded-xl text-slate-500 dark:text-slate-400 shrink-0">
                 <lucide-angular name="menu" [size]="24"></lucide-angular>
              </button>
              
              <div class="w-10 h-10 md:w-11 md:h-11 rounded-2xl bg-indigo-500 flex items-center justify-center text-white shadow-xl shadow-indigo-500/20 group-hover:scale-110 transition-all shrink-0">
                 <lucide-angular name="brain-circuit" [size]="24"></lucide-angular>
              </div>
              <div class="min-w-0">
                 <h2 class="text-base md:text-lg font-black tracking-tighter dark:text-white truncate">Sentiment Analysis</h2>
                 <p class="text-[9px] font-black uppercase tracking-[0.2em] text-emerald-500">{{ isBackendReady() ? 'Live Prediction Active' : 'Initializing...' }}</p>
              </div>
           </div>
           
           <div class="flex items-center gap-3">
              <button (click)="toggleTheme()" 
                      class="p-3 bg-black/5 dark:bg-white/5 hover:bg-indigo-500/10 rounded-2xl text-slate-500 dark:text-indigo-400 transition-all active:scale-95">
                 <lucide-angular name="sun" *ngIf="isDarkMode()" [size]="20"></lucide-angular>
                 <lucide-angular name="moon" *ngIf="!isDarkMode()" [size]="20"></lucide-angular>
              </button>
           </div>
        </header>

        <div #scrollContainer class="flex-1 overflow-y-auto px-4 py-6 md:px-8 lg:px-[25%] space-y-6 z-0 relative scroll-smooth">
           <div class="flex flex-col gap-2">
              <app-message-bubble *ngFor="let msg of messages()" [message]="msg"></app-message-bubble>
           </div>
        </div>

        <footer class="z-20 px-4 pb-8 pt-2">
           <div class="max-w-[1000px] mx-auto">
              <app-chat-input 
                (onSendText)="handleSendText($event)" 
                (onSendFile)="handleSendFile($event)" 
                (onSendAudio)="handleSendAudio($event)" 
                [disabled]="!isBackendReady()">
              </app-chat-input>
           </div>
        </footer>
      </section>
    </div>

  `,
})
export class AppComponent implements OnInit, OnDestroy {
  private sentimentService = inject(SentimentAnalysisService);
  private themeService = inject(ThemeService);
  
  messages = this.sentimentService.messages;
  isDarkMode = this.themeService.isDarkMode;
  scrollContainer = viewChild<ElementRef>('scrollContainer');
  isMenuOpen = signal(false);
  isSidebarMobileOpen = signal(false);
  isBackendReady = signal(false);
  
  private healthSub?: Subscription;

  constructor() {
    effect(() => {
      const msgs = this.messages();
      const container = this.scrollContainer()?.nativeElement;
      if (container) {
        setTimeout(() => {
          container.scrollTop = container.scrollHeight;
        }, 100);
      }
    });
  }

  ngOnInit() {
    this.pollHealth();
  }

  ngOnDestroy() {
    this.healthSub?.unsubscribe();
  }

  private pollHealth() {
    this.healthSub = interval(3000).pipe(
      switchMap(() => this.sentimentService.checkHealth().pipe(
        catchError(err => {
          console.log('Backend not ready yet...');
          return of(null);
        })
      )),
      takeWhile(res => !res, true)
    ).subscribe(res => {
      if (res) {
        this.isBackendReady.set(true);
        if (this.messages().length === 0) {
           this.sentimentService.addWelcomeMessage();
        }
        this.healthSub?.unsubscribe();
      }
    });
  }

  isAnalyzing(): boolean {
    const msgs = this.messages();
    return msgs.length > 0 && msgs[msgs.length - 1].sentiment === 'LOADING';
  }

  handleSendText(event: { text: string, lang: 'vi' | 'en' }) {
    this.sentimentService.sendMessage(event.text, event.lang);
  }

  handleSendFile(event: { file: File, lang: 'vi' | 'en' }) {
    this.sentimentService.sendDocument(event.file, event.lang);
  }

  handleSendAudio(event: { audio: Blob, lang: 'vi' | 'en' }) {
    this.sentimentService.sendAudio(event.audio, event.lang);
  }

  toggleTheme() {
    this.themeService.toggleTheme();
  }

  toggleMenu() {
    this.isMenuOpen.update(val => !val);
  }

  toggleSidebarMobile() {
    this.isSidebarMobileOpen.update(val => !val);
  }

  clearHistory() {
    if (confirm('Bạn có chắc muốn xóa lịch sử trò chuyện?')) {
      this.sentimentService.clearHistory();
    }
  }
}
