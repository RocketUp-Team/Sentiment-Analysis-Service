import { Component, inject, viewChild, ElementRef, effect, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SentimentAnalysisService } from './services/sentiment-analysis.service';
import { ThemeService } from './services/theme.service';
import { MessageBubbleComponent } from './components/message-bubble/message-bubble.component';
import { ChatInputComponent } from './components/chat-input/chat-input.component';
import { LucideAngularModule, Search, Menu, MoreVertical, Moon, Sun, Trash2, BrainCircuit, Sparkles, User, Settings, Archive, Star, Bookmark, X } from 'lucide-angular';
import { trigger, transition, style, animate, state } from '@angular/animations';

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
    <div class="flex h-screen w-screen overflow-hidden bg-white dark:bg-[#020617] transition-all duration-700">
      
      <!-- Premium Mesh Background -->
      <div class="chat-bg-pattern"></div>
      
      <!-- Mobile Sidebar Overlay Backdrop -->
      <div *ngIf="isSidebarMobileOpen()" (click)="toggleSidebarMobile()" class="fixed inset-0 bg-black/50 dark:bg-black/80 z-40 md:hidden backdrop-blur-sm animate-in fade-in transition-all"></div>

      <!-- LEFT SIDEBAR (Premium Bot List) -->
      <aside [ngClass]="isSidebarMobileOpen() ? 'translate-x-0' : '-translate-x-full md:translate-x-0'" class="fixed md:relative inset-y-0 left-0 w-[280px] md:w-[320px] lg:w-[400px] flex flex-col border-r border-black/5 dark:border-white/5 z-50 glass-tg transition-transform duration-300 bg-white/95 dark:bg-slate-950/95 md:bg-transparent backdrop-blur-3xl shadow-2xl md:shadow-none">
        <!-- Sidebar Header -->
        <div class="p-6 flex items-center gap-4">
          <button (click)="toggleMenu()" class="p-2.5 hover:bg-black/5 dark:hover:bg-white/5 rounded-2xl transition-all">
             <lucide-angular name="menu" [size]="24" class="text-slate-500"></lucide-angular>
          </button>
          <div class="flex-1 bg-black/5 dark:bg-white/5 ring-1 ring-inset ring-black/5 dark:ring-white/10 rounded-2xl flex items-center px-4 py-3 gap-3 focus-within:ring-indigo-500/50 focus-within:bg-white dark:focus-within:bg-slate-900 transition-all">
             <lucide-angular name="search" [size]="18" class="text-slate-400"></lucide-angular>
             <input type="text" placeholder="Search insights..." class="bg-transparent border-none focus:ring-0 text-[14.5px] w-full outline-none dark:text-white placeholder:text-slate-400">
          </div>
        </div>

        <!-- Bot List with Premium Transitions -->
        <div class="flex-1 overflow-y-auto space-y-1 pb-4">
           <div *ngFor="let chat of botList; let i = index" 
                class="chat-item"
                [style.animation-delay]="i * 50 + 'ms'"
                [ngClass]="{'active': chat.id === 'current-bot'}">
              <div class="relative">
                 <div class="w-12 h-12 rounded-2xl flex items-center justify-center text-white font-bold text-lg shadow-lg rotate-3 group-hover:rotate-0 transition-transform"
                      [style.background]="chat.avatar">
                    {{ chat.name[0] }}
                 </div>
                 <div *ngIf="chat.status === 'online' || chat.status === 'typing'" 
                      class="absolute -bottom-1 -right-1 w-4 h-4 bg-emerald-500 border-[3px] border-white dark:border-slate-900 rounded-full shadow-sm"></div>
              </div>
              <div class="flex-1 min-w-0">
                 <div class="flex justify-between items-center mb-0.5">
                    <h3 class="tg-title truncate">{{ chat.name }}</h3>
                    <span class="text-[11px] font-semibold text-slate-400 tabular-nums uppercase">{{ chat.time }}</span>
                 </div>
                 <p class="text-[13px] text-slate-500 dark:text-slate-400 truncate font-medium">
                    <span *ngIf="chat.status === 'typing'" class="text-indigo-500 dark:text-indigo-400 font-bold animate-pulse">Analyzing...</span>
                    <span *ngIf="chat.status !== 'typing'">{{ chat.lastMsg }}</span>
                 </p>
              </div>
              <div *ngIf="chat.unread" class="bg-indigo-500 shadow-lg shadow-indigo-500/20 text-white text-[10px] font-black px-2 py-1 rounded-lg">
                 {{ chat.unread }}
              </div>
           </div>
        </div>

        <!-- Premium Menu Slide-in -->
        <div *ngIf="isMenuOpen()" class="absolute inset-0 z-30 p-8 glass-panel animate-in fade-in slide-in-from-left duration-500">
           <div class="flex items-center justify-between mb-12">
              <div class="flex items-center gap-5">
                 <div class="w-20 h-20 rounded-[32px] bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center text-white text-3xl font-black shadow-2xl rotate-6">
                    LP
                 </div>
                 <div class="space-y-1">
                    <h4 class="font-bold text-2xl tracking-tighter dark:text-white">Long Phạm</h4>
                    <p class="text-[10px] bg-indigo-500/10 text-indigo-500 px-2.5 py-1 rounded-full font-black uppercase tracking-widest inline-block border border-indigo-500/20">God Tier Elite</p>
                 </div>
              </div>
              <button (click)="toggleMenu()" class="p-3 text-slate-400 hover:text-indigo-500 transition-colors">
                <lucide-angular name="search" [size]="24"></lucide-angular>
              </button>
           </div>
           <nav class="space-y-2">
              <div *ngFor="let item of menuItems" class="flex items-center gap-6 p-4 hover:bg-indigo-500/5 dark:hover:bg-indigo-500/10 rounded-3xl text-slate-600 dark:text-slate-300 cursor-pointer group transition-all hover:translate-x-2">
                 <div class="p-3 rounded-2xl bg-black/5 dark:bg-white/5 group-hover:bg-indigo-500 group-hover:text-white transition-all">
                    <lucide-angular [name]="item.icon" [size]="22"></lucide-angular>
                 </div>
                 <span class="font-bold text-lg tracking-tight">{{ item.label }}</span>
              </div>
           </nav>
        </div>
      </aside>

      <!-- MAIN CONTENT (Immersive Experience) -->
      <section class="flex-1 flex flex-col relative overflow-hidden">
        
        <!-- Background Glowing Orbs -->
        <div class="absolute top-[10%] left-[20%] w-[400px] h-[400px] bg-indigo-500/5 rounded-full blur-[120px] -z-10 animate-float"></div>
        <div class="absolute bottom-[20%] right-[10%] w-[300px] h-[300px] bg-violet-500/5 rounded-full blur-[100px] -z-10 animate-float" style="animation-delay: -2s"></div>

        <!-- Premium Navigation Header -->
        <header class="h-[74px] glass-tg flex items-center justify-between px-4 md:px-8 z-20 shrink-0">
           <div class="flex items-center gap-2 md:gap-5 cursor-pointer group w-[60%] md:w-auto">
              <!-- Mobile Hamburger Menu Toggle -->
              <button (click)="toggleSidebarMobile()" class="md:hidden p-2 -ml-2 hover:bg-black/5 dark:hover:bg-white/5 rounded-xl text-slate-500 dark:text-slate-400 shrink-0">
                 <lucide-angular name="menu" [size]="24"></lucide-angular>
              </button>
              
              <div class="w-10 h-10 md:w-12 md:h-12 rounded-[16px] md:rounded-[20px] bg-gradient-to-tr from-indigo-600 to-cyan-400 flex items-center justify-center text-white shadow-xl shadow-indigo-500/20 group-hover:scale-110 group-hover:rotate-6 transition-all duration-500 shrink-0">
                 <lucide-angular name="brain-circuit" class="w-5 h-5 md:w-[26px] md:h-[26px]"></lucide-angular>
              </div>
              <div class="min-w-0">
                 <h2 class="text-base md:text-xl font-black tracking-tighter dark:text-white truncate pr-2">Sentiment Pro <span class="text-indigo-500">AI</span></h2>
                 <div class="hidden md:flex items-center gap-2">
                    <span class="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></span>
                    <p class="text-[10px] font-black uppercase tracking-[0.2em] text-slate-400">Neural Engine v4.0</p>
                 </div>
              </div>
           </div>
           
           <div class="flex items-center gap-1.5 md:gap-3 shrink-0">
              <button (click)="toggleTheme()" 
                      class="p-2.5 md:p-3 bg-black/5 dark:bg-white/5 hover:bg-indigo-500/10 dark:hover:bg-indigo-500/20 rounded-xl md:rounded-2xl text-slate-500 dark:text-indigo-400 border border-black/5 dark:border-white/10 transition-all active:scale-95 hover:shadow-lg hover:shadow-indigo-500/10 shrink-0">
                 <lucide-angular name="sun" *ngIf="isDarkMode()" class="w-5 h-5 md:w-[22px] md:h-[22px]"></lucide-angular>
                 <lucide-angular name="moon" *ngIf="!isDarkMode()" class="w-5 h-5 md:w-[22px] md:h-[22px]"></lucide-angular>
              </button>
              <button class="hidden sm:block p-2.5 md:p-3 bg-black/5 dark:bg-white/5 hover:bg-black/10 dark:hover:bg-white/10 rounded-xl md:rounded-2xl text-slate-400 transition-all active:scale-95 border border-black/5 dark:border-white/10">
                 <lucide-angular name="search" class="w-5 h-5 md:w-[22px] md:h-[22px]"></lucide-angular>
              </button>
              <button (click)="clearHistory()" class="p-2.5 md:p-3 bg-black/5 dark:bg-white/5 hover:bg-red-500/10 rounded-xl md:rounded-2xl text-slate-400 hover:text-red-500 transition-all active:scale-95 border border-black/5 dark:border-white/10 shrink-0">
                 <lucide-angular name="trash-2" class="w-5 h-5 md:w-[22px] md:h-[22px]"></lucide-angular>
              </button>
              <div class="hidden md:block w-px h-8 bg-black/5 dark:bg-white/10 mx-1"></div>
              <button class="hidden md:flex w-11 h-11 rounded-2xl bg-gradient-to-br from-indigo-500 to-violet-600 items-center justify-center text-white shadow-xl shadow-indigo-500/30 active:scale-95 transition-all hover:rotate-3">
                 <lucide-angular name="star" [size]="20"></lucide-angular>
              </button>
           </div>
        </header>

        <!-- Message Flow Area -->
        <div #scrollContainer class="flex-1 overflow-y-auto px-4 py-6 md:px-6 md:py-8 lg:px-[22%] space-y-4 z-0 relative scroll-smooth">
           <div class="flex flex-col gap-1">
              <app-message-bubble *ngFor="let msg of messages()" [message]="msg"></app-message-bubble>
           </div>
        </div>

        <!-- Immersive Input Layer -->
        <footer class="z-20 px-4 pb-8 pt-2">
           <div class="max-w-[1000px] mx-auto">
              <app-chat-input 
                (onSendText)="handleSendText($event)" 
                (onSendFile)="handleSendFile($event)" 
                (onSendAudio)="handleSendAudio($event)" 
                [disabled]="isAnalyzing()">
              </app-chat-input>
           </div>
        </footer>
      </section>
    </div>

  `,
})
export class AppComponent {
  private sentimentService = inject(SentimentAnalysisService);
  private themeService = inject(ThemeService);
  
  messages = this.sentimentService.messages;
  isDarkMode = this.themeService.isDarkMode;
  scrollContainer = viewChild<ElementRef>('scrollContainer');
  isMenuOpen = signal(false);
  isSidebarMobileOpen = signal(false);

  botList: ChatItem[] = [
    { id: 'current-bot', name: 'Sentiment Pro AI', avatar: 'linear-gradient(135deg, #6366f1, #06b6d4)', lastMsg: 'Deep neural analysis active...', time: 'now', status: 'online', isBot: true },
    { id: 'dev-support', name: 'Core Infrastructure', avatar: 'linear-gradient(135deg, #f43f5e, #fb923c)', lastMsg: 'Cluster health: 100%', time: '16:48', status: 'online', isBot: true },
    { id: 'analytics', name: 'Global Optimizer', avatar: 'linear-gradient(135deg, #10b981, #34d399)', lastMsg: 'Processing market trends...', time: '16:07', status: 'typing' },
    { id: 'assistant', name: 'Elite AI Concierge', avatar: 'linear-gradient(135deg, #8b5cf6, #d946ef)', lastMsg: 'Ready for priority tasks', time: '15:20', status: 'online', unread: 5 },
    { id: 'news-bot', name: 'Intelligence Feed', avatar: 'linear-gradient(135deg, #3b82f6, #2563eb)', lastMsg: 'Sentiment spike detected in tech', time: 'Yesterday', status: 'offline' },
    { id: 'crawler', name: 'Neural Crawler VIP', avatar: 'linear-gradient(135deg, #0f172a, #334155)', lastMsg: 'Indexing social graphs...', time: 'Wed', status: 'typing', unread: 12 },
    { id: 'marketing', name: 'Strategic Insights', avatar: 'linear-gradient(135deg, #f59e0b, #fbbf24)', lastMsg: 'Omni-channel ROI optimized', time: 'Tue', status: 'online' },
    { id: 'security', name: 'Security Protocol', avatar: 'linear-gradient(135deg, #475569, #1e293b)', lastMsg: 'Encrypted tunnel stable', time: 'Mon', status: 'online' },
    { id: 'translator', name: 'Polyglot Nexus 🌍', avatar: 'linear-gradient(135deg, #2dd4bf, #0d9488)', lastMsg: 'Real-time contextual translation', time: 'Mar 12', status: 'offline' },
    { id: 'feedback', name: 'Loyalty Metrics', avatar: 'linear-gradient(135deg, #ec4899, #be185d)', lastMsg: 'KPI: Exceptional Experience', time: 'Feb 26', status: 'online', unread: 1 }
  ];

  menuItems = [
    { label: 'My Profile', icon: 'user' },
    { label: 'Saved Messages', icon: 'bookmark' },
    { label: 'Archive', icon: 'archive' },
    { label: 'Favorites', icon: 'star' },
    { label: 'Settings', icon: 'settings' }
  ] as const;

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

    if (this.messages().length === 0) {
       this.sentimentService.sendMessage('banner-team Xin chào Long Phạm! Hệ thống AI sẵn sàng phân tích dữ liệu đa nền tảng trên cả Điện thoại và Máy tính!');
    }
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
