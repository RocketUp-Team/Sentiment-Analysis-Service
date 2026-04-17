import { ApplicationConfig, importProvidersFrom } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { provideAnimations } from '@angular/platform-browser/animations';
import {
  LucideAngularModule,
  Search, Menu, MoreVertical, Moon, Sun, Trash2, BrainCircuit,
  Sparkles, User, Settings, Archive, Star, Bookmark, X,
  Paperclip, Smile, Mic, Send, CheckCheck, Frown, Meh,
  Loader2, AlertCircle, Square,
  // newly needed icons
  Clock, BarChart2, AlertTriangle, Zap, Globe
} from 'lucide-angular';

export const appConfig: ApplicationConfig = {
  providers: [
    provideHttpClient(),
    provideAnimations(),
    importProvidersFrom(LucideAngularModule.pick({
      Search, Menu, MoreVertical, Moon, Sun, Trash2, BrainCircuit,
      Sparkles, User, Settings, Archive, Star, Bookmark, X,
      Paperclip, Smile, Mic, Send, CheckCheck, Frown, Meh,
      Loader2, AlertCircle, Square,
      Clock, BarChart2, AlertTriangle, Zap, Globe,
    }))
  ]
};
