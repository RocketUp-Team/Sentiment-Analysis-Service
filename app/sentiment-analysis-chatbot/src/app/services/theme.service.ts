import { Injectable, signal, effect } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class ThemeService {
  private darkModeSignal = signal<boolean>(false);
  public isDarkMode = this.darkModeSignal.asReadonly();

  constructor() {
    // Load preference from localStorage
    const saved = localStorage.getItem('tg_night_mode');
    if (saved === 'true') {
      this.darkModeSignal.set(true);
      this.updateBodyClass(true);
    }

    // React to changes
    effect(() => {
      const isDark = this.darkModeSignal();
      this.updateBodyClass(isDark);
      localStorage.setItem('tg_night_mode', isDark.toString());
    });
  }

  public toggleTheme(): void {
    this.darkModeSignal.update(val => !val);
  }

  private updateBodyClass(isDark: boolean): void {
    if (isDark) {
      document.body.classList.add('dark');
    } else {
      document.body.classList.remove('dark');
    }
  }
}
