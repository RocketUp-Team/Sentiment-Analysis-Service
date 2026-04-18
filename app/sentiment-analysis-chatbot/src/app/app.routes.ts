import { Routes } from '@angular/router';
import { ChatPageComponent } from './components/chat-page/chat-page.component';
import { SlideComponent } from './components/present/slide/slide.component';

export const routes: Routes = [
  { path: '', component: ChatPageComponent },
  { path: 'present', component: SlideComponent },
  { path: '**', redirectTo: '' },
];
