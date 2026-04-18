import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-slide',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './slide.component.html',
  styleUrls: ['./slide.component.css'],
})
export class SlideComponent {
  isExporting = signal(false);
  private apiUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  exportToPDF() {
    this.isExporting.set(true);

    this.http.get(`${this.apiUrl}/export-pdf`, {
      responseType: 'blob'  // Receive as binary
    }).subscribe({
      next: (blob) => {
        // Auto-download the PDF
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'Sentiment_Analysis_Presentation.pdf';
        a.click();
        window.URL.revokeObjectURL(url);
        this.isExporting.set(false);
      },
      error: (err) => {
        console.error('PDF export failed:', err);
        alert('Export PDF thất bại. Hãy đảm bảo backend API đang chạy!');
        this.isExporting.set(false);
      }
    });
  }
}
