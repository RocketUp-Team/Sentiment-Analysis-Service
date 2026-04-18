import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';

@Component({
  selector: 'app-slide',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './slide.component.html',
  styleUrls: ['./slide.component.css'],
})
export class SlideComponent {
  exportToPDF() {
    // Sử dụng hộp thoại in mặc định của hệ điều hành.
    // CSS @media print tại styles.css đã cấu hình chi tiết (ẩn viền, đúng khổ màn hình, v.v)
    window.print();
  }
}
