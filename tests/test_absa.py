import sys
import os

# Add the project root to sys.path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.baseline import BaselineModelInference
print('Đang nạp mô hình, nếu là lần đầu sẽ mất ~30 giây để download từ Hugging Face...')
try:
    model = BaselineModelInference()
    print('✅ Khởi tạo Pipeline Model thành công!')
except Exception as e:
    print(f'❌ Lỗi khi khởi động Model: {e}')
    exit(1)
print('='*50)
print('🚀 CÔNG CỤ TEST NHANH ZERO-SHOT ABSA')
print('Nhập đoạn văn bản (tiếng Anh) bạn muốn phân tích.')
print('Gõ \"exit\" hoặc \"quit\" để thoát.')
print('='*50)
while True:
    text = input('\nNhập Text: ')
    if text.lower() in ('exit', 'quit', 'q'):
        print('Đang thoát...')
        break
    if not text.strip():
        continue
        
    print('-'*50)
    print('Đang phân tích suy luận (Inference)...')
    try:
        result = model.predict_single(text)
        print(f'\n[KẾT QUẢ TỔNG QUAN]')
        print(f'➜ Sentiment (Cảm xúc chung): {result.sentiment.upper()} (Độ tin cậy: {result.confidence:.2%})')
        print(f'➜ Nhãn mỉa mai (Sarcasm): {result.sarcasm_flag}')
        
        print('\n[KẾT QUẢ ASPECT - KHÍA CẠNH]')
        if result.aspects:
            for a in result.aspects:
                print(f'  • Điểm chạm: {a.aspect.upper()} ➔ Chiều hướng: {a.sentiment} (Độ tin cậy: {a.confidence:.2%})')
        else:
            print('  • Không phát hiện khía cạnh nào đặc thù.')
            
    except Exception as e:
        print(f'❌ Có lỗi khi predict: {e}')
    print('-'*50)