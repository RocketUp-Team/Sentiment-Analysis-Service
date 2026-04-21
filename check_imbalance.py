import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt

def check_imbalance(dataset_name, config_name=None, split='train', data_files=None):
    print(f"\n{'='*20} Checking: {dataset_name} ({split}) {'='*20}")
    
    # Tải dataset từ Hugging Face
    try:
        if data_files:
             ds = load_dataset("parquet", data_files=data_files, split=split)
        elif dataset_name == "tyqiangz/multilingual-sentiments":
            # Dataset này dùng script cũ (.py) không còn được hỗ trợ ở bản datasets mới.
            # Ta tải trực tiếp file parquet đã được convert.
            data_files_internal = {
                "train": "hf://datasets/tyqiangz/multilingual-sentiments@refs/convert/parquet/english/train/*.parquet"
            }
            ds = load_dataset("parquet", data_files=data_files_internal, split=split)
        else:
            ds = load_dataset(dataset_name, config_name, split=split)
        df = ds.to_pandas()
    except Exception as e:
        print(f"Lỗi khi tải dataset: {e}")
        return

    # Kiểm tra cột nhãn (thường là 'label' hoặc 'sentiment')
    label_col = 'label' if 'label' in df.columns else 'sentiment'
    
    if label_col not in df.columns:
        print(f"Không tìm thấy cột nhãn. Các cột hiện có: {df.columns.tolist()}")
        return

    # Thống kê số lượng
    counts = df[label_col].value_counts()
    percentages = df[label_col].value_counts(normalize=True) * 100
    
    # In thông tin
    stats = pd.DataFrame({
        'Count': counts,
        'Percentage (%)': percentages
    })
    print(stats)
    
    # Kiểm tra mất cân bằng
    max_pct = percentages.max()
    min_pct = percentages.min()
    ratio = max_pct / min_pct if min_pct > 0 else float('inf')
    
    print(f"\nRatio (Max/Min): {ratio:.2f}x")
    if ratio > 3:
        print("⚠️ CẢNH BÁO: Dataset có dấu hiệu mất cân bằng lớp nghiêm trọng (> 3x)!")
        
        print("\n--- XỬ LÝ BẰNG CLASS-WEIGHTED LOSS (PHƯƠNG PHÁP ĐỀ XUẤT) ---")
        print("Thay vì thay đổi số lượng dữ liệu, phương pháp này gán 'trọng số' cho hàm Loss.")
        
        num_classes = len(counts)
        n_samples = len(df)
        
        print(f"Tổng số mẫu: {n_samples} | Số lớp: {num_classes}")
        print("Trọng số (Weight) = N_samples / (N_classes * Count_per_class)\n")
        
        for label_val in sorted(counts.index):
            count = counts[label_val]
            weight = n_samples / (num_classes * count)
            
            # Tính toán "sức mạnh" tương đối so với lớp có trọng số nhỏ nhất
            impact = "↑ Tăng cường mạnh" if weight >= 2.0 else ("↑ Tăng nhẹ" if weight > 1.0 else "↓ Giảm sự ảnh hưởng")
            
            print(f"  > Lớp {label_val}: Mẫu = {count:5d} | Trọng số áp dụng = {weight:6.4f} ({impact})")
            
        print("\n=> KẾT LUẬN SAU XỬ LÝ:")
        print("   Dữ liệu được giữ nguyên 100% gốc (không sinh ảo, không cắt bớt).")
        print("   Tuy nhiên, khi huấn luyện, 1 mẫu của lớp thiểu số sẽ tạo ra mức độ cập nhật (Gradient)")
        print("   lớn gấp nhiều lần so với 1 mẫu của lớp đa số, ép mô hình phải học cân bằng.")
        
    else:
        print("✅ Dataset tương đối cân bằng. Không cần áp dụng class weights.")

if __name__ == "__main__":
    # 1. Kiểm tra Sarcasm (tweet_eval - irony)
    check_imbalance("cardiffnlp/tweet_eval", "irony")
    
    # 2. Kiểm tra Multilingual Sentiment (English)
    check_imbalance("tyqiangz/multilingual-sentiments")

    # 3. Kiểm tra UIT-VSFC (Vietnamese Students Feedback)
    vsfc_files = {
        "train": "hf://datasets/uitnlp/vietnamese_students_feedback@refs/convert/parquet/default/train/*.parquet"
    }
    check_imbalance("uitnlp/vietnamese_students_feedback", data_files=vsfc_files)

