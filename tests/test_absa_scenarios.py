import sys
import time
import os
# Add the project root to sys.path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.model.baseline import BaselineModelInference
except ImportError:
    print("❌ Lỗi: Hãy chắc chắn bạn đang chạy lệnh từ thư mục gốc của dự án.")
    sys.exit(1)

scenarios = [
    {
        "category": "1. Mixed Sentiments",
        "text": "The steak was perfectly cooked and delicious, but the waiter was extremely rude and we had to wait 40 minutes for our table.",
        "expected": "Food: positive | Service: negative",
        "expected_aspects": {"food": ["positive"], "service": ["negative"]}
    },
    {
        "category": "1. Mixed Sentiments",
        "text": "Great location right in the city center, though the room was a bit noisy and smaller than expected.",
        "expected": "Location: positive | Ambiance/General: negative",
        "expected_aspects": {"location": ["positive"], "ambiance/general": ["negative"]}
    },
    {
        "category": "2. Price vs. Quality",
        "text": "Expensive but worth every penny for the high-quality ingredients they use.",
        "expected": "Price: negative/neutral | Food: positive",
        "expected_aspects": {"price": ["negative", "neutral"], "food": ["positive"]}
    },
    {
        "category": "2. Price vs. Quality",
        "text": "The pizza is very cheap, but honestly, it tastes like cardboard.",
        "expected": "Price: positive | Food: negative",
        "expected_aspects": {"price": ["positive"], "food": ["negative"]}
    },
    {
        "category": "3. Ambiance & Atmosphere",
        "text": "Loved the cozy atmosphere and the live music was a nice touch, but the drinks are seriously overpriced.",
        "expected": "Ambiance: positive | Price: negative",
        "expected_aspects": {"ambiance": ["positive"], "price": ["negative"]}
    },
    {
        "category": "3. Ambiance & Atmosphere",
        "text": "The restaurant was way too dark and the music was so loud we couldn't even hear each other talk.",
        "expected": "Ambiance: negative",
        "expected_aspects": {"ambiance": ["negative"]}
    },
    {
        "category": "4. General & Service",
        "text": "Overall a decent experience, nothing special but the staff was friendly enough.",
        "expected": "General: neutral/positive | Service: positive",
        "expected_aspects": {"general": ["neutral", "positive"], "service": ["positive"]}
    },
    {
        "category": "4. General & Service",
        "text": "I would never go back there again. Terrible management and poor hygiene.",
        "expected": "Service/General: negative",
        "expected_aspects": {"service/general": ["negative"]}
    },
    {
        "category": "5. Sarcasm (Mỉa mai)",
        "text": "Thanks for the cold soup and the 1-hour wait, what a wonderful evening!",
        "expected": "Food: negative (Thách thức vì có từ 'wonderful')",
        "expected_aspects": {"food": ["negative"]}
    }
]

def evaluate_result(result, expected_aspects):
    if not expected_aspects:
        return True, "No specific aspect expectations."

    result_dict = {a.aspect.lower(): a.sentiment.lower() for a in result.aspects} if result.aspects else {}
    errors = []

    for exp_aspect, allowed_sentiments in expected_aspects.items():
        if exp_aspect == "ambiance/general":
            sent_ambiance = result_dict.get("ambiance")
            sent_general = result_dict.get("general")
            if sent_ambiance not in allowed_sentiments and sent_general not in allowed_sentiments:
                errors.append(f"Expected Ambiance/General to be in {allowed_sentiments}, got Ambiance: {sent_ambiance}, General: {sent_general}")
            continue

        if exp_aspect == "service/general":
            sent_service = result_dict.get("service")
            sent_general = result_dict.get("general")
            if sent_service not in allowed_sentiments and sent_general not in allowed_sentiments:
                errors.append(f"Expected Service/General to be in {allowed_sentiments}, got Service: {sent_service}, General: {sent_general}")
            continue

        if exp_aspect not in result_dict:
            errors.append(f"Aspect '{exp_aspect}' not found in predictions.")
        elif result_dict[exp_aspect] not in allowed_sentiments:
            errors.append(f"Aspect '{exp_aspect}' expected {allowed_sentiments}, got '{result_dict[exp_aspect]}'.")

    if errors:
        return False, " | ".join(errors)
    return True, "Passed"

def main():
    print("⏳ Đang nạp mô hình...")
    model = BaselineModelInference()
    print("✅ Load model thành công. Bắt đầu test kịch bản...\n")

    passed_count = 0

    for i, item in enumerate(scenarios, 1):
        print("="*80)
        print(f"[{item['category']}] Kịch bản #{i}")
        print(f"📝 Text: \"{item['text']}\"")
        print(f"🎯 Kỳ vọng: {item['expected']}")
        print("-" * 80)
        
        start_t = time.time()
        result = model.predict_single(item["text"])
        lat = time.time() - start_t
        
        print(f"👉 KẾT QUẢ TỔNG QUAN:")
        print(f"   - Sentiment chung: {result.sentiment.upper()} ({result.confidence:.2%})")
        print(f"   - Latency: {lat:.3f}s | Sarcasm: {result.sarcasm_flag}")
        
        print(f"👉 KẾT QUẢ ASPECT:")
        if result.aspects:
            for a in result.aspects:
                print(f"   ✓ {a.aspect.upper()}: {a.sentiment.upper()} (Score: {a.confidence:.2%})")
        else:
            print("   ✗ Không tìm thấy aspect nào.")
            
        print("-" * 80)
        
        is_pass, reason = evaluate_result(result, item.get("expected_aspects", {}))
        if is_pass:
            print(f"✅ ĐÁNH GIÁ: PASS")
            passed_count += 1
        else:
            print(f"❌ ĐÁNH GIÁ: FAILED")
            print(f"   Lý do: {reason}")
            
        print("="*80 + "\n")

    print(f"\n🎉 ĐÃ HOÀN THÀNH TOÀN BỘ KỊCH BẢN TEST! ({passed_count}/{len(scenarios)} PASS)")

if __name__ == "__main__":
    main()
