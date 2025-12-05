# test_translation_dashscope.py - Test using original DashScope SDK
# This is the version from your original app.py

import os
import dashscope
from dashscope import Generation
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError(
        "DASHSCOPE_API_KEY not found in .env file. "
        "Please get your key at https://dashscope.console.aliyun.com/api-keys"
    )

dashscope.api_key = DASHSCOPE_API_KEY

def translate_chinese_to_english_api(chinese_text):
    """
    Original translation function from app.py (DashScope SDK)

    Args:
        chinese_text (str): Chinese text to translate

    Returns:
        str: English translation
    """
    try:
        # Estimate required output length
        estimated_output_tokens = max(512, int(len(chinese_text) * 1.3))  # Rough scaling
        actual_max = min(estimated_output_tokens, 2048)  # Cap at safe limit

        print(f"Input length: {len(chinese_text)} chars")
        print(f"Max tokens: {actual_max}")
        print("-" * 60)

        response = Generation.call(
            model="qwen-plus",
            messages=[
                {
                    "role": "user",
                    "content": f"Translate the following Chinese text into fluent, natural English. Be complete and do not summarize:\n\n{chinese_text}"
                }
            ],
            temperature=0.7,
            top_p=0.9,
            max_tokens=actual_max,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )

        if response.status_code == 200:
            translated = response.output.choices[0].message.content.strip()

            # Display token usage info
            if hasattr(response.usage, 'total_tokens'):
                print(f"\nToken Usage:")
                print(f"  - Total tokens: {response.usage.total_tokens}")

            return translated
        else:
            error_msg = response.code, response.message
            return f"Translation API error: {error_msg}"

    except Exception as e:
        return f"Translation failed: {str(e)}"


# ========================================
# Test Cases
# ========================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Chinese to English Translation (DashScope SDK)")
    print("=" * 60)

    # Test Case 1: Simple greeting
    test_text_1 = "你好，世界！今天天气真好。"
    print(f"\n📝 Test 1 - Simple Text:")
    print(f"Chinese: {test_text_1}")
    result_1 = translate_chinese_to_english_api(test_text_1)
    print(f"English: {result_1}")

    # Test Case 2: Longer paragraph
    test_text_2 = """
    人工智能正在改变我们的生活方式。从智能手机到自动驾驶汽车，
    AI技术已经渗透到我们日常生活的方方面面。随着技术的不断进步，
    我们可以期待更多创新的应用出现。
    """
    print("\n" + "=" * 60)
    print(f"\n📝 Test 2 - Longer Paragraph:")
    print(f"Chinese: {test_text_2.strip()}")
    result_2 = translate_chinese_to_english_api(test_text_2.strip())
    print(f"English: {result_2}")

    # Test Case 3: Technical content
    test_text_3 = "机器学习是人工智能的一个分支，它使用算法和统计模型来让计算机系统从数据中学习和改进。"
    print("\n" + "=" * 60)
    print(f"\n📝 Test 3 - Technical Text:")
    print(f"Chinese: {test_text_3}")
    result_3 = translate_chinese_to_english_api(test_text_3)
    print(f"English: {result_3}")

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)