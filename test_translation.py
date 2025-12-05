# test_translation.py - Standalone test for Chinese to English translation
# Uses OpenAI-compatible API wrapper to call Qwen model

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError(
        "DASHSCOPE_API_KEY not found in .env file. "
        "Please get your key at https://dashscope.console.aliyun.com/api-keys"
    )

# Initialize OpenAI client with DashScope base URL
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

def translate_chinese_to_english_openai(chinese_text):
    """
    Translate Chinese text to English using Qwen model via OpenAI-compatible API

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

        # Call Qwen model using OpenAI-compatible interface
        completion = client.chat.completions.create(
            model="qwen-plus",  # You can also use: qwen-turbo, qwen-max, qwen-max-longcontext
            messages=[
                {
                    "role": "user",
                    "content": f"Translate the following Chinese text into fluent, natural English. Be complete and do not summarize:\n\n{chinese_text}"
                }
            ],
            temperature=0.7,
            top_p=0.9,
            max_tokens=actual_max
            
        )

        # Extract translated text
        translated = completion.choices[0].message.content.strip()

        # Display token usage info
        if completion.usage:
            print(f"\nToken Usage:")
            print(f"  - Prompt tokens: {completion.usage.prompt_tokens}")
            print(f"  - Completion tokens: {completion.usage.completion_tokens}")
            print(f"  - Total tokens: {completion.usage.total_tokens}")

        return translated

    except Exception as e:
        return f"Translation failed: {str(e)}"


# ========================================
# Test Cases
# ========================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Chinese to English Translation (OpenAI Wrapper)")
    print("=" * 60)

    # Test Case 1: Simple greeting
    test_text_1 = "你好，世界！今天天气真好。"
    print(f"\n📝 Test 1 - Simple Text:")
    print(f"Chinese: {test_text_1}")
    result_1 = translate_chinese_to_english_openai(test_text_1)
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
    result_2 = translate_chinese_to_english_openai(test_text_2.strip())
    print(f"English: {result_2}")

    # Test Case 3: Technical content
    test_text_3 = "机器学习是人工智能的一个分支，它使用算法和统计模型来让计算机系统从数据中学习和改进。"
    print("\n" + "=" * 60)
    print(f"\n📝 Test 3 - Technical Text:")
    print(f"Chinese: {test_text_3}")
    result_3 = translate_chinese_to_english_openai(test_text_3)
    print(f"English: {result_3}")

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)