import time
import base64
import io
import random
from openai import OpenAI
from functions import memory
from PIL import Image

# --- 原始配置 ---
ragflow_api_key = "ragflow-JkZGM1MjI0OGZiNzExZjBiMmE5N2E4ZG"
dialog_id = "36521dfebaea11f09fa9c24fd1ca77ef"
ragflow_address = "127.0.0.1"
model = "deepseek-reasoner"
client = OpenAI(api_key=ragflow_api_key, base_url=f"http://{ragflow_address}/api/v1/chats_openai/{dialog_id}")

stream = False
reference = False
memory_length = 1
print(f"\n[INFO] Use stream: {stream}\n[INFO] Use reference: {reference}\n[INFO] Memory length: {memory_length}")

system_messages = """
You are an infrared image processor for monitoring power battery conditions. An input will be provided:
1. Infrared image data encoded as a base64 string.
"""

# 初始化对话历史
messages = [
    {"role": "system", "content": system_messages},
    {"role": "user", "content": "Analyze the battery condition based on the infrared image."}
]

print("\n[INFO] Risk monitor starting.")

def load_infrared_image(file_path):
    """加载红外图像并转换为base64字符串。"""
    with Image.open(file_path) as img:
        with io.BytesIO() as output:
            img.save(output, format="PNG")
            encoded_string = base64.b64encode(output.getvalue()).decode('utf-8')
    return encoded_string


while True:
    # 读取并编码红外图像
    infrared_image_encoded = load_infrared_image('/home/lab-server/Projects/JMS2026/infrared.png')

    # 将输入整合到对话历史中
    messages.append({"role": "user", "content": infrared_image_encoded})

    #print(messages)
    print("\n[INFO] Sending request to the model...")
    start_time = time.time()

    # 模型调用
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=128000,
        stream=stream
    )

    elapsed_time = time.time() - start_time

    if hasattr(response, 'choices') and response.choices and response.choices[0].message:
        assistant_reply = response.choices[0].message.content
    else:
        assistant_reply = "Sorry, I could not get a response from the model."

    # 输出结果
    print("-" * 20)
    print(f"Time elapsed: {elapsed_time:.3f}s")
    print("\nAssistant:")
    print(assistant_reply)
    print("-" * 20)

    # 更新历史记录
    messages = memory.trim_messages(messages, memory_length)

    # 暂停以固定频率采样（可选）
    time.sleep(10)