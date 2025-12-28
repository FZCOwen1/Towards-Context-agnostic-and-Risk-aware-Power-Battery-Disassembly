import time
import random
import base64
import io
from openai import OpenAI
from functions import memory
from PIL import Image  # Make sure you have Pillow installed

# --- 原始配置 ---
ragflow_api_key = "ragflow-JkZGM1MjI0OGZiNzExZjBiMmE5N2E4ZG"
dialog_id = "487c8e7ec5df11f0a8396a5a8e4615b0"
ragflow_address = "127.0.0.1"
model = "deepseek-chat"
client = OpenAI(api_key=ragflow_api_key, base_url=f"http://{ragflow_address}/api/v1/chats_openai/{dialog_id}")

stream = False
reference = False
memory_length = 0
print(f"\n[INFO] Use stream: {stream}\n[INFO] Use reference: {reference}\n[INFO] Memory length: {memory_length}")

system_messages = """
You are a thermal & expansion force & voltage monitor for a power battery under disassembly tasks. You will be given 3 inputs, which are:
1. A set of temperature readings (t) per second from a thermal sensor for the last 10 seconds.
2. A set of expansion force readings (f) per second from a force sensor installed on the side of the battery cell for the last 10 seconds.
3. A set of voltage readings (v) from the voltage sensor per second for the last 10 seconds.
"""

assistant_messages_1 = """
#### I. Temperature stage (t is the temperature)
 1. **Initial Stage (if t<125°C for all t)
 2. **Acceleration Stage (if 125°C<t<180°C for all t)
 3. **Runaway Stage (if t>180°C for all t)

 Find the temperature stage.
 ***Under 1 and 2, the situation is still reversible. Under 3, it’s irreversible*** 

 #### II. Multi-signal Based Early Warning Method 
 **Three-level Warning Strategy (f is expansion force, v is voltage, t is temperature)**
 First Level only if average expansion force increase rate > 5N/s (earliest warning signal). 
 Second Level only if average voltage drop rate >0.02V/s (internal short circuit warning). 
 Third Level only if average temperature increase rate >0.2°C/s (thermal runaway warning). 
 Zero level if not first/second/third level
 where n = 1,2,3,.....9
 ***You must check the rule for each level and then give a final decision. Don't rush to conclusion before going through all of them!!***
 Find the warning level.
 ***Under level 1 and 2, the situation is still reversible. Under 3, it’s irreversible.***

***Strictly follow the output format and don't give any other intermediate reasoning***:
[Zero level: force rate < 5, voltage rate < 0.02, temperature rate < 0.2]
or
[First level: force rate > 5, voltage rate < 0.02, temperature rate < 0.2]
or
[Second level: voltage drop rate > 0.02, temperature rate < 0.2]
or
[Third level: temperature rate > 0.2]
[stage x: temperature within (a,b)]
***If and only if at zero level and at initial stage***
[
    "Safe. Sensor readings normal." 

] or
***If acceleration stage or first/second level***
[
    "Dangerous! Potential risks detected." (Temperature stage x (why), warning level x (why). Give brief measures.)
] or
***If runaway stage or third level***
[
    "Fatal failure! Leave the working area immediately." (Temperature stage x (why), warning level x (why). Give brief measures.)
]
"""

assistant_messages_2 = """
Strictly follow the output format and refer to the three stages of temperature and three levels of warning in the {knowledge}:
[
    "Safe. Sensor readings normal." (If no potential risk has been captured)

] or
[
    "Dangerous! Potential risks detected." Reason: (If still reversible. Give brief reasons.)
] or
[
    "Fatal failure! Leave the working area immediately." Reason: (If irreversible. Give brief reasons.)
]
"""

# 初始化对话历史
messages = [
    {"role": "system", "content": system_messages},
    {"role": "user", "content": "Analyze the battery condition."},
]

print("\n[INFO] Risk monitor starting.")


def simulate_temperature_readings(duration=10, frequency=1, change_rate=0.1):
    """模拟温度传感器读数"""
    readings = []
    current_temperature = random.uniform(20.0, 20.0)  # 初始温度
    for _ in range(duration * frequency):
        change = random.uniform(-change_rate, change_rate)
        current_temperature += change
        current_temperature = max(20.0, min(current_temperature,200.0))
        readings.append(round(current_temperature, 2))
    return readings


def simulate_force_sensor_readings(duration=10, frequency=1, change_rate=1.0):
    """模拟应变片读数"""
    readings = []
    current_force = random.uniform(1000.0, 1050.0)  # 初始力
    for _ in range(duration * frequency):
        change = random.uniform(0, change_rate)
        current_force += change
        current_force = max(1000.0, min(current_force, 5000.0))
        readings.append(round(current_force, 2))
    return readings


def simulate_voltage_readings(duration=10, frequency=1, change_rate=0.01):
    """模拟电压传感器读数"""
    readings = []
    current_voltage = random.uniform(12, 12)  # 初始电压
    for _ in range(duration * frequency):
        change = random.uniform(-change_rate, 0)
        current_voltage += change
        current_voltage = max(0.0, min(current_voltage, 12.0))
        readings.append(round(current_voltage, 2))
    return readings

while True:
    # 获取10秒的传感器读数
    temperature_readings = simulate_temperature_readings()
    force_sensor_readings = simulate_force_sensor_readings()
    voltage_readings = simulate_voltage_readings()
    print("force readings:", force_sensor_readings)
    print("voltage readings:", voltage_readings)
    print("temperature readings:", temperature_readings)
    print("\n"+"-------------------")
    # Load and encode the infrared image (replace 'infrared_image.png' with your actual file path)
    #infrared_image_encoded = load_infrared_image('/home/lab-server/Projects/JMS2026/infrared.png')
    '''
    # 准备输入数据
    combined_content = "Temperature readings (30s at 1Hz):\n" + \
                       "\n".join(map(str, temperature_readings)) + \
                       "\nExpansion force readings (30s at 1Hz):\n" + \
                       "\n".join(map(str, force_sensor_readings)) + \
                       "\nVoltage readings (30s at 1Hz):\n" + \
                       "\n".join(map(str, voltage_readings))

    # 将输入整合到对话历史中
    messages.append({"role": "user", "content": combined_content})

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

    if response.choices and response.choices[0].message:
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
    '''
    # 暂停以固定频率采样（可选）
    time.sleep(2)  # 这行可以根据需要调整间隔时间