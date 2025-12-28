import time
import base64
import io
from openai import OpenAI
from functions import memory
from functions.RRT.rrt3d import plan_pick_and_place_sequence_3d, animate_transport_3d, pos_center_from_xy
from functions.RRT.planner_rrt_resolver import split_objects_and_static_obstacles
from PIL import Image
import os

# ===================== 配置区 =====================
AUTO_MODE = True  # True=自动从固定文件读；False=交互式输入
IF_RRT = False
IF_RRT_ANIMATION = True
CAPTURE_DIR = "./functions/detection/captures"

IMAGE_FIXED_NAME = "yolo_preprocessed.png"
TXT_ALL_NAME = "yolo_all.txt"         # 场景中识别出的所有物体
TXT_FILTER_NAME = "yolo_filtered.txt" # 场景中只含“目标物体”类别

# 若文本里未提供 “Target … x y z”，则使用默认终点位姿（目前不发给LLM，仅保留以备扩展）
DEFAULT_TARGET_POSITION = (300, 300, 300)
# ===================== 配置区 =====================

# --- 原始配置 ---
ragflow_api_key = ""
dialog_id = ""
ragflow_address = ""
assistant_reply = None

model = "deepseek-chat"
client = OpenAI(api_key=ragflow_api_key, base_url=f"http://{ragflow_address}/api/v1/chats_openai/{dialog_id}")

stream = True
reference = True
memory_length = 0
print(f"\n[INFO] Use stream: {stream}\n[INFO] Use reference: {reference}\n[INFO] Memory length: {memory_length}")
print(f"[INFO] AUTO_MODE: {AUTO_MODE}")

# ====== 提示词（保持不变） ======
system_messages = """
You are a task & motion planner for a robot arm under EV battery disassembly tasks. You will be given 4 inputs, which are:
1. A pre-processed image that has gone through YOLO/SAM for segmentation of parts/tools. There are bounding boxes around these objects of interest.
2. The coordinates of the parts/tools to interact with within the whole step.
3. The coordinates of the bounding boxes around the detected obstacles.
4. The coordinate of the target position.

You have access to a motion planning knowledge base {knowledge}, which states general motion planning scenes and rules. 
Please refer to the knowledge base to generate sub-steps of the given complete step. You need to follow the following reasoning methods:
1. Rearrange the sequence of coordinates of the given parts/tools to interact with based on the {knowledge}.
2. Follow the sequence in 1 to take coordinate as the starting point of each sub step one by one.
3. Follow the sequence in 2 to generate a trajectory between the starting point and the target position. Make sure the trajectories avoid the obstacles. Make sure all trajectories end at the target position.

Output:
**Only output the trajectories for all sub steps.**
**Please strictly follow the format given by the user.**
**Don't output anything other than the trajectories.**
"""

system_messages_2 = """
You are a task planner for a robot arm under EV battery disassembly tasks. You will be given 4 inputs, which are:
1. A pre-processed image that has gone through YOLO/SAM for segmentation of parts/tools. There are bounding boxes around these objects of interest.
2. The coordinates of the ***target objects***.

You have access to a motion planning knowledge base {knowledge}, which states general motion planning scenes and rules. 
Please refer to the General Disassembly Rules (Priority Order) to rearrange the sequence of coordinates of the target parts/tools to interact with based on the {knowledge}.
***Don't output coordinates of the obstacles!!***

Output:
**Only output the reorganized coordinates and the type of the target objects. The obstacles and destination are only for reference, don't output them.**
**Please strictly follow the format given by the user.**
**Don't output any other things.**
"""

system_messages_3 = """
You are a task planner for a robot arm under EV battery disassembly tasks. You will be given 3 inputs in the format of list strictly following the sequence below, which are:
1. ***List 1***: The coordinates and names of the ***detected objects***.
2. ***List 2***: The names of the ***required objects***.
3. ***List 3***: The coordinates and names of the ***detected targets***.

You have access to a motion planning knowledge base {knowledge}, which states general motion planning scenes and rules. 
Please refer to the General Disassembly Rules (Priority Order) to rearrange the sequence of coordinates in ***List 3*** to interact with based on the {knowledge}.
***Don't output coordinates of the obstacles!!***

Output:
**First output whether the task is not ready for motion planning.**
**If the task is ready for motion planning: Only output the reorganized coordinates and the type of the target objects. The obstacles and destination are only for reference, don't output them.**
**Please strictly follow the format given by the user.**
**Don't output any other things.**
"""

assistant_messages = """
[
    ((x1,y1,z1),(x2,y2,z2),(x3,y3,z3),(x4,y4,z4),(x5,y5,z5)...(target position coordinate)),
    ((x1,y1,z1),(x2,y2,z2),(x3,y3,z3),(x4,y4,z4),(x5,y5,z5)...(target position coordinate)),
    ((x1,y1,z1),(x2,y2,z2),(x3,y3,z3),(x4,y4,z4),(x5,y5,z5)...(target position coordinate))
]
"""

assistant_messages_2 = """
[
    Reorganized coordinates of the target objects:
    [{'type':'type1','pos':(x1,y1,z1)},{'type':'type2','pos':(x2,y2,z2)},{'type':'type3','pos':(x3,y3,z3)}......]
    Why:
]
"""

assistant_messages_3 = """
you need to check the following steps in order to check whether the task is ready for motion planning:
Step 1. Extract the names in ***List 2*** and the names in ***List 1***. Then compare these names. Check whether there are more types of names in ***List 1*** (ignore number of names). If so, Step 1 fails. 
Step 2. If all names are included, check all the coordinates and objects in ***List 1***, and calculate all pairs of distance in X direction within (X, Y, Z) for every objects. Find all objects whose distances between it and ***all*** other objects are > 60 in value and whether its name is also in ***List 3***.  If you cannot find any object like this, Step 2 fails.
If Step 1 and 2 both got passed, first explain how 1 and 2 are verified in the following format:
[
    Names in List 1:
    Names in List 2:
    Names in List 3:
    Distances in List 1:
]
then record all the objects with their coordinates in ***List 1*** that fulfill Step 2 as ***List 4***
and then follow the following format
[
    Reorganized coordinates of the coordinates only in ***List 4***: (only if Abort = False)
    [{'type':'type1','pos':(x1,y1,z1)},{'type':'type2','pos':(x2,y2,z2)},{'type':'type3','pos':(x3,y3,z3)}......]
]
else: explain why step 1 or 2 fails
Finally you should judge IF_RRT. If both step 1 and step 2 got passed, IF_RRT=True; else IF_RRT=False.
"""

# 初始化对话历史（保持不变）
messages = [
    {"role": "system", "content": system_messages_3},
    {"role": "user", "content": "First analyze whether the task is ready for motion planning. Then refer to the Target obstacle coordinate reorganization rules and the General Disassembly Rules (Priority Order) to reorganize the coordinates. Strictly follow the format given by the user."},
    {"role": "assistant", "content": assistant_messages_3},
]
def extract_if_rrt(output: str) -> bool:
    lines = output.strip().splitlines()

    target_line = None
    for line in reversed(lines):
        if "IF_RRT" in line:
            target_line = line.strip()
            break

    if target_line is None:
        raise ValueError("❌ No IF_RRT line found in model output!")

    # Normalize spaces
    normalized = target_line.replace(" ", "")

    if "IF_RRT=True" in normalized:
        return True
    elif "IF_RRT=False" in normalized:
        return False
    else:
        raise ValueError(f"❌ Invalid IF_RRT line format: {target_line}")


def get_image_base64(image_path):
    """图像 -> Base64 data URL"""
    try:
        if not os.path.exists(image_path):
            print(f"[ERROR] Image file not found at: {image_path}")
            return None
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            buffered = io.BytesIO()
            rgb_image.save(buffered, format="JPEG", quality=90)
            img_str = base64.b64encode(buffered.getvalue())
            return f"data:image/jpeg;base64,{img_str.decode('utf-8')}"
    except Exception as e:
        print(f"[ERROR] Image processing error '{image_path}': {e}")
        return None

def parse_yolo_txt_fixed(txt_path):
    """
    解析固定命名的 YOLO 文本：
    第一行：时间戳（读取但不用于配对）
    其余每行：<Class(with spaces allowed)> <x> <y> <z>
    返回：
      items: list[(cls, (x,y,z))]
      maybe_target: (x,y,z) | None
      ts: 第一行时间戳字符串 | None
    """
    items, maybe_target, ts = [], None, None
    if not txt_path or not os.path.exists(txt_path):
        return items, maybe_target, ts

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if not lines:
        return items, maybe_target, ts

    ts = lines[0]  # 记录但不用于逻辑
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) < 4:
            continue
        try:
            x = float(parts[-3]); y = float(parts[-2]); z = float(parts[-1])
        except ValueError:
            continue
        cls = " ".join(parts[:-3])

        if cls.lower() == "target":
            maybe_target = (x, y, z)
        else:
            items.append((cls, (x, y, z)))

    return items, maybe_target, ts

def _fmt_xyz_tuple_commas_2dec(t):
    """
    输出形如：(287.66,-574.48,185.00)，总是保留两位小数，逗号分隔
    """
    x, y, z = t
    return f"({float(x):.2f},{float(y):.2f},{float(z):.2f})"

print("\n[INFO] Multi-modal chat started.")

if AUTO_MODE:
    # 固定文件路径
    img_path = os.path.join(CAPTURE_DIR, IMAGE_FIXED_NAME)
    all_path = os.path.join(CAPTURE_DIR, TXT_ALL_NAME)
    filter_path = os.path.join(CAPTURE_DIR, TXT_FILTER_NAME)

    # 读取图片（是否发给LLM看你需求，这里不强制附上）
    _ = get_image_base64(img_path)

    # 读取两个txt
    if not os.path.exists(all_path):
        raise FileNotFoundError(f"[FATAL] '{TXT_ALL_NAME}' not found in {CAPTURE_DIR}")
    if not os.path.exists(filter_path):
        raise FileNotFoundError(f"[FATAL] '{TXT_FILTER_NAME}' not found in {CAPTURE_DIR}")

    all_items, all_target, ts_all = parse_yolo_txt_fixed(all_path)
    filtered_items, filtered_target, ts_filter = parse_yolo_txt_fixed(filter_path)

    # 自动推断类别集合（目标=filtered里的类别；障碍物=all-目标）
    target_classes = {cls for cls, _ in filtered_items}
    obstacle_classes = {cls for cls, _ in all_items if cls not in target_classes}

    # 从 all_items 中取出属于目标类别的条目（保留原始类别名）
    target_entries = [(cls, pt) for cls, pt in all_items if cls in target_classes]

    # 构造你要求的输入： [{'type':'type1','pos':(x,y,z)}, ...]
    payload_items = []
    for cls, pt in target_entries:
        pos_str = _fmt_xyz_tuple_commas_2dec(pt)
        # 直接使用原始 cls 作为 type，保留大小写和空格
        payload_items.append(f"{{'type':'{cls}','pos':{pos_str}}}")

    if payload_items:
        text_block_filtered = "[" + ",".join(payload_items) + "]"
    else:
        text_block_filtered = "[]"
##############################################################################
    # 自动推断类别集合（所有检测到的物体）
    all_classes = {cls for cls, _ in all_items}
    # 从 all_items 中取出属于目标类别的条目（保留原始类别名）
    all_entries = [(cls, pt) for cls, pt in all_items if cls in all_classes]
    # 构造你要求的输入： [{'type':'type1','pos':(x,y,z)}, ...]
    payload_items = []
    for cls, pt in all_entries:
        pos_str = _fmt_xyz_tuple_commas_2dec(pt)
            # 直接使用原始 cls 作为 type，保留大小写和空格
        payload_items.append(f"{{'type':'{cls}','pos':{pos_str}}}")

    if payload_items:
        text_block_all = "[" + ",".join(payload_items) + "]"
    else:
        text_block_all = "[]"


    # 加到对话历史（不改提示词与assistant示例）
    messages.append({"role": "user", "content": text_block_all})
    messages.append({"role": "user", "content": "[Cell]"}) #List 2 -> required parts in this step
    messages.append({"role": "user", "content": text_block_filtered})

    print("[DEBUG] Final messages sent to LLM:\n", messages)

    # 发请求（不传 extra_body，避免阻塞）
    print("\n[INFO] Sending request to the model (AUTO_MODE)...")
    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=128000,
        stream=False
    )
    elapsed_time = time.time() - start_time

    if response and response.choices and response.choices[0].message and response.choices[0].message.content:
        assistant_reply = response.choices[0].message.content
    else:
        assistant_reply = "Sorry, I could not get a response from the model."

    print("-" * 20)
    print(f"Time elapsed: {elapsed_time:.3f}s")
    print("\nAssistant:")
    print(assistant_reply)
    print("-" * 20)

    IF_RRT = extract_if_rrt(assistant_reply)
    #IF_RRT = True

    if IF_RRT:
        targets = [
            (300, -160, 185),
            (300, -230, 185),
            (300, -300, 185),
        ]
        size_lookup = {
            'Cell': (175, 45, 185),
            'Side Frame': (175, 10, 170),
            'Reserve Box': (200, 65, 200),
            'Robot': (250,300,500)
        }

        bounds = (-800, -400, 165, 600, 400, 400)
        start_top = (-133, -307, 247)  # 机器人末端 TOP-CENTER
        objects, static_obstacles = split_objects_and_static_obstacles(assistant_reply,
        "/home/lab-server/Projects/RagFlow/functions/detection/captures/yolo_all.txt")
        print(objects)
        static_obstacles.append({'type': 'Robot', 'pos': (0,50,500)}) #添加机械臂本身作为障碍物
        segments, segments_meta, placed, path_infos = plan_pick_and_place_sequence_3d(
            start_top=start_top,
            objects=objects,
            targets=targets,
            obstacles=static_obstacles,
            size_lookup=size_lookup,
            out_dir="rrt3d_demo",
            bounds=bounds,
            step_size=10.0,
        )
        # print("✅ 规划完成，3D 静态图像已输出到 rrt3d_demo/ 目录。")
        if IF_RRT_ANIMATION:
            static_obstacles_center = [
                (pos_center_from_xy(o['type'], (o['pos'][0], o['pos'][1]), size_lookup), size_lookup[o['type']]) for o
                in static_obstacles]
            animate_transport_3d(
                segments_meta=segments_meta,
                static_obstacles_center=static_obstacles_center,
                out_file_mp4="rrt3d_demo/transport.mp4",
                out_file_gif="rrt3d_demo/transport.gif",
                fps=10,
                interval_ms=30,
                frame_stride=1,
                max_frames=20000,
                bounds=bounds,
                elev=24, azim=45,
            )
            # print("✅ 3D 动画导出流程结束。")

else:
    # ======= 原交互模式（保留）=======
    print("[INFO] Manual mode: use /text, /image, /done, /exit")
    while True:
        session_texts = []
        session_images = []

        while True:
            user_input = input("\nYou (add all inputs then '/done'): ")

            if user_input.lower() == '/exit':
                print("[INFO] Exiting the program.")
                exit()

            if user_input.lower() == '/done':
                if not session_texts and not session_images:
                    print("[WARNING] No input provided in this session. Please add some content before sending to '/done'.")
                else:
                    break

            elif user_input.lower().startswith('/text '):
                text = user_input.split(' ', 1)[1]
                session_texts.append(text)
                print(f"[INFO] Added text: '{text[:50]}...'")

            elif user_input.lower().startswith('/image '):
                image_path = user_input.split(' ', 1)[1]
                image_url = get_image_base64(image_path)
                if image_url:
                    session_images.append(image_url)
                    print(f"[INFO] Added image: '{os.path.basename(image_path)}'")

            else:
                print("[WARNING] Invalid command. Use '/text ...', '/image ...', '/done', or '/exit'.")

        combined_content = "Text inputs:\n" + "\n".join(session_texts) + "\nImage URLs:\n" + "\n".join(session_images)
        messages.append({"role": "user", "content": combined_content})

        print("\n[INFO] Sending request to the model...")

        start_time = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=128000,
            stream=False
        )
        elapsed_time = time.time() - start_time

        if response and response.choices and response.choices[0].message and response.choices[0].message.content:
            assistant_reply = response.choices[0].message.content
        else:
            assistant_reply = "Sorry, I could not get a response from the model."

        print("-" * 20)
        print(f"Time elapsed: {elapsed_time:.3f}s")
        print("\nAssistant:")
        print(assistant_reply)
        print("-" * 20)

        messages = memory.trim_messages(messages, memory_length)
