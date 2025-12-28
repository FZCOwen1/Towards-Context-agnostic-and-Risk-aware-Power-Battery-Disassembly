import time
from openai import OpenAI
from sympy.integrals.heurisch import components

from functions import memory, json_resolver
#from functions.detection import detect_realsense
from agent1_validation import calculate_accuracy
from ragflow_sdk import RAGFlow


ragflow_api_key = "ragflow-JkZGM1MjI0OGZiNzExZjBiMmE5N2E4ZG"
dialog_id = "20f65e9e8d5811f0b4a27a8dead3d5aa"
dialog_id_2 = "0143a88e984511f0ba6b86284a4c60ba" #For the model without knowledge base
ragflow_address = "127.0.0.1"
run_detect = False
model = "gemini-3.0-pro-preview"
client = OpenAI(api_key=ragflow_api_key, base_url=f"http://{ragflow_address}/api/v1/chats_openai/{dialog_id}")

stream = True
reference = False
memory_length = 0
print(f"\n[INFO] Use stream: {stream}\n[INFO] Use reference: {reference}\n[INFO] Memory length: {memory_length}")

system_messages = """
You are a **Task distribution core**, a specialized knowledge base assistant responsible for providing accurate tasks distributed to human and robot arm. The task scene is human robot collaboration (HRC) EV battery module disassembly.

The knowledge base is: {knowledge}
# Core Principles
1. **Rapid Output**
Read the given manual of a specific battery module first.
Retrieve from the knowledge base using the retrieval tool to learn the task distribution principles and then rapidly output distributed tasks following the format given by the user.
2. **Knowledge Base Only**: The task distribution principles should be 100% based on the knowledge base. 
3. **Content Creation**: Take the knowledge base as a dictionary that you refer to and you can generate content based on the information that you retrieved. 
4. **Source Transparency**: Always indicate when information comes from the knowledge base vs. when it's unavailable.

# Response Guidelines
Please strictly follow the format given by the user.

# Responsibility
Human or Robot Arm 

Be strict with the format.
**Only output the final distribution results**
**Don't show the reasoning or other procedures**
"""

#system_messages for model without knowledge base
system_messages_2 = """
You are a **Task distribution core**, a specialized assistant responsible for providing accurate tasks distributed to human and robot arm. The task scene is human robot collaboration (HRC) EV battery module disassembly.

# Core Principles
1. **Rapid Output**

Read the given manual of a specific battery module first. Rapidly output distributed tasks following the format given by the user. 

# Response Guidelines
Please strictly follow the format given by the user.

#Responsibility
Human or Robot Arm 

Be strict with the format.
**Only output the final distribution results**
**Don't show the reasoning or other procedures**
"""

assistant_messages = """
{
  "Disassembly Item": "",
  "Number of Steps": "",
  "Steps": {
    "1": {
      "Description": "",
      "Detail": "",
      "Responsibility": "Human" or "Robot Arm",
      "Parts": [
        {
         "Name": ""
        },
      ],
      "Tools": [
            "",
      ]
     }
   }
}
"""

messages = [
    #{"role": "system", "content": system_messages},
    {"role": "user", "content": "Distribute the task."},
    {"role": "assistant", "content": assistant_messages}
]


while True:
    user_input = input("User: ")

    # 记录开始时间
    start_time = time.time()

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=128000,
        stream=stream,
        # extra_body={"reference": reference}
    )

    # 计算耗时
    elapsed_time = time.time() - start_time

    #print(response)
    # assistant_reply = response.choices[0].message.content

    if stream:
        # for chunk in response:
        #     print(chunk)
        #     if reference and chunk.choices[0].finish_reason == "stop":
        #         # print(f"Reference:\n{chunk.choices[0].delta.reference}")
        #         print(f"Final content:\n{chunk.choices[0].delta.final_content}")

        full_content = ""
        for chunk in response:
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                full_content += delta_content

        content = json_resolver.LLM_Output_Resolver(full_content)

        accuracy = calculate_accuracy(content)
        print(f"Accuracy: {accuracy}")
        print(f"Time elapsed (s): {elapsed_time}")
        print("Final content:\n", full_content)
        part_components = json_resolver.Parts_Names_Resolver(content)
        if run_detect:
            detect_realsense.detect_realsense(object_model_path=r"runs/detect/train/weights/best.pt",
            human_model_path=r"runs/detect/train9/weights/best.pt",
            filtered_components=part_components,
            conf=0.6,
            show_frame=True,
            is_filtered=True,
            )


    else:
        #print(f"Accuracy: {accuracy}")
        print(f"Time elapsed (s): {elapsed_time}")
        print(response.choices[0].message.content)
        # if reference:
            # print(response.choices[0].message.reference)

    # messages = memory.trim_messages(messages, memory_length)
