# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utils.myagent import Cardiologist, Psychologist, MultidisciplinaryTeam
from dotenv import load_dotenv
import json, os
from huggingface_hub import InferenceClient


# 加载 hf.env 文件
load_dotenv("hf_1.env", override=True) # 把文件里的变量写进环境变量 os.environ 里。

# read the medical report
# with open("Medical Reports/medical_report_english.txt", "r") as file:
with open("Medical Reports/medical_report_chinese.txt", "r") as file: # 打开文件后，会把文件对象赋值给 file 变量。
    medical_report = file.read() # 一次性读取整个文件的内容，返回一个 字符串


agents = {
    "Cardiologist": Cardiologist(medical_report),
    "Psychologist": Psychologist(medical_report)
}

# Function to run each agent and get their response
def get_response(agent_name, agent):
    response = agent.run()
    return agent_name, response

# Run the agents concurrently and collect responses
responses = {}
with ThreadPoolExecutor() as executor: # ThreadPoolExecutor()：Python 的线程池 . 作用：让多个 agent 可以并行运行，节省时间
    futures = {executor.submit(get_response, name, agent): name for name, agent in agents.items()}
    # Python 字典推导式的通用形式：{key_expression: value_expression for variable(s) in iterable if condition}
    # futures 最终会变成 {
    # <Future 对象1>: "Cardiologist",
    # <Future 对象2>: "Psychologist"
    # }
    # executor.submit(function, *args)：将函数提交给线程池执行，返回一个 Future 对象, "Cardiologist" 和 "Psychologist" 的 get_response 函数被并发执行
    for future in as_completed(futures):
        agent_name, response = future.result() # as_completed(futures) 会按完成顺序迭代线程池中的 Future
        responses[agent_name] = response

team_agent = MultidisciplinaryTeam(
    cardiologist_report=responses["Cardiologist"],
    psychologist_report=responses["Psychologist"]
)

# Run the MultidisciplinaryTeam agent to generate the final diagnosis
final_diagnosis = team_agent.run()
final_diagnosis_text = "### Final Diagnosis:\n\n" + final_diagnosis
txt_output_path = "results/final_diagnosis.txt"

# Ensure the directory exists
os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)

# Write the final diagnosis to the text file
with open(txt_output_path, "w") as txt_file:
    txt_file.write(final_diagnosis_text)

print(f"Final diagnosis has been saved to {txt_output_path}")





# # 从环境变量获取 Hugging Face Token
# hf_token = os.getenv("HF_TOKEN")
# if not hf_token:
#     raise ValueError("HF_TOKEN 未正确加载！请检查 hf.env 文件")
# # 初始化客户端
# client = InferenceClient(token=hf_token)


