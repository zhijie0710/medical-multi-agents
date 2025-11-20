# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utils.agent_humanfeedback import Cardiologist, Psychologist, MultidisciplinaryTeam, human_review
from dotenv import load_dotenv
import json, os
from huggingface_hub import InferenceClient


# 加载 hf.env 文件
load_dotenv("hf_1.env", override=True) # 把文件里的变量写进环境变量 os.environ 里。

# read the medical report
# with open("Medical Reports/medical_report_english.txt", "r") as file:
with open("Medical Reports/medical_report_chinese.txt", "r") as file: # 打开文件后，会把文件对象赋值给 file 变量。
    medical_report = file.read() # 一次性读取整个文件的内容，返回一个 字符串



# -------------------------
# Step 1: Cardiologist
# -------------------------
cardio_initial = Cardiologist(medical_report).run()
cardio_final = human_review(cardio_initial, "Cardiologist")

# -------------------------
# Step 2: Psychologist
# -------------------------
psych_initial = Psychologist(medical_report).run()
psych_final = human_review(psych_initial, "Psychologist")

# -------------------------


# Run the MultidisciplinaryTeam agent to generate the final diagnosis
final_diagnosis = MultidisciplinaryTeam(cardio_final, psych_final).run()
final_diagnosis_text = "### Final Diagnosis:\n\n" + final_diagnosis
txt_output_path = "humanfeedback_results/final_diagnosis.txt"

# Ensure the directory exists
os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)

# Write the final diagnosis to the text file
with open(txt_output_path, "w") as txt_file:
    txt_file.write(final_diagnosis_text)

print(f"Final diagnosis has been saved to {txt_output_path}")


