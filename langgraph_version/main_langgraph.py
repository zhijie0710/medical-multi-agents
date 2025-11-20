from dotenv import load_dotenv
from agent_langgraph import build_medical_workflow

# -------------------------
# Load HuggingFace API key from hf.env
# -------------------------
load_dotenv("hf.env", override=True)  # Load environment variables into os.environ

# -------------------------
# Read medical report file
# -------------------------
with open("Medical Reports/panic.txt", "r") as file:
    medical_report = file.read()

# -------------------------
# Build and run workflow
# -------------------------
app = build_medical_workflow()

result_state = app.invoke({
    "medical_report": medical_report
})

print("\n=== Final MDT Medical Report ===\n")
print(result_state["mdt_report"])