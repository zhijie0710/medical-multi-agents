import os
from typing import TypedDict
from dotenv import load_dotenv

from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.graph import StateGraph, START, END



# ========== HF Model Wrapper (LangChain-Compatible) ==========
class HFChatModel:
    """
    A simple wrapper to replicate ChatOpenAI-style interface using HuggingFace InferenceClient.
    """
    def __init__(self, model_name="openai/gpt-oss-120b", temperature=0):
        self.client = InferenceClient(model_name) # InferenceClient 默认会去查找环境变量：默认读取 os.environ["HF_TOKEN"]
        self.temperature = temperature

    def invoke(self, prompt):
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=self.temperature,
        )
        # HuggingFace returns: response.choices[0].message
        return response.choices[0].message["content"]
    
# 初始化全局 LLM
llm = HFChatModel(
    model_name="openai/gpt-oss-120b",
    temperature=0
)

class Agent:
    def __init__(self, medical_report=None, role=None, extra_info=None):
        self.medical_report = medical_report
        self.role = role
        self.extra_info = extra_info
        self.prompt_template = self.create_prompt_template()

        self.model = HFChatModel(
            model_name="openai/gpt-oss-120b",
            temperature=0
        )

    
    def create_prompt_template(self):
        # --- Integrator Agent (MDT) ---
        if self.role == "MultidisciplinaryTeam":
            template = f"""
                Act like a multidisciplinary team consisting of a Cardiologist and a Psychologist.

                Task:
                - Integrate both specialist reports.
                - Provide exactly **3 possible diagnoses**.
                - For each diagnosis, explain briefly:
                    1. Why this diagnosis is plausible
                    2. Whether the cause is cardiac, psychological, or mixed.

                Output:
                - Bullet list with 3 items.
                - translate the final output to Chinese.

                Cardiologist Report:
                {self.extra_info.get('cardiologist_report', '')}

                Psychologist Report:
                {self.extra_info.get('psychologist_report', '')}
            """
            return PromptTemplate.from_template(template)

        # --- Specialist Agents ---
        templates = {
            "Cardiologist": """
                Act like a cardiologist.

                Task:
                - Analyze the patient's ECG, labs, symptoms, and cardiac history.
                - Identify possible cardiac causes: arrhythmias, coronary issues, structural problems.
                - Recommend next steps (tests, monitoring).

                Output:
                - Cardiac causes + recommended next steps.

                Medical Report:
                {medical_report}
            """,

            "Psychologist": """
                Act like a psychologist.

                Task:
                - Analyze emotional and behavioral symptoms.
                - Identify possible mental health issues: anxiety, depression, trauma, stress-related disorders.
                - Recommend next steps.

                Output:
                - Possible psychological issues + next steps.

                Patient Report:
                {medical_report}
            """
        }
        return PromptTemplate.from_template(templates[self.role])


    # -------------------------
    # Run the Agent
    # -------------------------
    def run(self):
        print(f"{self.role} is running...")

        if self.role == "MultidisciplinaryTeam":
            prompt = self.prompt_template.format()
        else:
            prompt = self.prompt_template.format(medical_report=self.medical_report)

        try:
            response = self.model.invoke(prompt)
            return response
        except Exception as e:
            print("Error occurred:", e)
            return None

# ========== Specialized Agents ==========
class Cardiologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report=medical_report, role="Cardiologist")

class Psychologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report=medical_report, role="Psychologist")

class MultidisciplinaryTeam(Agent):
    def __init__(self, cardiologist_report, psychologist_report):
        super().__init__(
            role="MultidisciplinaryTeam",
            extra_info={
                "cardiologist_report": cardiologist_report,
                "psychologist_report": psychologist_report
            }
        )



class MedicalState(TypedDict):
    medical_report: str
    cardio_initial: str
    cardio_final: str
    psycho_initial: str
    psycho_final: str
    mdt_report: str

def cardiologist_node(state: MedicalState):
    agent = Cardiologist(state["medical_report"])
    result = agent.run()
    return {"cardio_initial": result}

def cardiologist_review_node(state: MedicalState):
    print("\n=== Cardiologist Initial Report ===\n")
    print(state["cardio_initial"])

    feedback = input("\nPlease enter doctor's feedback for the cardiologist report (enter 'None' if no changes): ")

    if feedback.strip().lower() in ["none", "无"]:
        final_version = state["cardio_initial"]
    else:
        prompt = f"""
        The following is the initial cardiologist report. Please revise it according to the doctor's feedback:

        Initial Report:
        {state['cardio_initial']}

        Doctor's Feedback:
        {feedback}
        """
        final_version = llm.invoke(prompt)

    return {"cardio_final": final_version}

def psychologist_node(state: MedicalState):
    agent = Psychologist(state["medical_report"])
    result = agent.run()
    return {"psycho_initial": result}

def psychologist_review_node(state: MedicalState):
    print("\n=== Psychologist Initial Report ===\n")
    print(state["psycho_initial"])
    feedback = input("\nPlease enter doctor's feedback for the psychologist report (enter 'None' if no changes): ")
    if feedback.strip().lower() in ["none", "无"]:
        final_version = state["psycho_initial"]
    else:
        prompt = f"""
            The following is the initial psychologist report. Please revise it according to the doctor's feedback:
            Initial Report:
            {state['psycho_initial']}
            Doctor's Feedback:
            {feedback}
            """
        final_version = llm.invoke(prompt)
    return {"psycho_final": final_version}

def mdt_node(state: MedicalState):
    agent = MultidisciplinaryTeam(
        cardiologist_report=state["cardio_final"],
        psychologist_report=state["psycho_final"],
    )
    result = agent.run()
    return {"mdt_report": result}

# -------------------------
# Build Graph Workflow
# -------------------------
def build_medical_workflow():
    workflow = StateGraph(MedicalState)

    workflow.add_node("cardio", cardiologist_node)
    workflow.add_node("cardio_review", cardiologist_review_node)
    workflow.add_node("psych", psychologist_node)
    workflow.add_node("psych_review", psychologist_review_node)
    workflow.add_node("mdt", mdt_node)

    workflow.add_edge(START, "cardio")
    workflow.add_edge("cardio", "cardio_review")
    workflow.add_edge("cardio_review", "psych")
    workflow.add_edge("psych", "psych_review")
    workflow.add_edge("psych_review", "mdt")
    workflow.add_edge("mdt", END)

    return workflow.compile()

