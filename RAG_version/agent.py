import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from huggingface_hub import InferenceClient
import faiss
import pickle
import numpy as np

class MyRetriever:
    def __init__(self, index_path="medical_docs.index", docs_path="medical_docs.pkl", model_name="BAAI/bge-small-en", token=None):
        self.client = InferenceClient(token=token)
        self.index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            self.docs = pickle.load(f)
        self.model_name = model_name

    def retrieve(self, query, top_k=3):
        # embed query
        q_emb = self.client.feature_extraction(query, model=self.model_name)
        q_emb = np.array(q_emb)
        # 如果是一维向量 reshape 为 (1, dim)
        if len(q_emb.shape) == 1:
            q_emb = q_emb.reshape(1, -1)

        # 检索 top_k
        D, I = self.index.search(q_emb, top_k)
        results = [self.docs[i] for i in I[0]]
        return "\n".join(results)  # 拼成文本
    


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
    

class Agent:
    def __init__(self, medical_report=None, role=None, extra_info=None, retriever=None, extra_rag_context=None):
    # 在main.py中retrieve一次得到的extra_rag_context直接导入，就可以做到只检索一次，效率更高。所有 Specialist Agents 共享同一份 RAG 内容。
    # 如果要针对不同agent分别检索，就可以再调用 retriever
        self.medical_report = medical_report
        self.role = role
        self.extra_info = extra_info
        self.retriever = retriever 
        self.extra_rag_context = extra_rag_context  # 这里存一次检索结果
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

        rag_context = self.extra_rag_context if self.role in ["Cardiologist", "Psychologist"] else ""

        # 打印检索内容
        if rag_context:
            print(f"\n=== RAG Retrieved Docs for {self.role} ===")
            print(rag_context)
            print("="*50)

        # 格式化 prompt
        if self.role == "MultidisciplinaryTeam":
            prompt = self.prompt_template.format()
        else:
            prompt = self.prompt_template.format(medical_report=self.medical_report)
            if rag_context:
                prompt = f"### Reference from external medical library:\n{rag_context}\n\n{prompt}"

        try:
            response = self.model.invoke(prompt)
            return response
        except Exception as e:
            print("Error occurred:", e)
            return None



# ========== Specialized Agents ==========

class Cardiologist(Agent):
    def __init__(self, medical_report, retriever=None, extra_rag_context=None):
        super().__init__(medical_report=medical_report, role="Cardiologist", retriever=retriever)

class Psychologist(Agent):
    def __init__(self, medical_report, retriever=None, extra_rag_context=None):
        super().__init__(medical_report=medical_report, role="Psychologist", retriever=retriever)


class MultidisciplinaryTeam(Agent):
    def __init__(self, cardiologist_report, psychologist_report):
        super().__init__(
            role="MultidisciplinaryTeam",
            extra_info={
                "cardiologist_report": cardiologist_report,
                "psychologist_report": psychologist_report
            }
        )