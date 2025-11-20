# 🏥 Medical Multi-Agent AI Assistant with Human-in-the-Loop and Optional RAG

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-inference-orange)](https://huggingface.co/docs/api-inference/index)

---

A Python-based medical diagnostic assistant built with multiple domain-specialized AI agents.
Each agent represents a medical professional (e.g., Cardiologist, Psychologist, Pulmonologist) and
performs independent analysis on the same medical report.

All agents run in parallel using a thread pool, and their results are consolidated into
a unified, multi-disciplinary medical insight report.

---

## 🚀 Project Overview

This project is a **medical AI assistant** designed to analyze patient medical reports using a **multi-agent architecture**, incorporating two **independent optional modules** that extend the base LLM functionality:
1. **Human-in-the-Loop (HITL)** – Doctors can review, edit, and approve each agent's preliminary assessment to improve diagnostic accuracy.
2. **RAG (Retrieval-Augmented Generation)** – Retrieves relevant information from an external medical document library to enrich agent reasoning.


- **Multi-Agent Setup**: Different domain-specific agents analyze medical reports:
  - **Cardiologist Agent** – cardiac assessment
  - **Psychologist Agent** – mental health assessment
- **MDT Agent**: Aggregates agent outputs (and optional HITL feedback) into a final diagnostic report.
- **LangGraph Workflow Alternative**: Implements the same process using LangChain’s **StateGraph**, allowing node-based execution and state tracking.


---
# 🩺 Current Architecture

At the moment, the system includes two AI agents, each representing a medical specialist:

1. Cardiologist Agent

Focus: Cardiac-related symptoms, arrhythmias, chest discomfort, circulatory problems

Outputs:

Potential cardiac conditions

Recommended diagnostic tests

Risk-level assessment

2. Psychologist Agent

Focus: Psychological factors contributing to symptoms, such as anxiety or panic disorders

Outputs:

Psychological differential diagnoses

Suggested therapy directions

Behavioral & stress-related risk factors

# ⚙️ How the System Works

You input a medical report into the system.
The two agents (Cardiologist & Psychologist) run in parallel using Python threading.
Each agent independently produces domain-specific medical insights, optionally augmented by **RAG retrieval**.
Their outputs are then collected by the MultidisciplinaryTeam Agent, which:

- Integrates all findings
- Synthesizes the reasoning
- Provides a final structured medical report
- Highlights the most probable conditions and recommended next steps


# 🌟 Features

1. **Domain-Specific Agent Analysis** – Each agent generates its independent report from the medical data.
2. **Human-in-the-Loop (HITL)** – Doctors can review each agent’s output, modify it, and improve accuracy.
3. **Retrieval-Augmented Generation (RAG)** – Optionally injects relevant medical knowledge from an external library to enhance reasoning.
4. **MDT Aggregation** – Combines multiple agent outputs and HITL feedback into a final report.
5. **Flexible Workflow Options**
   - Traditional sequential/concurrent multi-agent execution
   - LangGraph StateGraph workflow with nodes, states, and transitions
6. **HuggingFace Inference Integration**
   - Supports `openai/gpt-oss-120b` and other HF models
   - Easily extendable to more agents and medical domains

---


# 🛠 Installation

## Clone repository

```bash
git clone https://github.com/yourusername/medical-multi-agent.git
cd medical-multi-agent
```

# Create virtual environment

python -m venv venv

source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt



# 🔑 Configuration

## Generate HuggingFace API token

Go to HuggingFace Tokens

Select Read permission

No expiration recommended (or set 90 days if desired)

Add token to .env file (ensure it is gitignored)

HF_TOKEN=hf_your_generated_token


## Load environment variables in Python

from dotenv import load_dotenv

load_dotenv("hf.env", override=True)


# 📝 Usage

## 1. Multi-Agent Workflow with HITL

```python
python humanfeedback_main.py
```

Behavior:

* Reads a medical report.

* Each agent generates its preliminary assessment.

* Doctor can review, edit, and approve each agent’s output.

* MultidisciplinaryTeam agent aggregates reviewed outputs into a final MDT report.

Output:

* Final integrated MDT diagnostic report (saved in humanfeedback_results/final_diagnosis.txt).

## 2. RAG-Enhanced Multi-Agent Workflow

```python
python RAG_version/rag_main.py
```
Behavior:

* Reads a medical report.

* Each agent retrieves relevant content from an external medical document library via RAG.

* Retrieved content is injected into the agent’s prompt to enrich reasoning.

* MultidisciplinaryTeam agent aggregates all agent outputs into a final report.

Output:

* Final RAG-enhanced MDT diagnostic report (saved in RAG_version/results/final_diagnosis.txt).

## 3. LangGraph StateGraph Workflow (Alternative)

```python
python langgraph_version/main_langgraph.py
```

# Project Structure

```python
medical-multi-agent/
├─ RAG_version/
│  ├─ agent.py                 # 多代理核心类定义，支持RAG注入
│  ├─ rag_main.py                  # RAG版本主脚本
│  ├─ vdb.py                   # RAG/FAISS向量数据库构建与检索
│  ├─ medical_docs.pkl         # 医学文档序列化文件
│  ├─ medical_docs.index       # FAISS向量索引
│  ├─ medical_docs.py          # 医学文档处理脚本
│  └─ medical_report_chinese.txt  # 示例中文医疗报告
├─ Utils/
│  ├─ agent_humanfeedback.py   # HITL辅助功能封装
│  ├─ myagent.py               # 自定义agent封装
├─ langgraph_version/
│  ├─ agent_langgraph.py       # LangGraph状态图实现
│  └─ main_langgraph.py        # LangGraph版本主脚本
├─ Medical Reports/
│  ├─ medical_report_chinese.txt
│  └─ medical_report_english.txt
├─ humanfeedback_results/
│  └─ final_diagnosis.txt
├─ results/
│  └─ final_diagnosis.txt
├─ myagent_main.py             # 自定义入口脚本
├─ humanfeedback_main.py       # HITL主入口脚本
├─ hf.env                      # HuggingFace API token（gitignored）
├─ requirements.txt
├─ README.md

```

# ⚡ Notes

Do not commit your .env / hf.env file to GitHub.

You can extend the system with more specialized agents (e.g., Pulmonologist, Neurologist).

HITL is optional but highly recommended for high-stakes medical use cases.

LangGraph workflow is an alternative; choose the workflow that best suits your project.

# References

- [LangChain Documentation](https://python.langchain.com/)
- [HuggingFace Inference API](https://huggingface.co/docs/api-inference)
- Multi-Agent AI and Human-in-the-Loop best practices