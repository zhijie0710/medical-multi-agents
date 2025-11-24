import re

# =========================
# Cardiology Tool
# =========================
def analyze_lab_values(medical_report: str) -> str:
    """
    简单分析心脏相关的实验室指标，比如血压、心率、血脂等。
    输出建议文本，便于注入 LLM prompt。
    """
    output = []

    # 检测血压
    bp_matches = re.findall(r"(血压|BP)[:：\s]*(\d+/\d+)", medical_report)
    if bp_matches:
        for match in bp_matches:
            output.append(f"Detected blood pressure: {match[1]}")
            systolic, diastolic = map(int, match[1].split("/"))
            if systolic > 140 or diastolic > 90:
                output.append("⚠ High blood pressure detected.")
            elif systolic < 90 or diastolic < 60:
                output.append("⚠ Low blood pressure detected.")
    
    # 检测心率
    hr_matches = re.findall(r"(心率|HR)[:：\s]*(\d+)", medical_report)
    if hr_matches:
        for match in hr_matches:
            hr = int(match[1])
            output.append(f"Detected heart rate: {hr} bpm")
            if hr > 100:
                output.append("⚠ Tachycardia (high heart rate).")
            elif hr < 60:
                output.append("⚠ Bradycardia (low heart rate).")
    
    # 检测血脂
    chol_matches = re.findall(r"(总胆固醇|TC)[:：\s]*(\d+\.?\d*)", medical_report)
    if chol_matches:
        for match in chol_matches:
            tc = float(match[1])
            output.append(f"Detected total cholesterol: {tc} mg/dL")
            if tc > 240:
                output.append("⚠ High cholesterol.")
    
    if not output:
        output.append("No key cardiology lab values detected.")

    return "\n".join(output)


# =========================
# Psychology Tool
# =========================
def assess_psych_risk(medical_report: str) -> str:
    """
    简单分析心理健康风险，检测文本中的关键症状词。
    返回辅助分析文本。
    """
    risk_factors = {
        "anxiety": ["焦虑", "紧张", "担心", "恐惧"],
        "depression": ["抑郁", "低落", "悲伤", "绝望"],
        "stress": ["压力", "紧张", "烦躁"]
    }

    output = []

    for factor, keywords in risk_factors.items():
        matches = [kw for kw in keywords if kw in medical_report]
        if matches:
            output.append(f"Possible {factor} indicators detected: {', '.join(matches)}")

    if not output:
        output.append("No clear psychological risk indicators detected.")

    return "\n".join(output)