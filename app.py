import json
from openai import OpenAI
from difflib import SequenceMatcher
import gradio as gr

# 初始化 DeepSeek 客户端
client = OpenAI(api_key="sk-874783538ff04df5bcf67aa3fec598a7", base_url="https://api.deepseek.com")

# 加载知识库数据
with open("huchenfeng_dialog_deepseek.json", "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

# 户晨风的核心观点
HUCHENFENG_ATTITUDES = {
    "高铁私有化": "支持高铁私有化",
    "文言文": "反对文言文",
    "中医": "反对中医",
    "大学开放": "支持大学开放"
}

# 知识库搜索函数
def search_knowledge_base(user_input, threshold=0.7):
    """通过相似度搜索知识库中的匹配回答"""
    best_match = None
    highest_similarity = 0
    for entry in knowledge_base:
        similarity = SequenceMatcher(None, user_input, entry["instruction"]).ratio()
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = entry
    # 如果相似度超过阈值，返回匹配的回答和对应条目
    if highest_similarity >= threshold:
        return f"（来自知识库）{best_match['response']}", best_match
    return None, None

# 动态生成函数
def generate_response_with_deepseek(user_input, predefined_attitude=None):
    """调用 DeepSeek 实时生成回答，并注入明确态度"""
    prompt = f"""
    你是户晨风，一位幽默且犀利的评论员。以下是你的核心观点：
    1. 支持高铁私有化
    2. 反对文言文
    3. 反对中医
    4. 支持大学开放

    用户提问：{user_input}
    {f"你的态度是：{predefined_attitude}" if predefined_attitude else ""}
    请生成一段完整的回答，简洁幽默并符合你的态度：
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are Hu Chenfeng, a humorous and insightful commentator."},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.7,
            max_tokens=300,  # 增加最大生成字数限制
            stop=["\n"]  # 设置回答结束标记，防止回答过长
        )
        return f"（来自实时生成）{response.choices[0].message.content.strip()}"
    except Exception as e:
        return f"生成回答失败，错误信息：{e}"

# 综合回答逻辑
def chat_with_huchenfeng(user_input):
    # Step 1: 在知识库中查找回答
    kb_response, matched_entry = search_knowledge_base(user_input)
    if kb_response:
        return kb_response

    # Step 2: 动态生成，但根据已知态度补充生成条件
    predefined_attitude = None
    for topic, attitude in HUCHENFENG_ATTITUDES.items():
        if topic in user_input:
            predefined_attitude = attitude
            break

    return generate_response_with_deepseek(user_input, predefined_attitude)

# Gradio 界面定义
def gradio_interface(user_input):
    """通过 Gradio 实现用户与 AI 对话的界面"""
    return chat_with_huchenfeng(user_input)

# 创建 Gradio 接口
iface = gr.Interface(
    fn=gradio_interface,
    inputs="text",
    outputs="text",
    title="AI 户晨风对话系统",
    description="与 AI 户晨风实时互动，获取幽默犀利的回答！"
)

# 启动 Gradio 应用
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    iface.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=True  # 生成公共链接（带 HTTPS），可用于 iframe
    )
