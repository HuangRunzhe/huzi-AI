from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from difflib import SequenceMatcher
import json
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)  # 启用 CORS

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

# 聊天记录存储文件
CHAT_HISTORY_FILE = "chat_history.json"

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

# 存储聊天记录到文件
def save_chat_history(user_message, bot_response):
    """将用户消息和 AI 回复存储到聊天记录文件"""
    try:
        chat_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_message": user_message,
            "bot_response": bot_response
        }

        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = []

        history.append(chat_record)

        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.error(f"Failed to save chat history: {e}")

# 读取聊天记录的 API
@app.route("/history", methods=["GET"])
def get_chat_history():
    """读取聊天记录"""
    try:
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    return jsonify({"history": history})

# 定义聊天 API
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        logging.info(f"Received user input: {data}")
        user_input = data.get("message", "").strip()
        if not user_input:
            return jsonify({"response": "请输入有效的问题"}), 400

        response = chat_with_huchenfeng(user_input)
        logging.info(f"Generated response: {response}")

        # 保存聊天记录
        save_chat_history(user_input, response)

        return jsonify({"response": response})
    except Exception as e:
        logging.error(f"Error occurred: {e}", exc_info=True)
        return jsonify({"response": "服务器出现问题，请稍后再试。"}), 500

# 启动 Flask 应用
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
