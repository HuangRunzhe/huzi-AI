from flask import Flask, request, jsonify,session
from flask_cors import CORS
from openai import OpenAI
from difflib import SequenceMatcher
import json
import logging
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from flask import render_template,send_file
import edge_tts
import asyncio
from gevent import monkey

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)  # 启用 CORS

# 初始化 DeepSeek 客户端
client = OpenAI(api_key="sk-fanUWx2HJOPgTj0Oa0DNqIHsV2aw2UypGD8gL1s794ph2orf", base_url="https://tbnx.plus7.plus/v1")

# 加载知识库数据
with open("Data/huchenfeng_dialog_deepseek.json", "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

# 户晨风的核心观点
HUCHENFENG_ATTITUDES = {
    "高铁私有化": "支持高铁私有化",
    "文言文": "反对文言文",
    "中医": "反对中医",
    "大学开放": "支持大学开放",
    "邮政私有化": "支持邮政私有化"
}

# 聊天记录存储目录
CHAT_HISTORY_DIR = "chat_history"
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)  # 创建目录（如果不存在）

# 反馈存储文件
FEEDBACK_FILE = "Data/feedback.json"
# 敏感词文件路径
SENSITIVE_WORDS_FILE = "/www/wwwroot/secret_file/sensitive_words_lines.txt"

# 读取敏感词列表
def load_sensitive_words():
    """加载敏感词列表"""
    try:
        with open(SENSITIVE_WORDS_FILE, "r", encoding="utf-8") as f:
            words = [line.strip().rstrip(',') for line in f if line.strip()]
        return set(words)
    except FileNotFoundError:
        logging.warning("敏感词文件不存在，将使用空列表。")
        return set()

SENSITIVE_WORDS = load_sensitive_words()


# 检测用户输入是否包含敏感词
def contains_sensitive_words(text):
    """检查输入是否包含敏感词"""
    for word in SENSITIVE_WORDS:
        if word in text:
            return True
    return False


# 知识库搜索函数
def search_knowledge_base(user_input, threshold=0.4):
    """通过 TF-IDF 算法搜索知识库中的匹配回答"""
    best_match = None
    highest_similarity = 0

    # 预处理文本：将用户输入和知识库条目放在一起
    corpus = [entry["instruction"] for entry in knowledge_base]
    corpus.append(user_input)  # 将用户输入也添加到语料库

    # 初始化 TF-IDF 向量化器
    vectorizer = TfidfVectorizer()

    # 计算 TF-IDF 矩阵
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # 获取用户输入与知识库每个条目的相似度（使用余弦相似度）
    similarities = np.dot(tfidf_matrix[-1], tfidf_matrix.T).toarray()[0][:-1]  # 用户输入与每个条目的相似度
    highest_similarity = max(similarities)
    best_match_idx = np.argmax(similarities)
    best_match = knowledge_base[best_match_idx] if highest_similarity >= threshold else None

    if best_match:
        return f"（来自知识库）{best_match['response']}", best_match
    return None, None


def generate_response_with_deepseek(user_input, predefined_attitude=None):
    """调用 DeepSeek 生成回答，并注入明确态度"""
    prompt = f"""
    你是户晨风，一位幽默且犀利的评论员。以下是你的核心观点，但只在用户问到时才提及：
    1. 支持高铁私有化，认为公营高铁效率低、服务差，私有化可以提高效率和竞争力。
    2. 反对文言文，认为文言文已经不再适应现代社会的沟通需求，现代教育应当关注实用性。
    3. 反对中医，认为中医体系缺乏科学依据，且常常被过度神话。
    4. 支持大学开放，认为大学应当追求学术自由，给学生更多自主选择的权利。
    
    一些常见的替代词，你要能读懂：
    1.”碧养地“就是代指”比亚迪“
    2.”菊花“代指”华为“
    3.”某米“代指”小米“
    3.“痢疾”代指”吉利汽车“
    请注意：当用户没有问到这些观点时，不要主动提及，仅回答他们的问题！不要回答用户没有问到的内容，不要偏离主题，更不能强行加上你自己的观点。
    你是用的手机：1TB版本的iPhone 16 Pro
    用户提问：{user_input}
    {f"你的态度是：{predefined_attitude}" if predefined_attitude else ""}

    请生成一段符合你风格的回答，简洁且幽默，并且加入批判性的视角。如果有必要，你可以适当使用一些讽刺、反问或夸张的语气来增强效果，但一定要直接有力：
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
            max_tokens=300,
            stop=["\n"]
        )
        
        return f"（来自实时生成）{response.choices[0].message.content.strip()}"

    except Exception as e:
        return f"生成回答失败，错误信息：{e}"



# 生成 AI 回答
def chat_with_huchenfeng(user_input):
    """综合逻辑，先搜索知识库，再调用 AI 生成"""
    kb_response, matched_entry = search_knowledge_base(user_input)
    if kb_response:
        return kb_response

    predefined_attitude = None
    for topic, attitude in HUCHENFENG_ATTITUDES.items():
        if topic in user_input:
            predefined_attitude = attitude
            break

    return generate_response_with_deepseek(user_input, predefined_attitude)


# 存储聊天记录（按 sessionID）
def save_chat_history(session_id, user_message, bot_response):
    """存储用户的聊天记录到 session 专属文件"""
    try:
        chat_file = os.path.join(CHAT_HISTORY_DIR, f"chat_history_{session_id}.json")
        chat_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_message": user_message,
            "bot_response": bot_response
        }

        try:
            with open(chat_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = []

        history.append(chat_record)

        with open(chat_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.error(f"Failed to save chat history: {e}")


# 读取聊天记录 API
@app.route("/history", methods=["GET"])
def get_chat_history():
    """根据 sessionID 返回聊天记录"""
    session_id = request.args.get("sessionID", "").strip()
    if not session_id:
        return jsonify({"error": "缺少 sessionID"}), 400

    chat_file = os.path.join(CHAT_HISTORY_DIR, f"chat_history_{session_id}.json")

    try:
        with open(chat_file, "r", encoding="utf-8") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    return jsonify({"history": history})


# 处理聊天请求 API
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        session_id = data.get("sessionID", "").strip()
        user_input = data.get("message", "").strip()

        if not session_id or not user_input:
            return jsonify({"response": "请输入有效的问题"}), 400
        
                # 敏感词检测
        if contains_sensitive_words(user_input):
            logging.warning(f"Session {session_id} - 触发敏感词: {user_input}")
            
            save_chat_history(session_id, user_input, "无法回答")
            return jsonify({"response": "抱歉，该内容无法回答。"}), 403  # HTTP 403 Forbidden

        logging.info(f"Session {session_id} - 用户提问: {user_input}")

        response = chat_with_huchenfeng(user_input)
        logging.info(f"Session {session_id} - 生成回答: {response}")

        save_chat_history(session_id, user_input, response)

        return jsonify({"response": response})

    except Exception as e:
        logging.error(f"Error occurred: {e}", exc_info=True)
        return jsonify({"response": "服务器出现问题，请稍后再试。"}), 500

# 存储反馈
def save_feedback(feedback_text, contact_info):
    """保存用户反馈"""
    try:
        feedback_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "feedback_text": feedback_text,
            "contact_info": contact_info
        }

        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                feedback_list = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            feedback_list = []

        feedback_list.append(feedback_data)

        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(feedback_list, f, ensure_ascii=False, indent=4)

    except Exception as e:
        logging.error(f"Failed to save feedback: {e}")

# 提交反馈 API
@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        data = request.json
        feedback_text = data.get("feedbackText", "").strip()
        contact_info = data.get("contactInfo", "").strip()

        if not feedback_text:
            return jsonify({"message": "反馈内容不能为空"}), 400

        save_feedback(feedback_text, contact_info)
        return jsonify({"message": "反馈提交成功"}), 200

    except Exception as e:
        logging.error(f"Error saving feedback: {e}")
        return jsonify({"message": "服务器错误，请稍后再试"}), 500



@app.route("/get_feedback", methods=["GET"])
def get_feedback():
    """返回所有用户反馈"""
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            feedback_list = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        feedback_list = []

    return jsonify({"feedback": feedback_list})




# 读取 balances.json 文件
def load_balances():
    try:
        with open('Data/balances.json', 'r', encoding='utf-8') as f:
            # 如果文件为空，直接返回一个空字典
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        # 如果文件存在但是无法解析，清空文件并返回空字典
        with open('Data/balances.json', 'w', encoding='utf-8') as f:
            f.write("{}")  # 写入空字典
        return {}

# 更新 balances.json 文件
def save_balances(balances):
    with open('Data/balances.json', 'w', encoding='utf-8') as f:
        json.dump(balances, f, ensure_ascii=False, indent=4)

# 为每个session生成一个简短数字ID
def generate_user_id(balances):
    return len(balances) + 1  # 用现有的用户数生成一个新的ID

@app.route('/check_balance', methods=['GET'])
def check_balance():
    session_id = request.args.get('sessionID')
    if not session_id:
        return jsonify({"error": "sessionID is required"}), 400

    # 读取余额数据
    balances = load_balances()

    # 如果该用户没有余额记录，初始化余额为 0，并生成一个简短的数字ID
    if session_id not in balances:
        user_id = generate_user_id(balances)
        balances[session_id] = {"balance": 0, "userID": user_id}
        save_balances(balances)  # 保存更新后的余额数据

    # 获取用户余额和用户ID
    balance = balances.get(session_id, {}).get("balance", 0)
    user_id = balances.get(session_id, {}).get("userID")

    print(f"Returning balance and userID: {balance}, {user_id}")  # 打印调试信息

    # 向前端返回余额和一个消息，告知用户ID
    return jsonify({
        "balance": balance,
        "userID": user_id,
        "message": f"您的用户ID是：{user_id}",
        "sessionID": session_id  # 显示sessionID，方便前端关联
    })

async def generate_speech(text, output_path="output.mp3"):
    """使用 Edge TTS 生成语音"""
    tts = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")  # 语音选择
    await tts.save(output_path)  # 使用 await 来异步调用 save
    return output_path

@app.route("/tts")
async def tts():
    """TTS API，前端访问 /tts?text=任意文字 获取语音"""
    text = request.args.get("text", "你好")  # 默认值 "你好"
    output_path = "static/output001.mp3"
    
    # 异步生成语音
    await generate_speech(text, output_path)
    
    # 确保文件生成
    try:
        return send_file(output_path, mimetype="audio/mpeg")
    except Exception as e:
        return f"Error: {e}"


@app.route('/')
def home():
    return render_template('index.html')  # 返回一个 HTML 页面

@app.route("/chat_page")
def chat_page():
    return render_template("chat.html")

@app.route("/donate")
def donate():
    return render_template("donate.html")

@app.route("/feedback_page")
def feedback_page():
    return render_template("feedback.html")

@app.route("/view_feedback_page")
def view_feedback_page():
    return render_template("view_feedback.html")

@app.route("/intro")
def intro_page():
    return render_template("intro.html")

# 启动 Flask 应用
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8090, debug=False)