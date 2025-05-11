# tts_app.py
from flask import Flask, request, send_file
import edge_tts

app = Flask(__name__)

def generate_speech(text, output_path="output.mp3"):
    """使用 Edge TTS 生成语音"""
    tts = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")  # 语音选择
    tts.save(output_path)  # 直接同步保存
    return output_path

@app.route("/tts")
def tts():
    """TTS API，前端访问 /tts?text=任意文字 获取语音"""
    text = request.args.get("text", "你好")  # 默认值 "你好"
    output_path = "static/output.mp3"
    
    # 同步生成语音
    generate_speech(text, output_path)
    
    return send_file(output_path, mimetype="audio/mpeg")

if __name__ == "__main__":
    app.run(debug=True)
