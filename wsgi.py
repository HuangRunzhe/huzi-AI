# wsgi.py
from app import app  # 这里假设你的 Flask 实例名是 app，且在 app.py 中定义

if __name__ == "__main__":
    app.run(debug=True)
