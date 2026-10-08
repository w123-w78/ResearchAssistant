# ============================================================
# 智能科研助理 —— 应用镜像
# 构建：docker build -f docker/Dockerfile -t research-assistant .
# 运行：docker run -p 8501:8501 -e DASHSCOPE_API_KEY=sk-xxx research-assistant
# ============================================================
FROM python:3.10-slim

WORKDIR /app

# 系统依赖（pymupdf 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 数据与日志目录（建议用 volume 挂载持久化）
RUN mkdir -p /app/data/uploads /app/data/index /app/logs

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

HEALTHCHECK --interval=60s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/frontend/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
