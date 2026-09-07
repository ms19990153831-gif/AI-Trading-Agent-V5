FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV SIMULATE=false
ENV USE_MOCK_LLM=false

CMD ["python", "main.py", "--mode", "v5", "--cycles", "288", "--interval", "300"]
