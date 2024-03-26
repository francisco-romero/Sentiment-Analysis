FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt Makefile .
COPY src/main.py src/imdb_bert.pth ./src
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "./src/main.py"]