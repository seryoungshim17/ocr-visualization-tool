FROM python:3.8.12-slim-buster

COPY . /app
WORKDIR /app/front
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "app.py"]