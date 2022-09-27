FROM python:3

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app
ENV PYTHONPATH "${PYTHONPATH}:/app/./app"

RUN cd /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "1611"]

EXPOSE 1611