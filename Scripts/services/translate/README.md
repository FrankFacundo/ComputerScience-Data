# Develop

Build:

```bash
sudo docker build -t traductor .
```

Run in development mode:

```bash
sudo docker run -it --mount type=bind,source="$(pwd)",target=/app -p 1611:1611 traductor uvicorn app.main:app --host 0.0.0.0 --port 1611 --reload
```

Test:

```bash
curl -X POST "http://localhost:1611/api/frank/traductor/translate/" \
  -d '{"text": "Hello World"}' \
  -H "Content-Type: application/json"
```
