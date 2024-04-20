docker build -t ai-postgres .

docker run --name ai-postgres -p 5432:5432 ai-postgres

psql -h localhost -p 5432 -U postgres
