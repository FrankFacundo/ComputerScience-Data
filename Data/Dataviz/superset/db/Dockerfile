FROM postgres:16.1-bullseye

ENV POSTGRES_USER postgres
ENV POSTGRES_PASSWORD postgres
ENV POSTGRES_DB ai_db

COPY docker-entrypoint-initdb.d /docker-entrypoint-initdb.d

EXPOSE 5432
