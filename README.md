# ezulhreasqkpvbct

Минимальный Hello World сайт на FastAPI, запускаемый через Docker Compose.

## Локальный запуск

```bash
docker compose up --build
```

После запуска:

- сайт: http://localhost
- healthcheck: http://localhost/health

## Деплой на сервере

В репозитории на сервере:

```bash
git pull
docker compose up -d --build
```
