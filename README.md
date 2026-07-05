# ezulhreasqkpvbct

MVP-заготовка: фронтенд работает на VPS, локальный FastAPI backend работает на ПК, ставит задачи сегментации в Redis/RQ и запускает PyTorch inference на локальной NVIDIA GPU.

## Backend на локальном ПК

```bash
docker compose -f docker-compose.backend.yml up -d --build
```

Проверка:

```bash
curl http://127.0.0.1:8001/api/health
```

Загрузка изображения на сегментацию:

```bash
curl -F "image=@/path/to/image.jpg" http://127.0.0.1:8001/api/segment
```

Ответ вернет `job_id`. Статус и результат:

```bash
curl http://127.0.0.1:8001/api/jobs/<job_id>
curl -o result.png http://127.0.0.1:8001/api/jobs/<job_id>/result
```

## Доразметка

Файлы будущей системы доразметки лежат в локальной папке `annotation-data/`; она игнорируется git.

По умолчанию backend запускается в заблокированном режиме:

```bash
curl http://127.0.0.1:8001/api/annotation/status
```

Разблокированный режим предназначен только для локальной работы при выключенном публичном frontend/tunnel. Для него нужно переопределить `ANNOTATION_EDITING_ENABLED=true` у backend API.

## Reverse SSH tunnel с ПК на VPS

Один раз добавить публичный ключ ПК на VPS:

```bash
ssh-copy-id root@109.73.203.55
```

Проверка вручную:

```bash
ssh -N -T \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -R 127.0.0.1:18080:127.0.0.1:8001 \
  root@109.73.203.55
```

Автозапуск туннеля через user systemd:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/ezulhreasqkpvbct-tunnel.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now ezulhreasqkpvbct-tunnel.service
loginctl enable-linger "$USER"
```

Проверка с VPS:

```bash
curl http://127.0.0.1:18080/api/health
```

## Frontend на VPS

В репозитории на сервере:

```bash
git pull
docker compose up -d
```

Публичные URL:

- фронтенд: http://109.73.203.55/
- API через туннель: http://109.73.203.55/api/health
