# ezulhreasqkpvbct

## Материалы решения

### Ссылка на развернутое решение

- ~~https://ezulhreasqkpvbct.pagekite.me~~ — нерабочая ссылка.
- http://109.73.203.55/ — рабочая ссылка.

### Описание решения

Решение для задачи «Скажи мне кто твой шлиф»: система автоматизации работы с изображениями полированных шлифов руды.

Основные компоненты:

- автоматизированная доразметка изображений;
- обучение модели сегментации в несколько этапов;
- инференс с двумя головами: сегментационная маска фаз и классификация руды на рядовую/труднообогатимую;
- heatmap вероятности труднообогатимости по тайлам;
- итоговый вердикт: оталькованная, рядовая или труднообогатимая руда.

### Ссылка на VCS

https://github.com/NikolaySus/ezulhreasqkpvbct

### Ссылка на презентацию

https://disk.yandex.ru/d/KcjCIQGEDk3m9A

### Ссылка на скринкаст

https://disk.yandex.ru/d/jKwsqUQrtWii9Q

## Техническое описание

MVP-заготовка: публичный nginx на VPS проксирует сайт на локальный FastAPI backend через reverse SSH tunnel. Backend отдает Jinja frontend, ставит задачи инференса в Redis/RQ и запускает PyTorch inference на локальной NVIDIA GPU.

Инференс использует сегментационную и классификационную головы из локального проекта `ml-days-2`. Checkpoint модели не хранится в git; для локального запуска он должен лежать здесь:

```text
model-artifacts/ml-days-2/classification/recommended.pth
```

Результат инференса:

- цветовая маска сегментации фаз;
- heatmap `P(труднообогатимая)` по тайлам, сглаженная в единую карту;
- доля талька как отношение площади класса `talc` к площади изображения;
- средняя вероятность труднообогатимости по всем тайлам;
- итоговый вердикт:
  - `оталькованная`, если доля талька больше 10%;
  - иначе `труднообогатимая`, если средняя вероятность труднообогатимости не меньше 0.5;
  - иначе `рядовая`.

## Backend на локальном ПК

```bash
docker compose -f docker-compose.backend.yml up -d --build
```

Проверка:

```bash
curl http://127.0.0.1:8001/api/health
```

## Локальное дообучение классификационной головы

Сайт использует checkpoint `recommended.pth`, полученный этим pipeline. Обучение запускается вручную в CUDA worker-контейнере, результаты пишутся в ignored `training-artifacts/`.

```bash
docker compose -f docker-compose.backend.yml build worker
docker compose -f docker-compose.backend.yml run --rm --user "$(id -u):$(id -g)" worker \
  python -m training.build_classification_dataset --overwrite
docker compose -f docker-compose.backend.yml run --rm --user "$(id -u):$(id -g)" worker \
  python -m training.train_classification_head
```

Быстрый smoke-тест:

```bash
docker compose -f docker-compose.backend.yml run --rm --user "$(id -u):$(id -g)" worker \
  python -m training.train_classification_head --epochs 1 --max-images-per-class 8 --num-workers 0 --run-name smoke
```

Загрузка изображения на инференс:

```bash
curl -F "image=@/path/to/image.jpg" http://127.0.0.1:8001/api/segment
```

Ответ вернет `job_id`. Статус, метрики и результаты:

```bash
curl http://127.0.0.1:8001/api/jobs/<job_id>
curl -o segmentation.png http://127.0.0.1:8001/api/jobs/<job_id>/segmentation
curl -o difficulty-heatmap.png http://127.0.0.1:8001/api/jobs/<job_id>/difficulty-heatmap
```

`/api/jobs/<job_id>/result` сохранен как совместимый alias на сегментационную маску.

## Доразметка

Система доразметки Nornik встроена во вкладку `Доразметка`. Ее рабочие файлы лежат в локальной папке `annotation-data/nornik/`; вся `annotation-data/` игнорируется git.

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
