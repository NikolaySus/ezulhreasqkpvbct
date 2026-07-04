from rq import Worker

from app.queue import get_queue, get_redis
from app.settings import RESULTS_DIR, UPLOADS_DIR


def main() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Worker([get_queue()], connection=get_redis()).work()


if __name__ == "__main__":
    main()
