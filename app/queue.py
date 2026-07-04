from redis import Redis
from rq import Queue

from app.settings import QUEUE_NAME, REDIS_URL


def get_redis() -> Redis:
    return Redis.from_url(REDIS_URL)


def get_queue() -> Queue:
    return Queue(QUEUE_NAME, connection=get_redis())
