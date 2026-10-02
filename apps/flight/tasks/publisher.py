"""At-least-once outbox publication with durable bounded backoff."""
from .queue import QueueError


class OutboxPublisher:
    def __init__(self, repository, queue):
        self.repository, self.queue = repository, queue

    def run_once(self, limit=100):
        published = 0
        for message in self.repository.pending_outbox(limit):
            try:
                self.queue.publish(message)
            except QueueError:
                self.repository.publication_failed(message['message_id'])
                break  # Stop a full/offline queue; remaining rows stay due.
            self.repository.mark_published(message['message_id'])
            published += 1
        return published
