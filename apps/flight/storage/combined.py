"""Keep synchronous writes while exposing both persistent task namespaces."""
from datetime import datetime

from .documents import validate_limit


class CombinedJobStore:
    def __init__(self, primary, durable):
        self.primary, self.durable = primary, durable

    def create(self, total, input_timezone):
        return self.primary.create(total, input_timezone)

    def finish(self, job_id, results):
        return self.primary.finish(job_id, results)

    def import_job(self, document):
        return self.primary.import_job(document)

    def get(self, job_id):
        try:
            return self.primary.get(job_id)
        except KeyError:
            return self.durable.get(job_id)

    def list_jobs(self, limit=50):
        validate_limit(limit)
        jobs = self.primary.list_jobs(limit) + self.durable.list_jobs(limit)
        return sorted(jobs, key=lambda j: (datetime.fromisoformat(j['created_at']).timestamp(), j['job_id']), reverse=True)[:limit]
