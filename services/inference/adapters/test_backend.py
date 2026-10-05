"""Explicit deterministic backend for protocol testing, never model accuracy."""
from datetime import timedelta, timezone

from flight.v1 import prediction_pb2 as pb


class TestBackend:
    __test__ = False
    source = pb.TEST
    model_version = 'test-v1'

    def predict(self, flight, cancelled, time_budget):
        departure = flight.planned_off_block.ToDatetime(tzinfo=timezone.utc)
        arrival = flight.planned_on_block.ToDatetime(tzinfo=timezone.utc)
        off_block = departure + timedelta(minutes=10)
        takeoff = departure + timedelta(minutes=20)
        landing = max(arrival, takeoff)
        on_block = max(arrival + timedelta(minutes=10), landing)
        prediction = pb.Prediction()
        for name, value in [('off_block', off_block), ('takeoff', takeoff),
                            ('landing', landing), ('on_block', on_block)]:
            getattr(prediction, name).FromDatetime(value)
        return prediction
