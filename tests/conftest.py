from dataclasses import replace
import numpy as np
import pytest
from faceauth.vision import FaceObservation


class DummyEngine:
    model_id = "unit-test-model-sha"

    def __init__(self, raise_error=None):
        self.raise_error = raise_error

    def extract(self, image, *, annotate=True):
        if self.raise_error:
            raise self.raise_error
        data = np.asarray(image)
        klass = int(data[0,0,0]) % 3
        emb = np.zeros(32, dtype=np.float32)
        emb[klass] = 1
        return FaceObservation(emb, 0.95, (10,10,80,80), 80,80, 120, 1,2)


@pytest.fixture
def fake_image():
    def make(index=0):
        im = np.zeros((128,128,3), np.uint8)
        im[0,0,0] = index
        return im
    return make


@pytest.fixture
def service(tmp_path):
    from faceauth.store import FaceStore
    from faceauth.service import FaceAuthService
    engine = DummyEngine()
    store = FaceStore(tmp_path/"test.sqlite3", model_id=engine.model_id)
    return FaceAuthService(engine, store)
