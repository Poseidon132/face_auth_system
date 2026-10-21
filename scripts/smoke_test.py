"""Model-independent full workflow smoke test. Not a face-model accuracy test."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from faceauth.store import FaceStore
from faceauth.service import FaceAuthService
from faceauth.vision import FaceObservation


class FakeEngine:
    model_id = "fake-engine-only-for-code-tests"

    def extract(self, image: np.ndarray, *, annotate: bool = True) -> FaceObservation:
        channel = int(np.argmax(np.asarray(image)[0, 0]))
        vec = np.zeros(32, np.float32)
        vec[channel] = 1
        return FaceObservation(vec, 0.98, (0, 0, 90, 90), 90, 90, 80.0, 1.0, 2.0)


def photo(class_index):
    im = np.zeros((128, 128, 3), np.uint8)
    im[:, :, class_index] = 255
    return im


def main():
    with tempfile.TemporaryDirectory() as tmp:
        store = FaceStore(Path(tmp) / "test.db", model_id=FakeEngine.model_id)
        service = FaceAuthService(FakeEngine(), store)
        service.register("Person A", photo(0), consent=True)
        service.register("Person A", photo(0), consent=True)
        service.register("Person B", photo(1), consent=True)
        assert service.verify("Person A", photo(0), threshold=0.45).accepted
        assert not service.verify("Person A", photo(1), threshold=0.45).accepted
        assert service.identify(photo(1), threshold=0.45).name == "Person B"
        assert not service.identify(photo(2), threshold=0.45).accepted
        assert store.sample_count() == 3
        assert len(store.logs()) == 4
        assert len(store.export_logs_csv()) > 30
        aid = store.person_by_name("Person A")["person_id"]
        assert store.delete_person(aid)
        assert store.sample_count() == 1
    print("PASS: consent -> enrollment -> verification -> unknown rejection -> 1:N search -> logs -> deletion")
    print("NOTE: This uses deterministic synthetic embeddings; it does NOT validate YuNet/SFace model inference.")


if __name__ == "__main__":
    main()
