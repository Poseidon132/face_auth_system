from pathlib import Path
import sqlite3
import numpy as np
import pytest
from faceauth.store import FaceStore, validate_person_name


def emb(which, n=32):
    x = np.zeros(n, np.float32)
    x[which] = 1.0
    return x


def test_consent_required(tmp_path):
    db = FaceStore(tmp_path/"db", model_id="v1")
    with pytest.raises(PermissionError):
        db.enroll("Alice", emb(0), detection_score=0.95, consent=False)
    assert db.persons() == []


def test_register_and_find_casefold(service, fake_image):
    service.register("  Alice   Smith ", fake_image(0), consent=True)
    assert service.store.person_by_name("alice smith")["name"] == "Alice Smith"
    service.register("alice smith", fake_image(0), consent=True)
    assert len(service.store.persons()) == 1
    assert service.store.sample_count() == 2


def test_verify_and_reject(service, fake_image):
    service.register("A", fake_image(0), consent=True)
    assert service.verify("A", fake_image(0), threshold=.45).accepted
    assert not service.verify("A", fake_image(1), threshold=.45).accepted
    unknown = service.verify("NOT_ENROLLED", fake_image(0), threshold=.45)
    assert not unknown.accepted and unknown.similarity is None
    assert len(service.store.logs()) == 3


def test_identify_and_unknown(service, fake_image):
    service.register("A", fake_image(0), consent=True)
    service.register("B", fake_image(1), consent=True)
    res = service.identify(fake_image(1), threshold=.45)
    assert res.accepted and res.name == "B"
    assert res.matches[0][0] == "B"
    unknown = service.identify(fake_image(2), threshold=.45)
    assert not unknown.accepted and unknown.name is None
    assert unknown.matches == ()  # do not reveal below-threshold names
    assert unknown.similarity == pytest.approx(0.0)


def test_delete_cascades_and_erases_events(service, fake_image):
    service.register("Private A", fake_image(0), consent=True)
    service.register("B", fake_image(1), consent=True)
    service.verify("Private A", fake_image(0))
    service.verify("B", fake_image(1))
    pid = service.store.person_by_name("Private A")["person_id"]
    assert service.store.delete_person(pid)
    assert not service.store.delete_person(pid)
    assert service.store.sample_count() == 1
    assert all(e["subject"] != "Private A" for e in service.store.logs())


def test_model_mismatch_rejected(tmp_path):
    path = tmp_path/"db.sqlite3"
    FaceStore(path, model_id="v1")
    with pytest.raises(RuntimeError, match="另一个"):
        FaceStore(path, model_id="v2")


def test_embeddings_corruption_detected(tmp_path):
    db = FaceStore(tmp_path/"db.sqlite3", model_id="x")
    db.enroll("A", emb(0), detection_score=1.0, consent=True)
    with sqlite3.connect(str(db.path)) as conn:
        conn.execute("UPDATE templates SET embedding=?", (sqlite3.Binary(b"bad"),))
    with pytest.raises(RuntimeError, match="长度"):
        db.templates_for()


def test_more_than_10_samples_rejected(tmp_path):
    db = FaceStore(tmp_path/"db.sqlite3", model_id="x")
    for _ in range(10):
        db.enroll("A", emb(0), detection_score=1.0, consent=True)
    with pytest.raises(ValueError, match="最多"):
        db.enroll("A", emb(0), detection_score=1.0, consent=True)
    assert db.sample_count() == 10


def test_feature_size_mismatch(tmp_path):
    db = FaceStore(tmp_path/"db.sqlite3", model_id="x")
    db.enroll("A", emb(0, 32), detection_score=0.9, consent=True)
    with pytest.raises(ValueError, match="维度"):
        db.enroll("B", emb(0, 64), detection_score=0.9, consent=True)
    assert db.person_by_name("B") is None


def test_names(tmp_path):
    for name in ("", "  ", "x"*51, "A\nB"):
        with pytest.raises(ValueError):
            validate_person_name(name)
    assert validate_person_name("  John   Doe  ") == "John Doe"


def test_events_csv_and_clear(service, fake_image):
    service.register("A", fake_image(0), consent=True)
    service.verify("A", fake_image(0))
    assert b"created_utc" in service.store.export_logs_csv()
    assert service.store.clear_logs() == 1
    assert service.store.logs() == []


def test_unsafe_threshold(service, fake_image):
    service.register("A", fake_image(0), consent=True)
    for value in (0, -1, 1.5, float('nan')):
        with pytest.raises(ValueError):
            service.verify("A", fake_image(0), threshold=value)
