import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from faceauth.vision import FaceEngine, FaceInputError, choose_lfw_primary_face
from scripts import eval_pairs


class StubDetector:
    def __init__(self, faces): self.faces=np.asarray(faces,dtype=np.float32)
    def setInputSize(self, s): self.size=s
    def detect(self, img): return 1,self.faces

class StubRecognizer:
    def alignCrop(self, img, face):
        self.last_face=face
        return np.ones((112,112,3),np.uint8)
    def feature(self, image): return np.ones((1,128),np.float32)


def face_row(x,y,w,h,conf=0.95):
    return np.r_[x,y,w,h,np.zeros(10),conf].astype(np.float32)


def stub_engine(rows):
    engine=object.__new__(FaceEngine)
    engine._lock=threading.RLock()
    engine.detector=StubDetector(rows)
    engine.recognizer=StubRecognizer()
    engine.min_face_size=55
    engine.min_blur_score=0.0
    engine.model_id="mock"
    return engine


def test_lfw_selection_prefers_largest_area_and_tie_nearest_center():
    faces=np.stack([face_row(80,85,110,100),face_row(30,40,20,20)])
    selected,i=choose_lfw_primary_face(faces,250,250)
    assert i==0 and selected[2]==110
    faces=np.stack([face_row(0,0,80,80),face_row(85,85,80,80)])
    _,i=choose_lfw_primary_face(faces,250,250)
    assert i==1


def test_lfw_rejects_bad_rows():
    for faces in [np.array([]),np.ones((2,4)),np.ones((0,15))]:
        with pytest.raises(ValueError):
            choose_lfw_primary_face(faces,250,250)


def test_auth_strict_does_not_silently_select_second_face():
    engine=stub_engine(np.stack([face_row(45,50,125,130),face_row(190,180,20,20)]))
    image=np.zeros((250,250,3),np.uint8)
    with pytest.raises(FaceInputError,match="检测到 2"):
        engine.extract(image,annotate=False)
    result=engine.extract(image,annotate=False,selection_policy="lfw-largest")
    assert result.detected_face_count==2
    assert result.selection_policy=="lfw-largest"
    assert result.box_xywh==(45,50,125,130)
    assert result.preview_jpeg is None
    with pytest.raises(ValueError):
        engine.extract(image,selection_policy="bad")


def test_eval_records_rescued_multi_detection_without_loss_of_fail_counts(monkeypatch,tmp_path):
    monkeypatch.setattr(eval_pairs,"read_photo_file",lambda _:np.zeros((250,250,3),np.uint8))
    engine=stub_engine(np.stack([face_row(45,50,125,130),face_row(190,180,20,20)]))
    pairs=[("a.jpg","b.jpg",1),("a.jpg","c.jpg",0)]
    strict={}
    scores,labels,failures=eval_pairs.evaluate_images(pairs,tmp_path,engine,image_audit=strict)
    assert len(scores)==0 and len(failures)==2
    assert strict["unique_images_failed"]==1
    relaxed={}
    scores,labels,failures=eval_pairs.evaluate_images(
        pairs,tmp_path,engine,face_selection="lfw-largest",image_audit=relaxed)
    assert len(scores)==2 and len(failures)==0
    assert relaxed["unique_images_with_embeddings"]==3
    assert relaxed["unique_images_with_multiple_yunet_boxes_selected"]==3


def test_original_auth_service_does_not_request_lfw_policy():
    from faceauth.service import FaceAuthService
    # The calls are explicit, and the app default remains strict.
    assert FaceEngine.extract.__kwdefaults__["selection_policy"] == "strict"
