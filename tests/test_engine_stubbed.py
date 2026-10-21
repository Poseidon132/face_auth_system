import threading
import numpy as np
import pytest
from faceauth.vision import FaceEngine, FaceInputError


class Detector:
    def __init__(self, faces):
        self.faces=faces
        self.size=None

    def setInputSize(self,size):
        self.size=size

    def detect(self,image):
        return 1,self.faces


class Recognizer:
    def alignCrop(self,image,face):
        return np.ones((112,112,3), np.uint8)

    def feature(self,crop):
        return np.ones((1,128), dtype=np.float32)


def face_row(x=30,y=30,w=110,h=110,conf=.95):
    return np.r_[x,y,w,h,np.zeros(10),conf].astype(np.float32)


def stub(faces, min_size=55, blur=0):
    engine=object.__new__(FaceEngine)
    engine._lock=threading.RLock()
    engine.detector=Detector(faces)
    engine.recognizer=Recognizer()
    engine.min_face_size=min_size
    engine.min_blur_score=blur
    engine.model_id='dummy'
    return engine


def test_engine_happy_path_without_weights():
    engine=stub(np.array([face_row()]))
    img=np.zeros((256,320,3),dtype=np.uint8)
    result=engine.extract(img)
    assert len(result.embedding)==128
    assert result.detection_score==pytest.approx(.95)
    assert result.preview_jpeg.startswith(b'\xff\xd8')
    assert engine.detector.size==(320,256)
    assert np.linalg.norm(result.embedding)==pytest.approx(1)


def test_no_face_rejected():
    with pytest.raises(FaceInputError,match='未检测到'):
        stub(None).extract(np.zeros((256,256,3),np.uint8))


def test_multi_face_rejected():
    with pytest.raises(FaceInputError,match='检测到 2'):
        stub(np.stack([face_row(),face_row(150,150)])).extract(np.zeros((300,300,3),np.uint8))


def test_small_face_rejected():
    with pytest.raises(FaceInputError,match='人脸过小'):
        stub(np.array([face_row(w=20,h=20)])).extract(np.zeros((256,256,3),np.uint8))


def test_blur_rejected():
    with pytest.raises(FaceInputError,match='模糊'):
        stub(np.array([face_row()]),blur=20).extract(np.zeros((256,256,3),np.uint8))
