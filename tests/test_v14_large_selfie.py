"""Mock-model regression tests for high-resolution portrait detection fallback."""
import threading
import numpy as np
import pytest
from faceauth.vision import FaceEngine, FaceInputError, FaceObservation


def face_row(x, y, w, h, conf=0.94):
    # landmarks chosen with known nonzero values so both axes can be tested.
    points = [x+10,y+12,x+w-10,y+12,x+w/2,y+h/2,x+20,y+h-12,x+w-20,y+h-12]
    return np.array([x,y,w,h,*points,conf], dtype=np.float32)

class Detector:
    def __init__(self, per_size):
        self.per_size=per_size
        self.calls=[]
    def setInputSize(self, size):
        self.size=tuple(size)
    def detect(self, image):
        assert (image.shape[1],image.shape[0])==self.size
        self.calls.append(self.size)
        rows=self.per_size.get(self.size)
        return 1, rows

class Recognizer:
    def alignCrop(self, image, face):
        self.original_size=(image.shape[1],image.shape[0]);self.last_face=face.copy()
        return np.ones((112,112,3),np.uint8)
    def feature(self, crop): return np.ones((1,128),np.float32)

def mock_engine(per_size):
    e=object.__new__(FaceEngine)
    e._lock=threading.RLock()
    e.detector=Detector(per_size)
    e.recognizer=Recognizer()
    e.min_face_size=55
    e.min_blur_score=0.0
    e.model_id='mock'
    return e

def test_large_vertical_selfie_fallback_and_landmark_restoration():
    # 720x1600: v1.3 used the entire portrait; v1.4 tries 432x960 then 288x640.
    short=np.stack([face_row(30,200,190,230)])
    e=mock_engine({(432,960):None,(288,640):short})
    result=e.extract(np.zeros((1600,720,3),np.uint8),annotate=False)
    assert e.detector.calls==[(432,960),(288,640)]
    assert result.detection_input_size==(288,640)
    assert result.box_xywh==(75,500,475,575)
    assert result.detected_face_count==1
    assert e.recognizer.original_size==(720,1600)
    assert e.recognizer.last_face[4] == pytest.approx((30+10)*2.5)
    assert e.recognizer.last_face[5] == pytest.approx((200+12)*2.5)
    assert result.detection_score == pytest.approx(.94)

def test_multi_face_at_fallback_still_rejected():
    rows=np.stack([face_row(20,80,110,150),face_row(160,220,80,100)])
    e=mock_engine({(432,960):None,(288,640):rows})
    with pytest.raises(FaceInputError,match="检测到 2"):
        e.extract(np.zeros((1600,720,3),np.uint8),annotate=False)
    assert e.detector.calls==[(432,960),(288,640)]

def test_no_detection_remains_failure_not_wrong_identity():
    e=mock_engine({})
    with pytest.raises(FaceInputError,match="未检测到人脸"):
        e.extract(np.zeros((1600,720,3),np.uint8),annotate=False)
    assert e.detector.calls==[(432,960),(288,640),(216,480)]

def test_small_image_keeps_original_scale_first():
    e=mock_engine({(250,250):np.stack([face_row(10,10,120,120)])})
    result=e.extract(np.zeros((250,250,3),np.uint8),annotate=False)
    assert e.detector.calls==[(250,250)]
    assert result.detection_input_size==(250,250)

def test_lfw_largest_keeps_original_detection_input_even_large_photo():
    e=mock_engine({(720,1600):np.stack([face_row(50,50,140,140),face_row(20,60,80,80)])})
    result=e.extract(np.zeros((1600,720,3),np.uint8),annotate=False,selection_policy='lfw-largest')
    assert e.detector.calls==[(720,1600)]
    assert result.detected_face_count==2

def test_live_multi_face_at_first_pass_blocks_fallback():
    e=mock_engine({(432,960):np.stack([face_row(10,40,150,180),face_row(200,40,70,90)])})
    with pytest.raises(FaceInputError,match="检测到 2"):
        e.extract(np.zeros((1600,720,3),np.uint8),annotate=False)
    assert e.detector.calls==[(432,960)]
