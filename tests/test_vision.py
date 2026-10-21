import io
import numpy as np
import pytest
from PIL import Image
from faceauth.vision import FaceInputError, FaceEngine, normalize_feature, cosine_similarity, read_photo, read_photo_file


def make_image(w=100, h=100):
    im = Image.new("RGB", (w,h), color=(190,80,20))
    buff=io.BytesIO(); im.save(buff,format="PNG")
    return buff.getvalue()


def test_decode_image():
    im=read_photo(make_image())
    assert im.shape == (100,100,3)
    # RGB is (190,80,20), BGR should be (20,80,190)
    assert tuple(im[0,0]) == (20,80,190)


def test_reject_nonimage():
    for value in (b"",b"not an image"):
        with pytest.raises(FaceInputError):
            read_photo(value)


def test_reject_small_and_oversized_image():
    with pytest.raises(FaceInputError):
        read_photo(make_image(32,32))
    with pytest.raises(FaceInputError):
        read_photo(make_image(4097,65))
    with pytest.raises(FaceInputError):
        read_photo(b"x" * (12*1024*1024+1))


def test_normalized_cosine():
    a = np.zeros(32,np.float32); a[0] = 3
    b = np.zeros(32,np.float32); b[0] = 1
    c = np.zeros(32,np.float32); c[1] = 1
    assert np.linalg.norm(normalize_feature(a)) == pytest.approx(1)
    assert cosine_similarity(a,b) == pytest.approx(1)
    assert cosine_similarity(a,c) == pytest.approx(0)


def test_invalid_feature():
    for x in (np.zeros(32), np.ones(3), np.r_[np.ones(31),float("nan")]):
        with pytest.raises(ValueError):
            normalize_feature(x)
    with pytest.raises(ValueError):
        cosine_similarity(np.ones(32), np.ones(64))


def test_missing_model_friendly_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_models"):
        FaceEngine(model_dir=tmp_path)


def test_disallow_gif():
    with io.BytesIO() as buff:
        Image.new("RGB", (100,100)).save(buff, format="GIF")
        with pytest.raises(FaceInputError, match="仅支持"):
            read_photo(buff.getvalue())
