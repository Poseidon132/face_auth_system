from pathlib import Path
import hashlib
import pytest
from scripts.download_models import sha256, validated, fetch
from scripts.eval_pairs import safe_image_path, get_pairs


def test_sha_small_file(tmp_path):
    p=tmp_path/'file.onnx'
    p.write_bytes(b"123")
    assert sha256(p) == hashlib.sha256(b"123").hexdigest()
    assert not validated(p,sha256(p))


def test_safe_image_root(tmp_path):
    p=tmp_path/'abc.jpg';p.write_bytes(b'')
    assert safe_image_path(tmp_path,'abc.jpg') == p
    with pytest.raises(ValueError):
        safe_image_path(tmp_path,'../some.jpg')


def test_pair_csv_validation(tmp_path):
    csvpath=tmp_path/'pairs.csv'
    csvpath.write_text('image_a,image_b,same\na.jpg,b.jpg,1\na.jpg,c.jpg,0\n',encoding='utf-8')
    assert get_pairs(csvpath) == [('a.jpg','b.jpg',1),('a.jpg','c.jpg',0)]
    csvpath.write_text('image_a,image_b,same\na,b,3\n')
    with pytest.raises(ValueError):
        get_pairs(csvpath)
