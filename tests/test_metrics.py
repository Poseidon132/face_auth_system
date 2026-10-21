import numpy as np
import pytest
from faceauth.metrics import rates_at, auc_rank, choose_threshold_on_validation, approximate_eer


def test_perfect_binary_metrics():
    s=[.8,.9,-.1,.1]; y=[1,1,0,0]
    m=rates_at(s,y,.5)
    assert m['accuracy']==1
    assert m['far']==0 and m['frr']==0
    assert auc_rank(s,y)==1
    cal=choose_threshold_on_validation(s,y,target_far=0.0)
    assert cal['far']==0 and cal['tar']==1
    assert .1 < cal['threshold'] <= .8
    assert approximate_eer(s,y)['eer_approx'] == 0


def test_overlap():
    y=[1,1,0,0]; s=[.7,.3,.6,.2]
    r=rates_at(s,y,.5)
    assert r['accuracy']==.5
    assert r['far']==.5 and r['frr']==.5
    assert auc_rank(s,y)==.75


def test_auc_with_ties():
    assert auc_rank([.5,.5], [1,0]) == .5
    assert auc_rank([.5,.5,.7,.3], [1,0,1,0]) == .875


def test_no_negative():
    assert auc_rank([.8,.7],[1,1]) is None
    assert approximate_eer([.8,.7],[1,1]) is None
    with pytest.raises(ValueError):
        choose_threshold_on_validation([.8,.7],[1,1])


def test_invalid_labels():
    with pytest.raises(ValueError):
        rates_at([.1,.2],[1,5], .3)
    with pytest.raises(ValueError):
        rates_at([float('nan')],[1], .3)
