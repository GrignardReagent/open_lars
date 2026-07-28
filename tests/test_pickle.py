"""LARS must survive (de)serialisation with its hyperparameters intact."""

import copy
import pickle

import pytest
import torch

from open_lars import LARS


@pytest.fixture
def lars():
    return LARS([torch.zeros(2, 2)], lr=0.1, momentum=0.8,
                weight_decay=1e-4, trust_coefficient=0.42)


def assert_hyperparams_preserved(opt):
    assert isinstance(opt, LARS)
    group = opt.param_groups[0]
    assert group['lr'] == 0.1
    assert group['momentum'] == 0.8
    assert group['weight_decay'] == 1e-4
    assert group['trust_coefficient'] == 0.42


def test_pickle_roundtrip(lars):
    assert_hyperparams_preserved(pickle.loads(pickle.dumps(lars)))


def test_copy(lars):
    assert_hyperparams_preserved(copy.copy(lars))


def test_deepcopy(lars):
    assert_hyperparams_preserved(copy.deepcopy(lars))
