import numpy as np
import pytest
from app.edge_models.preprocess import URL_ALPHABET, SMS_ALPHABET, encode_chars, one_hot_chars
from app.edge_models.config import DEFAULT_SPECS
from app.ml.flow_reducer import MODEL_FEATURES


def test_character_encoder_is_fixed_length_and_unknown_safe():
    ids = encode_chars("HTTPS://Exämple.com", URL_ALPHABET, 32)
    assert ids.shape == (32,)
    assert ids.dtype == np.int64
    assert np.all(ids >= 0)
    assert ids[0] == URL_ALPHABET.index("h") + 2


def test_one_hot_shape_and_single_active_character():
    x = one_hot_chars("ab", URL_ALPHABET, 8)
    assert x.shape == (len(URL_ALPHABET) + 2, 8)
    assert np.allclose(x.sum(axis=0), 1.0)


def test_sms_preprocessor_shape():
    x = one_hot_chars("Your bank account is blocked", SMS_ALPHABET, 512)
    assert x.shape == (len(SMS_ALPHABET) + 2, 512)


def test_rk3588_core_assignment_is_explicit():
    assert DEFAULT_SPECS["network"]["core_id"] == 0
    assert DEFAULT_SPECS["url"]["core_id"] == 1
    assert DEFAULT_SPECS["sms"]["core_id"] == 2
    assert len(MODEL_FEATURES) == DEFAULT_SPECS["network"]["input_shape"][1]


def test_invalid_core_is_rejected_before_sdk_load(tmp_path):
    from app.edge_models.rknn_pool import RKNNModel
    with pytest.raises(ValueError):
        RKNNModel("url", str(tmp_path / "missing.rknn"), 3, (61, 512))
