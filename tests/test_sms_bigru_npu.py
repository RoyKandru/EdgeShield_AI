"""Tests for Phap (2026) BiGRU+MaskedPool+WBCE on RK3588J NPU_CORE_2."""
from __future__ import annotations

import torch
from torch import nn

from app.detectors.sms import analyse_sms
from app.edge_models.models import FullSplitBiGRUMaskedPool
from app.edge_models.sms_bigru_npu import get_sms_bigru_npu_client, preprocess_sms_tokens


def test_entity_abstraction_and_deobfuscation():
    toks = preprocess_sms_tokens("U R G E N T ! Visit http://fake-kyc.xyz/login or dial *401*9876543210# Rs. 5000 ?")
    assert "urgent" in toks
    assert "<url>" in toks
    assert "<ussd>" in toks
    assert "<cur>" in toks
    assert "<num>" in toks
    assert "!" in toks
    assert "?" in toks


def test_split_bigru_matches_pack_padded_sequence_exactly():
    torch.manual_seed(7)
    vocab_size, emb_dim, hidden, t_max, L = 50, 32, 32, 16, 7
    split = FullSplitBiGRUMaskedPool(vocab_size, emb_dim, hidden, dropout=0.0).eval()
    native = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=True).eval()
    with torch.no_grad():
        for k in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
            getattr(native, k).copy_(getattr(split.gru_fwd, k))
            getattr(native, f"{k}_reverse").copy_(getattr(split.gru_bwd, k))

        ids = torch.randint(2, vocab_size, (1, L))
        ids_fwd = torch.zeros(1, t_max, dtype=torch.long)
        ids_bwd = torch.zeros(1, t_max, dtype=torch.long)
        ids_fwd[0, :L] = ids[0]
        ids_bwd[0, :L] = torch.flip(ids[0], dims=[0])
        pw = torch.zeros(1, 1, t_max)
        pw[0, 0, :L] = 1.0 / L

        split_logit = split(ids_fwd, ids_bwd, pw).item()
        emb = split.embedding(ids_fwd)
        packed = nn.utils.rnn.pack_padded_sequence(emb, torch.tensor([L]), batch_first=True, enforce_sorted=False)
        out_p, _ = native(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out_p, batch_first=True, total_length=t_max)
        pooled = torch.bmm(pw, unpacked).squeeze(1)
        native_logit = split.fc(pooled).squeeze(-1).item()
        assert abs(split_logit - native_logit) < 1e-5


def test_live_rk3588_npu_core2_inference_and_sms_fusion():
    client = get_sms_bigru_npu_client()
    assert client.ready, f"Expected RK3588 NPU_CORE_2 BiGRU ready on board, got: {client.status()}"

    benign = client.infer("Hey, are we still meeting for coffee at 5pm today?")
    scam = client.infer("URGENT: You won a free cash reward! Call 08001234567 or click http://claim-prize-now.xyz immediately!")
    assert benign["backend"] == "rknn-npu"
    assert benign["core_id"] == 2
    assert benign["core"] == "NPU_CORE_2"
    assert benign["npu_latency_ms"] < 50.0
    assert benign["score"] < 0.25
    assert scam["score"] > 0.80

    # End-to-end hybrid SMS detector fusion check (including genuine bank OTP guard!)
    ev_scam = analyse_sms("WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Claim call 09061701461.")
    assert ev_scam.model_version == "sms-hybrid-v3-npu-bigru"
    assert ev_scam.prediction in ("likely_scam", "suspicious")
    assert ev_scam.metadata.get("npu_bigru_backend") == "rknn-npu"
    assert ev_scam.metadata.get("npu_bigru_core") == "NPU_CORE_2"

    ev_otp = analyse_sms("849201 is your OTP for Rs. 450 transaction at Swiggy on HDFC Card XX0000. Do NOT share this OTP with anyone.")
    assert ev_otp.prediction == "no_obvious_indicators"

    ev_rbi = analyse_sms("RBI Kehta Hai: Stay cyber safe. Never share your OTP, UPI PIN, CVV or install AnyDesk on request.")
    assert ev_rbi.prediction == "no_obvious_indicators"
