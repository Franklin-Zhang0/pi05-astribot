from pathlib import Path

import numpy as np
import pytest

from openpi.models import tokenizer as _tokenizer


@pytest.fixture
def local_paligemma(monkeypatch):
    path = Path("/tmp/engram-openpi-cache/big_vision/paligemma_tokenizer.model")
    if not path.is_file():
        pytest.skip("Cached PaliGemma sentencepiece model is unavailable")
    monkeypatch.setattr(_tokenizer.download, "maybe_download", lambda *_a, **_kw: path)


def test_json_prompt_preserves_underscores_with_real_sentencepiece(local_paligemma):
    prompt = '{"instruction":"pick_up cup","robot_type":"dex_w1","end_effector_type":"linker_l20"}'
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=256, preserve_text=True, reject_truncation=True)
    actual, mask = tokenizer.tokenize(prompt)
    expected = tokenizer._tokenizer.encode(prompt, add_bos=True) + tokenizer._tokenizer.encode("\n")  # noqa: SLF001
    np.testing.assert_array_equal(actual[mask], expected)
    legacy = _tokenizer.PaligemmaTokenizer(max_len=256)
    legacy_tokens, legacy_mask = legacy.tokenize(prompt)
    assert not np.array_equal(legacy_tokens[legacy_mask], actual[mask])


def test_pi05_json_state_truncation_is_rejected_at_exact_boundary(local_paligemma):
    prompt = '{"instruction":"pick_up cup","robot_type":"dex_w1","end_effector_type":"linker_l20"}'
    state = np.linspace(-1, 1, 64, dtype=np.float32)
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=1024, preserve_text=True, reject_truncation=True)
    tokens, mask = tokenizer.tokenize(prompt, state)
    count = int(mask.sum())
    boundary = _tokenizer.PaligemmaTokenizer(max_len=count, preserve_text=True, reject_truncation=True)
    exact, exact_mask = boundary.tokenize(prompt, state)
    np.testing.assert_array_equal(exact, tokens[mask])
    assert exact_mask.all()
    truncated = _tokenizer.PaligemmaTokenizer(max_len=count - 1, preserve_text=True, reject_truncation=True)
    with pytest.raises(ValueError, match=f"require {count} tokens"):
        truncated.tokenize(prompt, state)
    # Legacy callers keep their original opt-in-free truncation behavior.
    legacy = _tokenizer.PaligemmaTokenizer(max_len=10)
    assert legacy.tokenize(prompt, state)[0].shape == (10,)


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)
