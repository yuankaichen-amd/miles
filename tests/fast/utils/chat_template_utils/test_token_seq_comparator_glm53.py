"""TokenSeqComparator with the GLM-5.3 TITO tokenizer: assistant boundary whitespace.

A generation cut at max_tokens on a token ending in whitespace keeps that whitespace in the
stored tokens, while the canonical re-render of the message drops it. The two cases in
``glm53_truncated_tail_whitespace.json`` are the Phase 0a check 7 mismatches of that kind
(decoder5_cd28b19f episodes 59b29cd0e600 leaf 6 and 8ad833915a7a leaf 17). Assistant content is compared with
leading/trailing whitespace ignored; anything else must still be reported.

Uses the real GLM-5.3-Flash tokenizer from ``GLM53_TOKENIZER_PATH``; skipped when it cannot be loaded.
"""

import json
import os
from pathlib import Path

import pytest

from miles.utils.chat_template_utils.token_seq_comparator import MismatchType
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizerType, get_tito_tokenizer
from miles.utils.processing_utils import load_tokenizer

CASES = json.loads((Path(__file__).parent / "glm53_truncated_tail_whitespace.json").read_text())["cases"]


@pytest.fixture(scope="module")
def tito():
    path = os.environ.get("GLM53_TOKENIZER_PATH", "zai-org/GLM-5.3-Flash")
    try:
        tokenizer = load_tokenizer(path, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001 - offline CI without the model
        pytest.skip(f"GLM-5.3 tokenizer unavailable at {path!r}: {exc}")
    return get_tito_tokenizer(tokenizer, TITOTokenizerType.GLM53)


def _turn(tito, role_token: str, text: str, next_token: str = "<|user|>") -> list[int]:
    tok = tito.tokenizer
    return (
        [tok.convert_tokens_to_ids(role_token)]
        + tok.encode(text, add_special_tokens=False)
        + [tok.convert_tokens_to_ids(next_token)]
    )


@pytest.mark.parametrize("case", CASES, ids=[f"leaf{c['leaf_node_id']}" for c in CASES])
def test_check7_tail_whitespace_cases_match(tito, case):
    assert case["actual_text"] != case["expected_text"]
    assert case["actual_text"].strip() == case["expected_text"].strip()
    comparator = tito.create_comparator()
    expected = _turn(tito, "<|assistant|>", case["expected_text"])
    actual = _turn(tito, "<|assistant|>", case["actual_text"])
    assert comparator.compare_sequences(expected, actual) == []


@pytest.mark.parametrize("case", CASES, ids=[f"leaf{c['leaf_node_id']}" for c in CASES])
def test_non_whitespace_assistant_change_still_fails(tito, case):
    text = case["actual_text"]
    changed = text.replace("conv", "CONV", 1)
    assert changed != text
    comparator = tito.create_comparator()
    result = comparator.compare_sequences(
        _turn(tito, "<|assistant|>", case["expected_text"]), _turn(tito, "<|assistant|>", changed)
    )
    assert [m.type for m in result] == [MismatchType.ASSISTANT_TEXT]


def test_inner_whitespace_change_in_assistant_content_still_fails(tito):
    comparator = tito.create_comparator()
    result = comparator.compare_sequences(
        _turn(tito, "<|assistant|>", "<think>tile 64x64</think>Use 64x64 tiles."),
        _turn(tito, "<|assistant|>", "<think>tile 64x64</think>Use 64x64\ntiles."),
    )
    assert [m.type for m in result] == [MismatchType.ASSISTANT_TEXT]


def test_boundary_whitespace_in_user_content_still_fails(tito):
    comparator = tito.create_comparator()
    result = comparator.compare_sequences(
        _turn(tito, "<|user|>", "Tile a 512x512 convolution.", "<|assistant|>"),
        _turn(tito, "<|user|>", "Tile a 512x512 convolution.\n", "<|assistant|>"),
    )
    assert [m.type for m in result] == [MismatchType.NON_ASSISTANT_TEXT]
