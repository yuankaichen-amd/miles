"""GLM-5.3: a generation cut at max_tokens inside its reasoning gets ``</think>`` on commit.

The GLM-5.3 template renders every history assistant turn as ``<think>…</think>``,
while a ``length`` stop inside the reasoning never samples ``</think>``. The
session server appends it outside the completion span (``closing_span``), so
the stored tokens stay the canonical render and the token is never trained.

Uses the real GLM-5.3-Flash tokenizer and template from ``GLM53_TOKENIZER_PATH``
(a model directory or hub id); skipped when it cannot be loaded.
"""

import os
from types import SimpleNamespace

import pytest

from miles.rollout.session.errors import TruncatedGenerationError
from miles.rollout.session.samples.merge import compute_samples_from_openai_records
from miles.rollout.session.types import SessionRecord
from miles.rollout.session.v2.session_state import (
    SessionRegistryV2,
    SessionStateV2,
    attach_point_for_request,
    commit_generation,
)
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizerType, get_tito_tokenizer
from miles.utils.processing_utils import load_tokenizer

_ARGS = SimpleNamespace(save_debug_trajectory_data=None, sglang_speculative_algorithm=None)
_USER = {"role": "user", "content": "Tile a 512x512 convolution for MFMA."}
_REASONING = "The output is 512 channels by 16 pixels so a 64 by 64 tile gives eight blocks"
_ANSWER = "Use 64x64 tiles."


@pytest.fixture(scope="module")
def tito():
    path = os.environ.get("GLM53_TOKENIZER_PATH", "zai-org/GLM-5.3-Flash")
    try:
        tokenizer = load_tokenizer(path, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001 - offline CI without the model
        pytest.skip(f"GLM-5.3 tokenizer unavailable at {path!r}: {exc}")
    return get_tito_tokenizer(tokenizer, TITOTokenizerType.GLM53)


@pytest.fixture
def registry(tito):
    return SessionRegistryV2(tokenizer=tito.tokenizer, tito_tokenizer=tito)


def _ids(tito, token: str) -> int:
    return tito.tokenizer.convert_tokens_to_ids(token)


def _encode(tito, text: str) -> list[int]:
    return tito.tokenizer.encode(text, add_special_tokens=False)


def _record(prompt_ids, output_ids, finish_reason, messages) -> SessionRecord:
    return SessionRecord(
        timestamp=0.0,
        method="POST",
        path="/v1/chat/completions",
        status_code=200,
        request={"messages": messages, "input_ids": list(prompt_ids)},
        response={
            "choices": [
                {
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": finish_reason,
                    "meta_info": {"output_token_logprobs": [(-0.5, t) for t in output_ids]},
                }
            ]
        },
    )


def _commit(state, tito, request_messages, assistant_message, completion_ids, finish_reason):
    parent = attach_point_for_request(state, request_messages).node
    if parent is None:
        prompt_ids = tito.apply_chat_template(request_messages, add_generation_prompt=True, tokenize=True)
    else:
        prompt_ids = tito.merge_tokens(
            old_messages=parent.path_messages(), new_messages=request_messages, pretokenized_token_ids=parent.token_ids
        )
    node = commit_generation(
        state,
        parent=parent,
        request_messages=request_messages,
        assistant_message=assistant_message,
        prompt_token_ids=prompt_ids,
        completion_token_ids=completion_ids,
        max_trim_tokens=tito.max_trim_tokens,
        record=_record(prompt_ids, completion_ids, finish_reason, request_messages),
        response_id=f"resp-{len(state.tree.nodes)}",
        finish_reason=finish_reason,
        tito_tokenizer=tito,
    )
    return node, prompt_ids


def _render(tito, messages, *, add_generation_prompt=False):
    return tito.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=True)


def test_generation_prompt_opens_reasoning(tito):
    prompt = _render(tito, [_USER], add_generation_prompt=True)
    assert prompt[-1] == _ids(tito, "<think>")


def test_truncated_mid_thinking_appends_closing_think(tito, registry):
    state = SessionStateV2()
    completion = _encode(tito, _REASONING)
    assistant = {"role": "assistant", "content": "", "reasoning_content": _REASONING}
    node, prompt = _commit(state, tito, [_USER], assistant, completion, "length")

    generated = prompt + completion
    close = _ids(tito, "</think>")
    assert node.completion_span == (len(prompt), len(generated))
    assert node.closing_span == (len(generated), len(generated) + 1)
    assert node.token_ids == generated + [close]
    assert node.generated_token_ids == generated

    assert node.token_ids == _render(tito, [_USER, assistant])
    assert registry.compute_session_mismatch(state) == []
    # Without the closing span the stored tokens differ from the render by exactly ``</think>``.
    assert registry.compute_mismatch([_USER, assistant], generated, turn_args=node.turn_args)

    # The sample is the sampled tokens only: ``</think>`` carries no loss and no logprob.
    (sample,) = compute_samples_from_openai_records(
        _ARGS, [node.record], tito.tokenizer, accumulated_token_ids=node.generated_token_ids
    )
    assert sample.tokens == generated
    assert sample.loss_mask == [1] * len(completion)
    assert len(sample.rollout_log_probs) == len(completion)


def test_truncated_after_closing_think_appends_nothing(tito, registry):
    state = SessionStateV2()
    completion = _encode(tito, _REASONING) + [_ids(tito, "</think>")] + _encode(tito, _ANSWER)
    assistant = {"role": "assistant", "content": _ANSWER, "reasoning_content": _REASONING}
    node, prompt = _commit(state, tito, [_USER], assistant, completion, "length")

    assert node.closing_span is None
    assert node.token_ids == node.generated_token_ids == prompt + completion
    assert tito.close_truncated_reasoning(node.token_ids) == []


def test_normal_stop_appends_nothing(tito, registry):
    state = SessionStateV2()
    stop = _ids(tito, "<|user|>")
    completion = _encode(tito, _REASONING) + [_ids(tito, "</think>")] + _encode(tito, _ANSWER) + [stop]
    assistant = {"role": "assistant", "content": _ANSWER, "reasoning_content": _REASONING}
    node, prompt = _commit(state, tito, [_USER], assistant, completion, "stop")

    assert node.closing_span is None
    assert node.token_ids == prompt + completion
    # Only a ``length`` stop is closed, even if the reasoning were somehow still open.
    open_node, _ = _commit(SessionStateV2(), tito, [_USER], assistant, _encode(tito, _REASONING), "stop")
    assert open_node.closing_span is None


def test_continuing_truncated_turn_renders_after_session_tokens(tito, registry):
    state = SessionStateV2()
    completion = _encode(tito, _REASONING)
    assistant = {"role": "assistant", "content": "", "reasoning_content": _REASONING}
    node, prompt = _commit(state, tito, [_USER], assistant, completion, "length")
    follow_up = [_USER, assistant, {"role": "user", "content": "Continue."}]

    # Serving policy is unchanged: a truncated node is never extended in place.
    with pytest.raises(TruncatedGenerationError):
        attach_point_for_request(state, follow_up)

    # The canonical render of a continuation starts with the stored session tokens,
    # and TITO's incremental merge on top of them reproduces that render.
    full = _render(tito, follow_up, add_generation_prompt=True)
    assert full[: len(node.token_ids)] == node.token_ids
    merged = tito.merge_tokens(
        old_messages=node.path_messages(), new_messages=follow_up, pretokenized_token_ids=node.token_ids
    )
    assert merged == full

    # In a continuation turn ``</think>`` is prompt, so its loss mask is 0.
    closing_index = node.closing_span[0]
    output = _encode(tito, _ANSWER) + [_ids(tito, "<|user|>")]
    records = [
        node.record,
        _record(merged, output, "stop", follow_up),
    ]
    first, second = compute_samples_from_openai_records(
        _ARGS, records, tito.tokenizer, accumulated_token_ids=merged + output
    )
    assert first.tokens == node.generated_token_ids
    assert second.tokens[closing_index] == _ids(tito, "</think>")
    response_start = len(second.tokens) - second.response_length
    assert closing_index < response_start
    assert second.loss_mask == [1] * len(output)
