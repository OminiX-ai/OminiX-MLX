"""Geometry/order regressions; the independent reference test checks model parity."""

import numpy as np
import pytest
from pear_server import Engine, image_positions, language_positions, split_answer


def test_image_positions_keep_merge_order_and_interpolate_corners():
    indices, weights, cos, sin = image_positions((1, 4, 6), 48, 2, 72)
    assert indices.shape == weights.shape == (4, 24)
    np.testing.assert_allclose(weights.sum(axis=0), 1)
    assert indices[0, 0] == 0
    assert indices[-1, -1] == 2303
    assert cos.shape == sin.shape == (1, 1, 24, 72)
    # First 2x2 merge block: top left, top right, bottom left, bottom right.
    np.testing.assert_allclose(cos[0, 0, 0], 1)
    assert cos[0, 0, 1, 18] == pytest.approx(np.cos(1))
    assert cos[0, 0, 2, 0] == pytest.approx(np.cos(1))


def test_multimodal_positions_compress_each_image_and_continue_decode():
    pos = language_positions(12, [(2, 6)], [(1, 4, 6)], 3, 2)
    np.testing.assert_array_equal(pos[:, :2], [[0, 1]] * 3)
    np.testing.assert_array_equal(pos[:, 2:8], [[2] * 6, [2, 2, 2, 3, 3, 3], [2, 3, 4, 2, 3, 4]])
    np.testing.assert_array_equal(pos[:, 8:], [list(range(5, 12))] * 3)


def test_bad_image_layout_rejected():
    with pytest.raises(ValueError):
        language_positions(12, [(2, 5)], [(1, 4, 6)], 3, 2)
    with pytest.raises(ValueError):
        image_positions((2, 4, 6), 48, 2, 72)


def test_reasoning_never_becomes_robot_action():
    assert split_answer("<think>move dangerous distance</think>```json\n{}\n```", True) == (
        "```json\n{}\n```",
        "move dangerous distance",
    )
    assert split_answer("unfinished thought", True) == ("", "unfinished thought")
    assert split_answer("answer", False) == ("answer", None)


@pytest.mark.parametrize(
    "body",
    [
        [],
        {"messages": [], "max_tokens": True},
        {"messages": [], "max_tokens": 4097},
        {"messages": [], "stream": True},
        {"messages": [], "temperature": 0.5},
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/camera.png"},
                        }
                    ],
                }
            ]
        },
    ],
)
def test_bad_requests_fail_before_model_work(body, tmp_path):
    engine = object.__new__(Engine)
    with pytest.raises(ValueError):
        engine.prepare(body, tmp_path)
