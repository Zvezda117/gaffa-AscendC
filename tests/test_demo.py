import gaffa
import pytest


def test_vector_add_empty() -> None:
    assert gaffa.vector_add([], []) == []


def test_ascend_device_count_is_non_negative() -> None:
    assert gaffa.ascend_device_count() >= 0


@pytest.mark.skipif(
    gaffa.ascend_device_count() == 0,
    reason="Ascend device is not visible",
)
def test_vector_add_ascend_kernel() -> None:
    assert gaffa.vector_add(
        [1.0, 2.5, -3.0], [2.0, 0.5, 3.0]
    ) == [3.0, 3.0, 0.0]
