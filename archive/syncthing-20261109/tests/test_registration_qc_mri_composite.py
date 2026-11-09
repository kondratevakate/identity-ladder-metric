import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from registration_qc_mri_composite import physical_grid, pull_determinant, reflected_slice


def test_full_physical_jacobian_includes_affine_and_direction():
    direction = np.diag([-1., -1., 1.])
    spacing = [1., 2., 3.]
    xyz = physical_grid((8, 9, 10), [-10., 20., 5.], spacing, direction)
    matrix = np.array([[1.2, .1, 0], [0, .8, .2], [0, 0, 1.1]])
    moved = xyz @ matrix.T + [12, -4, 8]
    assert np.allclose(pull_determinant(moved, spacing, direction), np.linalg.det(matrix), atol=1e-5)


def test_half_open_reflection_has_exact_indices_and_is_involutive():
    original = np.arange(19)
    for start, stop in [(0, 3), (2, 9), (13, 19)]:
        low, high = reflected_slice(start, stop, len(original))
        assert np.array_equal(original[::-1][start:stop], original[low:high][::-1])
        assert reflected_slice(low, high, len(original)) == [start, stop]
