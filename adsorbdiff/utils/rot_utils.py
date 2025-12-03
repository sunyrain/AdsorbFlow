import os
import pickle
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import ase.io
import torch
import math

MIN_EPS, MAX_EPS, N_EPS = 0.01, 2, 1000
X_N = 2000

"""
    Preprocessing for the SO(3) sampling and score computations, truncated infinite series are computed and then
    cached to memory, therefore the precomputation is only run the first time the repository is run on a machine
"""


def quaternion_to_matrix(quaternions):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_quaternion(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles],
        dim=-1,
    )
    return quaternions


def axis_angle_to_matrix(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to axis-angle vectors."""
    single = matrix.dim() == 2
    if single:
        matrix = matrix.unsqueeze(0)
    rotvec = Rotation.from_matrix(matrix.detach().cpu().numpy()).as_rotvec()
    rotvec = torch.from_numpy(rotvec).to(matrix.device, dtype=matrix.dtype)
    if single:
        rotvec = rotvec[0]
    return rotvec


def random_quaternions(count, device=None, dtype=None):
    """Sample quaternions that are uniform on SO(3)."""
    if dtype is None:
        dtype = torch.float32
    q = torch.randn(count, 4, device=device, dtype=dtype)
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    # Ensure a consistent hemisphere to avoid ambiguity during slerp.
    q = torch.where(q[..., :1] < 0.0, -q, q)
    return q


def quaternion_multiply(q1, q2):
    """Hamilton product of two quaternions (real part first)."""
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q2, dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    out = torch.stack((w, x, y, z), dim=-1)
    return out


def quaternion_to_axis_angle(quaternions, eps: float = 1.0e-8):
    """Convert quaternions (real part first) to axis-angle vectors."""
    q = quaternions / torch.linalg.norm(quaternions, dim=-1, keepdim=True)
    q = torch.where(q[..., :1] < 0.0, -q, q)
    xyz = q[..., 1:]
    qw = torch.clamp(q[..., :1], -1.0, 1.0)
    sin_theta = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_theta, qw)
    axis = xyz / torch.clamp(sin_theta, min=eps)
    axis_angle = axis * angle
    small = sin_theta < eps
    if small.any():
        axis_angle = torch.where(small, 2.0 * xyz, axis_angle)
    return axis_angle


def quaternion_slerp(q0, q1, t, eps: float = 1.0e-6):
    """Spherical linear interpolation between two quaternions."""
    q0 = q0 / torch.linalg.norm(q0, dim=-1, keepdim=True)
    q1 = q1 / torch.linalg.norm(q1, dim=-1, keepdim=True)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.clamp(torch.sum(q0 * q1, dim=-1, keepdim=True), -1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    while t.dim() < q0.dim():
        t = t.unsqueeze(-1)
    sin_omega_safe = torch.clamp(sin_omega, min=eps)
    coeff0 = torch.sin((1.0 - t) * omega) / sin_omega_safe
    coeff1 = torch.sin(t * omega) / sin_omega_safe
    result = coeff0 * q0 + coeff1 * q1
    linear = (1.0 - t) * q0 + t * q1
    use_linear = sin_omega.abs() < eps
    result = torch.where(use_linear, linear, result)
    result = result / torch.linalg.norm(result, dim=-1, keepdim=True)
    return result


def rigid_transform_Kabsch_3D_torch(A, B):
    # R = 3x3 rotation matrix, t = 3x1 column vector
    # This already takes residue identity into account.

    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise: 3 x 1
    centroid_A = torch.mean(A, axis=1, keepdims=True)
    centroid_B = torch.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = torch.linalg.svd(H)

    R = Vt.T @ U.T
    # special reflection case
    if torch.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = torch.diag(torch.tensor([1.0, 1.0, -1.0], device=A.device))
        R = (Vt.T @ SS) @ U.T
    assert (
        math.fabs(torch.linalg.det(R) - 1) < 3e-3
    )  # note I had to change this error bound to be higher

    t = -R @ centroid_A + centroid_B
    return R, t


omegas = np.linspace(0, np.pi, X_N + 1)[1:]


def _compose(r1, r2):  # R1 @ R2 but for Euler vecs
    return Rotation.from_matrix(
        Rotation.from_rotvec(r1).as_matrix()
        @ Rotation.from_rotvec(r2).as_matrix()
    ).as_rotvec()


def _expansion(omega, eps, L=2000):  # the summation term only
    p = 0
    for l in range(L):
        p += (
            (2 * l + 1)
            * np.exp(-l * (l + 1) * eps**2)
            * np.sin(omega * (l + 1 / 2))
            / np.sin(omega / 2)
        )
    return p


def _density(
    expansion, omega, marginal=True
):  # if marginal, density over [0, pi], else over SO(3)
    if marginal:
        return expansion * (1 - np.cos(omega)) / np.pi
    else:
        return (
            expansion / 8 / np.pi**2
        )  # the constant factor doesn't affect any actual calculations though


def _score(exp, omega, eps, L=2000):  # score of density over SO(3)
    dSigma = 0
    for l in range(L):
        hi = np.sin(omega * (l + 1 / 2))
        dhi = (l + 1 / 2) * np.cos(omega * (l + 1 / 2))
        lo = np.sin(omega / 2)
        dlo = 1 / 2 * np.cos(omega / 2)
        dSigma += (
            (2 * l + 1)
            * np.exp(-l * (l + 1) * eps**2)
            * (lo * dhi - hi * dlo)
            / lo**2
        )
    return dSigma / exp


PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "so3_precompute")
if os.path.exists(os.path.join(PATH, "so3_omegas_array2.npy")):
    _omegas_array = np.load(os.path.join(PATH, "so3_omegas_array2.npy"))
    _cdf_vals = np.load(os.path.join(PATH, "so3_cdf_vals2.npy"))
    _score_norms = np.load(os.path.join(PATH, "so3_score_norms2.npy"))
    _exp_score_norms = np.load(os.path.join(PATH, "so3_exp_score_norms2.npy"))
else:
    print("Precomputing and saving to cache SO(3) distribution table")
    _eps_array = 10 ** np.linspace(np.log10(MIN_EPS), np.log10(MAX_EPS), N_EPS)
    _omegas_array = np.linspace(0, np.pi, X_N + 1)[1:]

    _exp_vals = np.asarray(
        [_expansion(_omegas_array, eps) for eps in _eps_array]
    )
    _pdf_vals = np.asarray(
        [_density(_exp, _omegas_array, marginal=True) for _exp in _exp_vals]
    )
    _cdf_vals = np.asarray([_pdf.cumsum() / X_N * np.pi for _pdf in _pdf_vals])
    _score_norms = np.asarray(
        [
            _score(_exp_vals[i], _omegas_array, _eps_array[i])
            for i in range(len(_eps_array))
        ]
    )

    _exp_score_norms = np.sqrt(
        np.sum(_score_norms**2 * _pdf_vals, axis=1)
        / np.sum(_pdf_vals, axis=1)
        / np.pi
    )

    np.save(os.path.join(PATH, "so3_omegas_array2.npy"), _omegas_array)
    np.save(os.path.join(PATH, "so3_cdf_vals2.npy"), _cdf_vals)
    np.save(os.path.join(PATH, "so3_score_norms2.npy"), _score_norms)
    np.save(os.path.join(PATH, "so3_exp_score_norms2.npy"), _exp_score_norms)


def sample(eps):
    eps_idx = (
        (np.log10(eps) - np.log10(MIN_EPS))
        / (np.log10(MAX_EPS) - np.log10(MIN_EPS))
        * N_EPS
    )
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)

    x = np.random.rand()
    return np.interp(x, _cdf_vals[eps_idx], _omegas_array)


def sample_vec(eps):
    x = np.random.randn(3)
    x /= np.linalg.norm(x)
    return x * sample(eps)


def score_vec(eps, vec):
    eps_idx = (
        (np.log10(eps) - np.log10(MIN_EPS))
        / (np.log10(MAX_EPS) - np.log10(MIN_EPS))
        * N_EPS
    )
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)

    om = np.linalg.norm(vec)
    return np.interp(om, _omegas_array, _score_norms[eps_idx]) * vec / om


def score_norm(eps):
    eps = eps.numpy()
    eps_idx = (
        (np.log10(eps) - np.log10(MIN_EPS))
        / (np.log10(MAX_EPS) - np.log10(MIN_EPS))
        * N_EPS
    )
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min=0, a_max=N_EPS - 1)
    return torch.from_numpy(_exp_score_norms[eps_idx]).float()


def rotate_atoms(atoms):

    # Rotate around the z-axis
    zrot = torch.rand(1) * 360
    zrot_rad = zrot * (math.pi / 180)  # Convert to radians
    rotation_matrix = torch.tensor(
        [
            [torch.cos(zrot_rad), -torch.sin(zrot_rad), 0],
            [torch.sin(zrot_rad), torch.cos(zrot_rad), 0],
            [0, 0, 1],
        ],
        device=atoms.device,
    )
    center = atoms.mean(dim=0)
    atoms_centered = atoms - center
    atoms_rotated_z = torch.mm(atoms_centered, rotation_matrix) + center

    # Generate a random rotation vector
    z = torch.rand(1) * 2 - 1
    phi = torch.rand(1) * 2 * math.pi
    rotvec = torch.tensor(
        [
            torch.sqrt(1 - z**2) * torch.cos(phi),
            torch.sqrt(1 - z**2) * torch.sin(phi),
            z,
        ],
        device=atoms.device,
    )

    # Rotate atoms using the generated rotation vector
    rotation_matrix = torch.tensor(
        [
            [
                1 - 2 * rotvec[1] ** 2 - 2 * rotvec[2] ** 2,
                2 * rotvec[0] * rotvec[1] - 2 * rotvec[2] * rotvec[2],
                2 * rotvec[0] * rotvec[2] + 2 * rotvec[1] * rotvec[2],
            ],
            [
                2 * rotvec[0] * rotvec[1] + 2 * rotvec[2] * rotvec[2],
                1 - 2 * rotvec[0] ** 2 - 2 * rotvec[2] ** 2,
                2 * rotvec[1] * rotvec[2] - 2 * rotvec[0] * rotvec[2],
            ],
            [
                2 * rotvec[0] * rotvec[2] - 2 * rotvec[1] * rotvec[2],
                2 * rotvec[1] * rotvec[2] + 2 * rotvec[0] * rotvec[2],
                1 - 2 * rotvec[0] ** 2 - 2 * rotvec[1] ** 2,
            ],
        ],
        device=atoms.device,
    )

    center = atoms_rotated_z.mean(dim=0)
    atoms_centered = atoms_rotated_z - center
    atoms_rotated = torch.mm(atoms_centered, rotation_matrix) + center

    return atoms_rotated


if __name__ == "__main__":

    tag_path = (
        "/home/jovyan/shared-scratch/adeesh/data/oc20_dense/oc20dense_tags.pkl"
    )
    with open(os.path.join(tag_path), "rb") as h:
        tags_map = pickle.load(h)
    ads_idx = tags_map["2_2861_5"] == 2

    traj = ase.io.read(
        "/home/jovyan/shared-scratch/adeesh/denoising/overfit_pbccorr/overfit-xy_std0.01-10_numstep50x10_lr1.e-4_sample1/2_2861_5.traj",
        ":",
    )
    init_system = traj[0]
    ads_positions = init_system.positions[ads_idx]
