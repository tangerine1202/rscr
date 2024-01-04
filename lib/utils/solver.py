import cv2
import numpy as np
import torch

import logging


def procrustes(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (B, N, 3) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (B, N, 3) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation (B, 3, 3)
        -    t: optimal translation  (B, 3, 1)
    Based on: https://gist.github.com/bougui505/e392a371f5bab095a3673ea6f4976cc8
    """
    assert len(A.shape) == len(B.shape) == 3, 'three dimensions are required'
    assert A.shape[0] == B.shape[0], 'batch size must match'
    assert A.shape[1] == B.shape[1], 'number of correspondences must match'
    assert A.shape[2] == B.shape[2], 'number of spatial dimensions must be 3'

    a_mean = A.mean(axis=1, keepdim=True)
    b_mean = B.mean(axis=1, keepdim=True)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.transpose(1, 2) @ B_c

    # FIXME: torch.svd on GPU has bugs that cause Segmentation Fault.
    #        Move to CPU as workaround for now.
    # U, S, V = torch.svd(H)
    U, S, V = torch.svd(H.cpu())
    U = U.to(A.device)
    V = V.to(A.device)

    # Fixes orientation such that Det(R) = + 1
    Z = torch.eye(3).unsqueeze(0).repeat(A.shape[0], 1, 1).to(A.device)
    Z[:, -1, -1] = torch.sign(torch.linalg.det(U @ V.transpose(1, 2)))
    # Rotation matrix
    R = V @ Z @ U.transpose(1, 2)
    # Translation vector
    t = b_mean - a_mean @ R.transpose(1, 2)
    return R, t


# TODO: update according to Map-free impl,
# see https://github.com/nianticlabs/map-free-reloc/blob/904f1c479dae497d87fb24d449ad5ac6869c7659/lib/models/matching/pose_solver.py#L175
def pnp(pts_B3, pts_B2, K_B33, flags):
    """
    See: https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
    Calculate pose from 2D-3D correspondences.
        Args:
        -    pts_B3: Torch tensor of shape (B, 3, ...)
        -    pts_B2: Torch tensor of shape (B, 2, ...)
        -    K_B33: Torch tensor of shape (B, 3, 3)
        -   flags: PnP flags (e.g. cv2.SOLVEPNP_EPNP)
        Returns:
        -    R: optimal rotation (B, 3, 3)
        -    t: optimal translation  (B, 3, 1)
    Based on: https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d
    """
    # assert len(pts_B3.shape) == len(pts_B2.shape) == 3, 'at least three dimensions are required'
    assert pts_B3.shape[0] == pts_B2.shape[0], 'batch size must match'
    assert pts_B3.shape[1] == 3, 'number of 3D points dimensions must be 3'
    assert pts_B2.shape[1] == 2, 'number of 2D points dimensions must be 2'
    assert pts_B3.shape[2:] == pts_B2.shape[2:], 'number of correspondences must match'

    B = pts_B3.shape[0]
    pts_B3N = pts_B3.view(B, 3, -1)  # (B, 3, ...) -> (B, 3, N)
    pts_B2N = pts_B2.view(B, 2, -1)  # (B, 2, ...) -> (B, 2, N)

    R_B33 = torch.empty(B, 3, 3).to(pts_B3.device)
    t_B13 = torch.empty(B, 1, 3).to(pts_B3.device)

    for idx in range(B):
        pts_3d = pts_B3N[idx].cpu().numpy().T  # (N, 3)
        pts_2d = pts_B2N[idx].cpu().numpy().T  # (N, 2)
        K = K_B33[idx].cpu().numpy()  # (3, 3)
        R, t = safe_solve_pnp(pts_3d, pts_2d, K, flags=flags)
        R_B33[idx] = torch.from_numpy(R)
        t_B13[idx] = torch.from_numpy(t).view(1, 3)
    return R_B33, t_B13


def safe_solve_pnp(pts_3d, pts2d, K, flags):
    assert len(pts_3d.shape) == len(pts2d.shape) == 2, 'two dimensions are required'
    assert pts_3d.shape[0] == pts2d.shape[0], 'number of correspondences must match'
    assert pts_3d.shape[1] == 3, 'number of spatial dimensions must be 3'
    assert pts2d.shape[1] == 2, 'number of spatial dimensions must be 2'

    camera_matrix = K
    dist_coeffs = None
    retval, rvec, tvec = cv2.solvePnP(pts_3d, pts2d, camera_matrix, dist_coeffs, flags=flags)

    if retval:
        R, _ = cv2.Rodrigues(rvec)
        t = tvec
    else:
        # Handle the case where solvePnP fails
        logging.warning('solvePnP failed, using identity pose')
        R = np.eye(3)
        t = np.zeros(3)

    return R, t
