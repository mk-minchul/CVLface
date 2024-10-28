import torch
import os
import math


@torch.no_grad()
def piecewise_index(relative_position, alpha, beta, gamma, dtype):
    """piecewise index function defined in Eq. (18) in our paper.

    Parameters
    ----------
    relative_position: torch.Tensor, dtype: long or float
        The shape of `relative_position` is (L, L).
    alpha, beta, gamma: float
        The coefficients of piecewise index function.

    Returns
    -------
    idx: torch.Tensor, dtype: long
        A tensor indexing relative distances to corresponding encodings.
        `idx` is a long tensor, whose shape is (L, L) and each element is in [-beta, beta].
    """
    rp_abs = relative_position.abs()
    mask = rp_abs <= alpha
    not_mask = ~mask
    rp_out = relative_position[not_mask]
    rp_abs_out = rp_abs[not_mask]
    y_out = (torch.sign(rp_out) * (alpha +
                                   torch.log(rp_abs_out / alpha) /
                                   math.log(gamma / alpha) *
                                   (beta - alpha)).round().clip(max=beta)).to(dtype)

    idx = relative_position.clone()
    if idx.dtype in [torch.float32, torch.float64]:
        # round(x) when |x| <= alpha
        idx = idx.round().to(dtype)

    # assign the value when |x| > alpha
    idx[not_mask] = y_out
    return idx


@torch.no_grad()
def _rp_2d_euclidean(diff, **kwargs):
    """2D RPE with Euclidean method.

    Parameters
    ----------
    diff: torch.Tensor
        The shape of `diff` is (L, L, 2),
        where L is the sequence length,
        and 2 represents a 2D offset (row_offset, col_offset).

    Returns
    -------
    index: torch.Tensor, dtype: long
        index to corresponding encodings.
        The shape of `index` is (L, L),
        where L is the sequence length.
    """
    dis = diff.square().sum(2).float().sqrt().round()
    return piecewise_index(dis, **kwargs)


@torch.no_grad()
def _rp_2d_quant(diff, **kwargs):
    """2D RPE with Quantization method.

    Parameters
    ----------
    diff: torch.Tensor
        The shape of `diff` is (L, L, 2),
        where L is the sequence length,
        and 2 represents a 2D offset (row_offset, col_offset).

    Returns
    -------
    index: torch.Tensor, dtype: long
        index to corresponding encodings.
        The shape of `index` is (L, L),
        where L is the sequence length.
    """

    dis = diff.square().sum(2)
    return piecewise_index(dis, **kwargs)


@torch.no_grad()
def _rp_2d_product(diff, **kwargs):
    """2D RPE with Product method.

    Parameters
    ----------
    diff: torch.Tensor
        The shape of `diff` is (L, L, 2),
        where L is the sequence length,
        and 2 represents a 2D offset (row_offset, col_offset).

    Returns
    -------
    index: torch.Tensor, dtype: long
        index to corresponding encodings.
        The shape of `index` is (L, L),
        where L is the sequence length.
    """
    # convert beta to an integer since beta is a float number.
    beta_int = int(kwargs['beta'])
    S = 2 * beta_int + 1
    # the output of piecewise index function is in [-beta_int, beta_int]
    r = piecewise_index(diff[:, :, 0], **kwargs) + \
        beta_int  # [0, 2 * beta_int]
    c = piecewise_index(diff[:, :, 1], **kwargs) + \
        beta_int  # [0, 2 * beta_int]

    pid = r * S + c

    return pid


@torch.no_grad()
def _rp_2d_cross_rows(diff, **kwargs):
    """2D RPE with Cross for rows.

    Parameters
    ----------
    diff: torch.Tensor
        The shape of `diff` is (L, L, 2),
        where L is the sequence length,
        and 2 represents a 2D offset (row_offset, col_offset).

    Returns
    -------
    index: torch.Tensor, dtype: long
        index to corresponding encodings.
        The shape of `index` is (L, L),
        where L is the sequence length.
    """
    dis = diff[:, :, 0]
    return piecewise_index(dis, **kwargs)


@torch.no_grad()
def _rp_2d_cross_cols(diff, **kwargs):
    """2D RPE with Cross for columns.

    Parameters
    ----------
    diff: torch.Tensor
        The shape of `diff` is (L, L, 2),
        where L is the sequence length,
        and 2 represents a 2D offset (row_offset, col_offset).

    Returns
    -------
    index: torch.Tensor, dtype: long
        index to corresponding encodings.
        The shape of `index` is (L, L),
        where L is the sequence length.
    """

    dis = diff[:, :, 1]
    return piecewise_index(dis, **kwargs)

