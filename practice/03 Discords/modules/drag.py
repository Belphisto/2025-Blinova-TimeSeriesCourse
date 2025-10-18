import numpy as np
from stumpy import core, config, stump


def _get_chunks_ranges(a, shift=None):
    """
    This function takes an array that contains only integer numbers in ascending order, and return the
    `(inclusive) start index` and `(exclusive) stop index + shift` for each continuous segment of array.

    Parameters
    --------
    a : numpy.ndarray
        1-dim array that contains integer numbers in ascending order.

    shift : int, default None
        an integer number by which the stop index of each segement should be shifted. If None, no shift will be applied.

    Returns
    -------
    out : numpy.ndarray
        a 2-dim numpy array. The first column is the (inclusive) start index of each segment. The second column is the
        (exclusive) stop index shifted by `shift` units.
    """
    repeats = np.full(len(a), 2)
    diff_is_one = np.diff(a) == 1
    repeats[1:] -= diff_is_one
    repeats[:-1] -= diff_is_one
    out = np.repeat(a, repeats).reshape(-1, 2)
    out[:, 1] += 1

    if shift is not None:
        out[:, 1] += shift

    return out


def find_candidates(T, m, M_T, Σ_T, r, init_cands=None, right=True, finite=False):
    """
    For a time series T, this function finds a set of candidates whose distance to all of their right (left) neighbors
    is at least `r` when parameter `right` is TRUE (FALSE). If there is no such candidate, all elements of is_cands
    becomes False.

    Parameters
    ---------
    T : numpy.ndarray
        The time series or sequence from which the candidates are being selected.

    m : int
        Window size

    M_T : ndarray
        Sliding mean of `T`

    Σ_T : ndarray
        Sliding standard deviation of `T`

    r : float
        An estimate of discord_dist. The selected candidates retuned by this function have distances of at least `r`
        to all of their right(left) neighbors when input `right` is set to True(False).

        Choosing different values for `r`can affect the performance of the algorithm
        (see Fig. 5 of the paper). For instance, choosing a very large value for `r` may result in no candidates
        while choosing a very small value may result in a lot of candidates.
        (note: `r` is passed to this private function when it is called inside the top-level function `_discords`).

    init_cands : numpy.ndarray, default None
        is a 1-dim boolean array, with shape=(k,) where `k` is the total number of subsquences in the time series.
        `init_cands[i]` is True if the subsequence with start index `i` is considered as one of the
        prospective candidates.

    right : bool, default True
        If True (False), candidates returned by the function are guaranteed to have at least the distance of `r`
        to all of their 'right`('left') neighbors.

    finite : bool, default False
        If True, subsequence with infinite values will not be considered as candidates.

    Returns
    --------
    is_cands : numpy.ndarray
        is a 1-dim boolean array, with shape=(k,) where `k` is the total number of subsquences in the time series.
        `is_cands[i]` is True if the subsequence with start index `i` has minimum distance of `r` to all of its
        right (left) neighbors when right is True (False).

    NOTE
    -------
    Unlike the MERLIN paper where the exclusion zone is m, the default exclusion zone considered here
    is the STUMPY default config m/4. This can be changed by setting config.STUMPY_EXCL_ZONE_DENOM.
    """
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))

    k = T.shape[0] - m + 1

    is_cands = np.ones(k, dtype=bool)
    if init_cands is not None:
        is_cands[:] = init_cands

    T_subseq_isfinite = np.isfinite(M_T)
    if not finite:
        T_subseq_isfinite[:] = True
    is_cands[~T_subseq_isfinite] = False

    for i in np.flatnonzero(T_subseq_isfinite):
        if np.all(is_cands == False):
            break

        cands_idx = np.flatnonzero(is_cands)

        if right:
            non_trivial_cands_idx = cands_idx[cands_idx < max(0, i - excl_zone)]
        else:
            non_trivial_cands_idx = cands_idx[cands_idx > i + excl_zone]

        if len(non_trivial_cands_idx) > 0:
            cand_idx_chunks = _get_chunks_ranges(non_trivial_cands_idx, shift=m - 1)

            for start, stop in cand_idx_chunks:
                # Use stump to compute distances for this chunk
                # This is a workaround for the _mass function signature issues
                chunk_data = T[start:stop]
                if len(chunk_data) >= m:
                    # Compute matrix profile for this chunk
                    mp = stump(T[i:i + m], chunk_data, ignore_trivial=False)
                    if len(mp) > 0:
                        D = mp[:, 0]  # Get the distances

                        mask = np.flatnonzero(D < r)
                        chunk_cand_indices = np.arange(start, stop - m + 1)
                        if len(mask) > 0:
                            affected_indices = chunk_cand_indices[mask]
                            is_cands[affected_indices] = False
                            is_cands[i] = False

    return is_cands


def refine_candidates(T, m, M_T, Σ_T, is_cands):
    """
    For a time series `T`, this function searches the candidates (i.e. subsequences indicated by `is_cands`) and
    return candidates discords in descending order according to their distance to their nearest neighbor.
    After finding the top-discord among candidates, the discord subsequence and its trivial neighbors will be excluded
    from candidates before finding the next top-discord.

    Parameters
    ---------
    T : numpy.ndarray
        The time series or sequence from which the top discord (out of selected candidates) is discovered.

    m : int
        Window size

    M_T : numpy.ndarray
        Sliding mean of `T`

    Σ_T : numpy.ndarray
        Sliding standard deviation of `T`

    is_cands : numpy.ndarray
        is a 1-dim boolean array, with shape=(k,) where `k` is the total number of subsquences in the time series.
        when `is_cands[i]` is True, a subsequence with start index `i` is a discord candidate.

    Returns
    ---------
    out : numpy.ndarray
        is a 2-dim array with three columns. The first column is indices of discords, sorted according to their
        corresponding distances to their nearest neighbor, provided in the second column.
        The third column is the indices of the discords' nearest neighbor.
    """
    excl_zone = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    k = T.shape[0] - m + 1

    P = np.full(k, np.NINF, dtype=np.float64)  # matrix profile
    I = np.full(k, -1, dtype=np.int64)  # index of Nearest Neighbor

    for idx in np.flatnonzero(is_cands):
        # Use stump to find nearest neighbor for this candidate
        mp = stump(T[idx:idx + m], T, ignore_trivial=False)
        if len(mp) > 0:
            # Apply exclusion zone
            D = mp[:, 0].copy()
            core.apply_exclusion_zone(D, idx, excl_zone, val=np.inf)

            nn_idx = np.argmin(D)
            if D[nn_idx] == np.inf:
                nn_idx = -1
            P[idx] = D[nn_idx]
            I[idx] = nn_idx

    discords_idx = []
    discords_dist = []
    discords_nn_idx = []
    while np.any(P >= 0):
        idx = np.argmax(P)
        discords_idx.append(idx)
        discords_dist.append(P[idx])
        discords_nn_idx.append(I[idx])
        core.apply_exclusion_zone(P, idx, excl_zone, np.NINF)

    return discords_idx, discords_dist, discords_nn_idx


def DRAG(data, m, r, include=None):
    if include is None:
        include = np.ones(len(data) - m + 1, dtype=bool)
    else:
        include = include[:len(data) - m + 1]

    # Get all returned values from preprocess
    preprocess_result = core.preprocess(data, m)

    # Handle different return formats from different stumpy versions
    if len(preprocess_result) == 3:
        T, M_T, Σ_T = preprocess_result
    else:
        # For newer versions that might return more values
        T = preprocess_result[0]
        M_T = preprocess_result[1]
        Σ_T = preprocess_result[2]

    is_cands = find_candidates(T, m, M_T, Σ_T, r, init_cands=include, right=True)
    cand_index = np.flatnonzero(is_cands)
    is_cands = find_candidates(T, m, M_T, Σ_T, r, init_cands=is_cands, right=False)
    cands = np.flatnonzero(is_cands)
    return refine_candidates(T, m, M_T, Σ_T, is_cands)