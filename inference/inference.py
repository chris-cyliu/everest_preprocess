"infernece frames by cheap model"
import numpy as np
from inference.poibin import PoiBin


def build_uncertain_table(scores):
    """Convert detection results to uncertain table offline

    Parameters
    ----------
    scores : numpy ndarray
        Detection scores, shape should be (N, M), N is number of frames and M is number of detected
        objects(append 0 if having insufficient objects).

    Returns
    -------
    uncertain_table : numpy ndarray
        Uncertain table for top-k, shape should be (N, M+1).
    """
    num_frames = scores.shape[0]
    max_bboxes = scores.shape[1]
    num_pw = 2 ** max_bboxes
    uncertain_table = np.zeros((num_frames, max_bboxes + 1))

    assert num_pw >= 0, "Too many possible worlds({}) exists.".format(num_pw)

    all_scores = np.stack([1 - scores, scores], axis=-1)

    # enumerate all possible worlds
    for i in range(num_pw):
        binary_str = ('{:0' + str(max_bboxes) + 'b}').format(i)
        ind = np.array([int(j) for j in binary_str])
        assert len(ind) == max_bboxes, "Length of indices({}) should equal to number of bounding " \
            "boxes({}).".format(len(ind), max_bboxes)

        sel_scores = all_scores[:, range(max_bboxes), ind[range(max_bboxes)]]
        pw_prob = np.multiply.reduce(sel_scores, axis=1)
        num_objects = np.sum(ind)
        uncertain_table[:, num_objects] += pw_prob

    return uncertain_table


def build_uncertain_table_fast(scores):
    """Convert detection results to uncertain table offline with faster way

    Parameters
    ----------
    scores : numpy ndarray
        Detection scores, shape should be (N, M), N is number of frames and M is number of detected
        objects(append 0 if having insufficient objects).

    Returns
    -------
    uncertain_table : numpy ndarray
        Uncertain table for top-k, shape should be (N, M+1).
    """
    num_frames = scores.shape[0]
    max_bboxes = scores.shape[1]
    uncertain_table = np.zeros((num_frames, max_bboxes + 1))

    # enumerate all possible worlds
    for i in range(num_frames):
        # encounter error when having too many zeros
        num_nonzeros = np.sum(scores[i] != 0)
        s = scores[i][:num_nonzeros]
        pb = PoiBin(s)
        uncertain_table[i][:num_nonzeros+1] = pb.pmf(range(num_nonzeros+1))
        # s = scores[i]
        # pb = PoiBin(s)
        # uncertain_table[i] = pb.pmf(range(max_bboxes+1))

    return uncertain_table
