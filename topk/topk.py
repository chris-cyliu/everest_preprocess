"""Top-k operator."""
import copy
import heapq
import json
import logging
from blist import sortedlist
from collections import namedtuple, deque
from recordclass import recordclass
from models.models import YOLOv3
from yolov3.utils.utils import load_classes
import numpy as np
from . import op


__all__ = ['CPRow', 'CPTable', 'TopKOp']


class CRow(object):
    """Row in count table (no uncertainty)

    Parameters
    ----------
    timestamp : int
        timestamp of frame in video.
    count : int
        count of object in video.
    """

    def __init__(self, timestamp, count):
        self.timestamp_ = timestamp
        self.count_ = count

    def __lt__(self, other):
        return self.count_ < other.count_ \
            or (self.count_ == other.count_ and self.timestamp_ < other.timestamp_)

    @property
    def t(self):
        return self.timestamp_

    @property
    def timestamp(self):
        return self.timestamp_

    @property
    def count(self):
        return self.count_


class CTable(object):
    """Count table (no uncertainty).

    Parameters
    ----------
    data : list of CRow
        list of frame's timestamp and count.
    """

    def __init__(self, data):
        self.data_ = data

    def clone(self):
        return CTable(self.data_[:])

    def __getitem__(self, key):
        return self.data_[key]

    def sort(self):
        self.data_.sort(reverse=True)

    def push_back(self, row):
        self.data_.append(row)

    def pop_back(self):
        self.data_.pop()

    def print(self):
        print("-------------------")
        for row in self.data_:
            print("%d \t %d" % (row.timestamp, row.count))
        print("-------------------")

    def __len__(self):
        return len(self.data_)


class CPRow(object):
    """Row in count probability table.

    Parameters
    ----------
    timestamp : int
        timestamp of frame in video.
    prob : list of float or numpy ndarray
        list of probabilities having a certain number of objects.
    image_path : str or None
        path to image
    """
    def __init__(self, timestamp, prob, image_path=None):
        self.timestamp_ = timestamp
        self.prob_ = prob
        self.image_path_ = image_path

    def __getitem__(self, key):
        return self.prob_[key]

    def __lt__(self, other):
        return self.timestamp_ < other.timestamp_

    @property
    def t(self):
        return self.timestamp_

    @property
    def timestamp(self):
        return self.timestamp_

    @property
    def prob(self):
        return self.prob_

    @property
    def image_path(self):
        return self.image_path_


class CPTable(object):
    """Count probability table.

    Parameters
    ----------
    data : list of CPRow
        list of frame's timestamp and count probability.
    """
    def __init__(self, data):
        self.data_ = data
        self.num_row_ = len(self.data_)
        self.num_col_ = len(self.data_[0].prob)

        # preprocess to get top confidence
        self.pdf_, self.cdf_, self.cdf_red_, self.cdf_cumprod_ = self._preprocess_v2()

    def __getitem__(self, key):
        return self.data_[key]

    def __len__(self):
        return self.num_row_

    @property
    def data(self):
        return self.data_

    @property
    def pdf(self):
        return self.pdf_

    @property
    def cdf(self):
        return self.cdf_

    @property
    def cdf_red(self):
        return self.cdf_red_

    @property
    def cdf_cumprod(self):
        return self.cdf_cumprod_

    @property
    def num_row(self):
        return self.num_row_

    @property
    def num_col(self):
        return self.num_col_

    def sort(self, key):
        return np.sort(self.data_)

    def argsort(self):
        return np.argsort(self.data_)

    def argsortfn(self, fn):
        return np.argsort(list(map(fn, range(self.num_row_))))

    def _possible_worlds_r(self, row_index, c_table, prob):
        if row_index >= self.num_row_:
            yield c_table.clone(), prob
            return

        timestamp = self.data_[row_index].timestamp
        for count, prob_row in enumerate(self.data_[row_index]):
            if prob_row == 0:
                continue
            # add a new row to table
            new_row = CRow(timestamp, count)
            c_table.push_back(new_row)
            prob *= prob_row
            yield from self._possible_worlds_r(row_index+1, c_table, prob)
            # roll back and try another possible row in next iteration
            c_table.pop_back()
            prob /= prob_row

    def iter_possible_worlds(self):
        return self._possible_worlds_r(0, CTable([]), 1)

    def _preprocess(self):
        num_row = self.num_row_
        num_col = self.num_col_

        cdf = np.empty((num_row, num_col))
        for ind in range(num_row):
            cdf[ind] = np.cumsum(self.data_[ind].prob)

        cdf_red = np.multiply.reduce(cdf)

        return cdf, cdf_red

    def _preprocess_v2(self):
        num_row = self.num_row_
        num_col = self.num_col_

        error = 1e-3

        cdf = np.empty((num_row, num_col))
        pdf = np.empty((num_row, num_col))
        for i in range(num_row):
            cdf_i = np.array(self.data_[i].prob)
            cdf_i[np.where(cdf_i > 1-error)] = 1
            # https://stackoverflow.com/questions/38666924/what-is-the-inverse-of-the-numpy-cumsum-function
            pdf_i = cdf_i.copy()
            pdf_i[1:] -= pdf_i[:-1].copy()
            cdf[i] = cdf_i
            pdf[i] = pdf_i

        cdf_red = np.multiply.reduce(cdf)
        cdf_cumprod = np.cumprod(cdf, axis=0)

        return pdf, cdf, cdf_red, cdf_cumprod


class PQueue(object):
    def __init__(self, capacity=0, initializer_list=None):
        self.capacity_ = capacity
        self.size_ = 0 if initializer_list is None else len(initializer_list)
        self.pqueue_ = [] if initializer_list is None else initializer_list

        if self.capacity_ != 0 and self.size_ > self.capacity_:
            self.size_ = self.capacity_
            self.pqueue_ = sorted(initializer_list, reverse=True)[:self.size_]

        heapq.heapify(self.pqueue_)

    def __len__(self):
        return self.size_

    def __str__(self):
        return 'PQueue({0})'.format(self.pqueue_.__str__())

    @property
    def size(self):
        return self.size_

    @property
    def capacity(self):
        return self.capacity_

    @property
    def data(self):
        return self.pqueue_

    def sorted(self):
        return heapq.nlargest(self.size_, self.pqueue_)

    def push(self, value):
        if self.capacity_ == 0 or self.size_ < self.capacity_:
            heapq.heappush(self.pqueue_, value)
            self.size_ += 1
        else:
            heapq.heappushpop(self.pqueue_, value)

    def pop(self):
        value = None
        if self.size_ > 0:
            value = heapq.heappop(self.pqueue_)
            self.size_ -= 1
        else:
            raise IndexError

        return value

    def min(self, n=1):
        value = None
        if self.size_ >= n:
            value = heapq.nsmallest(n, self.pqueue_)[-1]
        else:
            raise IndexError

        return value

    def max(self, n=1):
        value = None
        if self.size_ >= n:
            value = heapq.nlargest(n, self.pqueue_)[-1]
        else:
            raise IndexError

        return value

    def mean(self):
        return np.mean([x[0] for x in self.pqueue_])


class GSList(object):
    """"
    sorted list with gap

    Note: remember to use timestamp as key
    """
    def __init__(self, gap, k):
        # sort in descending order by value
        self.sortedlist_ = sortedlist(key=lambda x: -x.v)
        # sort in ascending order by index
        self.top_idx_ = sortedlist()

        self.size_ = len(self.sortedlist_)
        self.gap_ = gap
        self.k_ = k

    def __str__(self):
        return 'GSList({0})'.format(self.sortedlist_.__str__())

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def overlap_with_top(self, value):
        overlap = []
        for i in range(len(self.top_idx_)):
            top_i = self.top_idx_[i]
            top_i = self.sortedlist_[top_i]
            top_s = top_i.t - self.gap_
            top_e = top_i.t + self.gap_
            if top_s <= value.t and value.t <= top_e:
                overlap.append(top_i.v)
                break

        return overlap

    def push(self, value):
        """
        Time Compexity:
            O(log^2(n) + k * g * (log^2(k) + k))
            can be reduced to O(log^2(n) + k^2)
        """
        if self.size_ == 0:
            self.sortedlist_.add(value)
            self.size_ += 1
            self.top_idx_.add(0)
        else:
            # insert to list and update top index
            # O(log^2(n) + k + k * log^2(k))
            insert_idx = self.sortedlist_.bisect(value)
            i = 0
            while i < len(self.top_idx_) and self.top_idx_[i] < insert_idx:
                i += 1
            up_idx = list(self.top_idx_[i:])
            up_idx = [x+1 for x in up_idx]
            del self.top_idx_[i:]
            self.top_idx_.update(up_idx)

            # this will add value to the same place indicated by bisect
            # as there doesn't exist two same elements
            self.sortedlist_.add(value)
            self.size_ += 1

            # compute overlap with top elements
            # O(k)
            overlap_v = self.overlap_with_top(value)

            if len(overlap_v) == 0:
                # if no overlap with top index
                # delete the smallest one, and add to top
                # O(log^2(k))
                if len(self.top_idx_) == self.k_:
                    top_min_idx = self.top_idx_[-1]
                    top_min = self.sortedlist_[top_min_idx]
                    if top_min.v < value.v:
                        del self.top_idx_[-1]
                        self.top_idx_.add(insert_idx)
                else:
                    self.top_idx_.add(insert_idx)
            elif max(overlap_v) >= value.v:
                # if maximum value of overlap larger than inserted value
                # keep top indices
                # O(1)
                pass
            else:
                # if maximum value of overlap is smaller
                # 1. remove all the elements smaller than value
                # O(k * log(k))
                top_v = [self.sortedlist_[i].v for i in self.top_idx_]
                new_top_idx = [insert_idx]
                for i in range(len(self.top_idx_)):
                    if top_v[i] >= value.v:
                        new_top_idx.append(self.top_idx_[i])
                self.top_idx_ = sortedlist(new_top_idx)

                # 2. aggressively add to top and clean until no overlap
                # O(k * g * (log^2(k) + k))
                # TODO: use slabs to skip
                for i in range(self.size_):
                    if i in self.top_idx_:
                        continue
                    elif len(self.top_idx_) == self.k_:
                        break
                    else:
                        overlap = self.overlap_with_top(self.sortedlist_[i])
                        if len(overlap) == 0:
                            self.top_idx_.add(i)

    def top(self):
        top_vt = [self.sortedlist_[i] for i in self.top_idx_]
        return top_vt

    def min(self):
        min_idx = self.top_idx_[-1]
        min_val = self.sortedlist_[min_idx]
        return min_val

    def max(self):
        max_idx = self.top_idx_[0]
        max_val = self.sortedlist_[max_idx]
        return max_val

    def mean(self):
        top_vt = [self.sortedlist_[i] for i in self.top_idx_]
        return np.mean([x[0] for x in top_vt])

    def empty(self):
        return self.size_ == 0


class MaxQueue(object):
    def __init__(self):
        self.queue_ = deque()
        self.max_queue_ = deque()

    def __len__(self):
        return len(self.queue_)

    def __getitem__(self, key):
        return self.queue_[key]

    def __str__(self):
        return self.queue_.__str__().replace('deque', 'MaxQueue')

    def empty(self):
        return len(self.queue_) == 0

    def push(self, element):
        self.queue_.append(element)

        while len(self.max_queue_) > 0 and self.max_queue_[-1] < element:
            self.max_queue_.pop()
        self.max_queue_.append(element)

    def pop(self):
        element = self.queue_.popleft()
        if self.max_queue_[0] == element:
            self.max_queue_.popleft()

        return element

    def max(self):
        return self.max_queue_[0]


# (value, timestamp)
VT = namedtuple('VT', ['v', 't'])
# (timestamp, start or not, value)
TSV = namedtuple('TSV', ['t', 's', 'v'])
# (start, end, value)
SEV = recordclass('SEV', ['s', 'e', 'v'])


def point_to_slab(points, gap, left_bound, right_bound):
    if len(points) == 0:
        result = [SEV(s=left_bound, e=right_bound, v=0)]
        return result

    # points is a list of VT in descending order
    sp = [TSV(t=x.t-gap, s=True, v=x.v) for x in points]
    ep = [TSV(t=x.t+gap, s=False, v=x.v) for x in points]
    sep = sp + ep
    sep.sort()

    start = left_bound
    rk = points[-1].v
    perm_score = rk
    pending = MaxQueue()
    result = []
    for p in sep:
        if p.s:
            pending.push(p.v)
            pending_max = pending.max()
            # shrink recently added slab
            if len(result) > 0 and result[-1].e == p.t and result[-1].v < p.v:
                result[-1].e -= 1
                start = result[-1].e + 1
            if pending_max > perm_score:
                if start >= p.t:
                    perm_score = pending_max
                    continue
                end = p.t - 1
                result.append(SEV(s=start, e=end, v=perm_score))
                perm_score = pending_max
                start = end + 1
        else:
            pending.pop()
            end = min(p.t, right_bound)
            # merge slabs if score is same as recently added slab
            if len(result) > 0 and result[-1].v == perm_score:
                result[-1].e = end
            else:
                result.append(SEV(s=start, e=end, v=perm_score))
            start = end + 1
            perm_score = rk if pending.empty() else pending.max()

    if start <= right_bound:
        # merge slabs
        if len(result) > 0 and result[-1].v == perm_score:
            result[-1].e = right_bound
        else:
            result.append(SEV(s=start, e=right_bound, v=perm_score))

    return result


class TopKOp(op.CustomOp):
    """Top-k operator.

    Parameters
    ----------
    k : int
        k of the top-k operator.
    confidence : float
        confidence threshold of the top-k results.
    """
    def __init__(self, config_path, weight_path, class_path, class_name, table_path, k, confidence, batch_size):
        self.config_path_ = config_path
        self.weight_path_ = weight_path
        self.class_path_ = class_path
        self.class_name_ = class_name
        self.table_path_ = table_path
        self.k_ = k
        self.confidence_ = confidence
        self.batch_size_ = batch_size

        self.cpt = None
        self.lam0 = None
        self.mu0 = None
        self.cdf_red_c = None
        self.Ri = None
        self.arg2indices = None
        self.certain_set = None

    def _get_cptable(self, table_path):
        with open(table_path, 'r') as f:
            reader = json.load(f)
            cprow_list = []
            for row in reader:
                cprow_list.append(CPRow(
                    row['timestamp'],
                    row['prob'],
                    row['image_path']
                ))

            cptable = CPTable(cprow_list)

            return cptable

    def _init_full_model(self):
        self.model = YOLOv3(self.config_path_, self.weight_path_)
        self.classes = load_classes(self.class_path_)

    def _call_full_model(self, image_path, class_name):
        detections = self.model.predict(image_path)
        detections = detections[0]
        # filter non-target objects
        count = 0
        if detections is not None:
            detections = [obj for obj in detections if self.classes[int(obj[-1])] == class_name]
            count = len(detections)
        return count

    def L(self, idx):
        cdf = self.cpt.cdf
        lam0 = self.lam0
        mu0 = self.mu0

        ret = (1 - cdf[idx, lam0]) / max(cdf[idx, mu0], 1e-10)

        return ret

    def _cdf_red_u(self, s):
        num_col = self.cpt.num_col
        s = max(min(s, num_col-1), 0)

        numerator = self.cpt.cdf_red[s]
        denominator = self.cdf_red_c[s]

        ret = numerator / denominator if denominator else 0

        return ret

    def E(self, idx, Pi):
        num_col = self.cpt.num_col
        pdf = self.cpt.pdf[idx]
        cdf = self.cpt.cdf[idx]
        lam = self.Ri.min(1).v
        mu = self.Ri.min(2).v if self.k_ >= 2 else lam

        lam = min(lam, num_col-1)
        mu = min(mu, num_col-1)

        # if score \in [0, lam), lam' = lam
        s = Pi
        # if score \in [lam, mu), lam' = score
        s += np.sum([self._cdf_red_u(i) / cdf[i] * pdf[i] for i in range(lam, mu)])
        # if score \in [mu, max), lam' = mu
        s += self._cdf_red_u(mu) / cdf[mu] * np.sum(pdf[mu:num_col])
        return s

    def _topk_prob(self):
        lam = self.Ri.min(1).v
        Pi = self._cdf_red_u(lam)
        return Pi

    def _select_row(self, Ei, Pi):
        num_row = len(self.cpt)

        for i in range(num_row):
            ind = self.arg2indices[i]

            if ind in self.certain_set:
                continue

            mu = self.Ri.min(2).v if self.k_ >= 2 else self.Ri.min(1).v
            gam = self._cdf_red_u(mu)
            Ui = Pi + gam * self.L(ind)
            max_ei = Ei.max(1).v if Ei.size else 0
            if Ui <= max_ei and Ei.size >= Ei.capacity:
                break

            ei = self.E(ind, Pi)
            Ei.push(VT(v=ei, t=ind))

        idx_list = [vt.t for vt in Ei.data]

        return idx_list, i+1

    def _update_order(self):
        num_row = len(self.cpt)
        self.lam0 = self.Ri.min(1).v
        self.mu0 = self.Ri.min(2).v if self.k_ >= 2 else self.lam0

        cpt_arg = self.cpt.argsortfn(self.L)[::-1]
        self.arg2indices = np.arange(num_row)[cpt_arg]

    def forward(self, in_data, out_data):
        """Forward interface for top-k operator.

        Parameters
        ----------
        in_data : list
            count probability table.

        Returns
        -------
        out_data : list
            k-length list of indices for top-k frames.
            signal to call full model or not.
        """
        if in_data is not None:
            self.cpt = in_data[0]
        else:
            self.cpt = self._get_cptable(self.table_path_)

        # Ri keeps top-k scores
        self.Ri = PQueue(self.k_, [VT(v=0, t=-1)] * self.k_)
        self.cdf_red_c = np.ones(self.cpt.num_col)

        # update order to clean row in descending by L(f)
        self._update_order()

        niter = 0
        niter_select = []
        batch_size = self.batch_size_
        self.certain_set = set()

        self._init_full_model()

        while len(self.certain_set) < len(self.cpt):
            Pi = self._topk_prob()

            # finish cleaning
            if niter*batch_size >= self.k_ and Pi >= self.confidence_:
                break

            Ei = PQueue(batch_size)
            idx_list, niter_sel = self._select_row(Ei, Pi)
            niter_select.append(niter_sel)

            # call full model
            image_path_list = [self.cpt[idx].image_path for idx in idx_list]
            num_bboxes_list = []
            for image_path in image_path_list:
                num_bboxes = self._call_full_model(image_path, self.class_name_)
                num_bboxes_list.append(num_bboxes)

            for i in range(len(idx_list)):
                num_bboxes = num_bboxes_list[i]
                idx = idx_list[i]
                self.Ri.push(VT(v=num_bboxes, t=self.cpt[idx].t))
            self.cdf_red_c *= np.multiply.reduce(self.cpt.cdf[idx_list])
            self.certain_set.update(idx_list)

            niter += 1
            # update upper bound every 10 iterations
            if niter % 10 == 0:
                self._update_order()

                logging.info(
                    'Iter-{}, Ri: min {}, max {}, mean {}, Pi: {:.3f}, Ei: {:.3f}'
                    .format(niter, self.Ri.min(1).v, self.Ri.max(1).v, self.Ri.mean(), Pi, Ei.max(1).v)
                )

        logging.info(
            'Iter-{}, Ri: min {}, max {}, mean {}, Pi: {:.3f}, Ei: {:.3f}'
            .format(niter, self.Ri.min(1).v, self.Ri.max(1).v, self.Ri.mean(), Pi, Ei.max(1).v)
        )

        topk = self.Ri.sorted()
        topk_value = [x[0] for x in topk]
        topk_index = [x[1] for x in topk]

        out_data.append(topk_value)
        out_data.append(topk_index)
        out_data.append(niter)
        out_data.append(niter_select)


class TopKGAPOp(TopKOp):
    """Top-k operator with gap.

    Parameters
    ----------
    k : int
        k of the top-k operator.
    confidence : float
        confidence threshold of the top-k results.
    gap : int
        gap between frames
    """
    def __init__(self, config_path, weight_path, class_path, class_name, table_path, k, confidence, batch_size, gap):
        self.config_path_ = config_path
        self.weight_path_ = weight_path
        self.class_path_ = class_path
        self.class_name_ = class_name
        self.table_path_ = table_path
        self.k_ = k
        self.confidence_ = confidence
        self.gap_ = gap
        self.batch_size_ = batch_size

        self.cpt = None
        self.lam0 = None
        self.eta0 = None
        self.cdf_red_c = None
        self.Ri = None
        self.arg2indices = None
        self.certain_list = None
        self.slabs = None
        self.left_bound = None
        self.right_bound = None
        self.eps = np.finfo(np.float64).eps

    def L(self, idx):
        cdf = self.cpt.cdf
        lam0 = self.lam0
        eta0 = self.eta0
        num_row = len(self.cpt)

        numerator = 1 - cdf[idx, lam0]
        denominator = cdf[idx, lam0]
        start = max(idx-self.gap_, 0)
        end = min(idx+self.gap_, num_row-1)
        for i in range(start, end+1):
            if i != idx and (i not in self.certain_list):
                denominator *= cdf[i, eta0] ** 2
        ret = numerator / max(denominator, self.eps)

        return ret

    def _cdf_red_u(self, slabs, certain_list):
        cdf_cumprod = self.cpt.cdf_cumprod
        left_bound = self.left_bound

        num_slabs = len(slabs)

        numerator = 1
        denominator = 1
        # compute for D
        for i in range(num_slabs):
            start = slabs[i].s - left_bound
            end = slabs[i].e - left_bound
            perm_score = slabs[i].v
            end_cumprod = cdf_cumprod[end, perm_score]
            start_cumprod = cdf_cumprod[start-1, perm_score] if start > 0 else 1
            numerator *= end_cumprod / max(start_cumprod, self.eps)
        certain_size = len(certain_list)
        # compute for D^c
        slab_idx = 0
        for i in range(certain_size):
            idx = certain_list[i]

            while idx > slabs[slab_idx].e - left_bound:
                slab_idx += 1

            denominator *= self.cpt.cdf[idx, slabs[slab_idx].v]

        ret = numerator / max(denominator, self.eps)

        return ret

    def E(self, idx, Pi):
        num_col = self.cpt.num_col
        pdf = self.cpt.pdf[idx]
        left_bound = self.left_bound

        slabs = self.slabs
        certain_list = self.certain_list

        slab_idx = 0
        while idx > slabs[slab_idx].e - left_bound:
            slab_idx += 1
        Pf = slabs[slab_idx].v

        s = Pi
        for i in range(Pf, num_col-1):
            # add <i, t> and update top-k
            # get new slabs
            # x_f = P(D^u < P_f)
            new_Ri = self.Ri.deepcopy()
            new_Ri.push(VT(v=i, t=idx+left_bound))
            new_slabs = point_to_slab(new_Ri.top(), self.gap_, self.left_bound, self.right_bound)
            new_certain_list = copy.deepcopy(certain_list)
            new_certain_list.add(idx)

            s += pdf[i] * self._cdf_red_u(new_slabs, new_certain_list)

        return s

    def _topk_prob(self):
        Pi = self._cdf_red_u(self.slabs, self.certain_list)
        return Pi

    def _select_row(self, Ei, Pi):
        num_row = len(self.cpt)
        cdf_red = self.cpt.cdf_red
        cdf_red_c = self.cdf_red_c

        for i in range(num_row):
            ind = self.arg2indices[i]

            if ind in self.certain_list:
                continue

            eta = self.Ri.max().v if not self.Ri.empty() else 0
            gam = cdf_red[eta] / max(cdf_red_c[eta], self.eps)
            Ui = Pi + gam * self.L(ind)
            max_ei = Ei.max(1).v if Ei.size else 0
            if Ui <= max_ei and Ei.size >= Ei.capacity:
                break

            ei = self.E(ind, Pi)
            Ei.push(VT(v=ei, t=ind))

        idx_list = [vt.t for vt in Ei.data]

        return idx_list, i+1

    def _update_order(self):
        num_row = len(self.cpt)
        self.lam0 = self.Ri.min().v if not self.Ri.empty() else 0
        self.eta0 = self.Ri.max().v if not self.Ri.empty() else 0

        cpt_arg = self.cpt.argsortfn(self.L)[::-1]
        self.arg2indices = np.arange(num_row)[cpt_arg]

    def forward(self, in_data, out_data):
        """Forward interface for top-k operator.

        Parameters
        ----------
        in_data : list
            count probability table.

        Returns
        -------
        out_data : list
            k-length list of indices for top-k frames.
            signal to call full model or not.
        """
        if in_data is not None:
            self.cpt = in_data[0]
        else:
            self.cpt = self._get_cptable(self.table_path_)

        self.left_bound = self.cpt[0].t
        self.right_bound = self.cpt[-1].t
        assert self.right_bound - self.left_bound + 1 == len(self.cpt)

        # Ri keeps top-k scores
        self.Ri = GSList(self.gap_, self.k_)
        self.cdf_red_c = np.ones(self.cpt.num_col)
        self.certain_list = sortedlist()

        # update order to clean row in descending by L(f)
        self._update_order()

        niter = 0
        niter_select = []
        batch_size = self.batch_size_

        self._init_full_model()

        while len(self.certain_list) < len(self.cpt):
            self.slabs = point_to_slab(self.Ri.top(), self.gap_, self.left_bound, self.right_bound)
            Pi = self._topk_prob()

            # finish cleaning
            if niter*batch_size >= self.k_ and Pi >= self.confidence_:
                break

            Ei = PQueue(batch_size)
            idx_list, niter_sel = self._select_row(Ei, Pi)
            niter_select.append(niter_sel)

            # call full model
            image_path_list = [self.cpt[idx].image_path for idx in idx_list]
            num_bboxes_list = []
            for image_path in image_path_list:
                num_bboxes = self._call_full_model(image_path, self.class_name_)
                num_bboxes_list.append(num_bboxes)

            for i in range(len(idx_list)):
                num_bboxes = num_bboxes_list[i]
                idx = idx_list[i]
                self.Ri.push(VT(v=num_bboxes, t=self.cpt[idx].t))
            self.cdf_red_c *= np.multiply.reduce(self.cpt.cdf[idx_list])
            self.certain_list.update(idx_list)

            niter += 1
            # update upper bound every 10 iterations or when lambda changes
            if niter % 10 == 0 or self.Ri.min().v != self.lam0:
                self._update_order()

                logging.info(
                    'Iter-{}, Ri: min {}, max {}, mean {}, Pi: {:.3f}, Ei: {:.3f}'
                    .format(niter, self.Ri.min().v, self.Ri.max().v, self.Ri.mean(), Pi, Ei.max(1).v)
                )

        logging.info(
            'Iter-{}, Ri: min {}, max {}, mean {}, Pi: {:.3f}, Ei: {:.3f}'
            .format(niter, self.Ri.min().v, self.Ri.max().v, self.Ri.mean(), Pi, Ei.max(1).v)
        )

        topk = self.Ri.top()
        topk_value = [x[0] for x in topk]
        topk_index = [x[1] for x in topk]

        out_data.append(topk_value)
        out_data.append(topk_index)
        out_data.append(niter)
        out_data.append(niter_select)
