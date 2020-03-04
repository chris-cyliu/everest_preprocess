#include <algorithm>
#include <iostream>
#include "slab.h"

bool _comp_endpoints(const EndPoint& a, const EndPoint& b) {return a.p < b.p;}

int getSlabs(CertainFrame topk[], int k, int gap, int left_bound, int right_bound, Slab *o_slabs)
{
    EndPoint *ends      = new EndPoint[2 * k];
    double   cur_perm   = topk[k-1].sf;
    CertainFrame *cur_dominant = &topk[k-1];
    int      cur_start  = left_bound;
    CertainFrame **queue = new CertainFrame*[2 * k];
    int      queue_head = 0;
    int      queue_tail = -1;
    int      o_idx = 0;

    for (int i = 0; i < k; ++i)
    {
        ends[i * 2].p = std::max(topk[i].f - gap, left_bound);
        ends[i * 2].is_start = true;
        ends[i * 2].rank = i;
        ends[i * 2 + 1].p = std::min(topk[i].f + gap, right_bound-1);
        ends[i * 2 + 1].is_start = false;
        ends[i * 2 + 1].rank = i;
    }
    std::sort(ends, ends + 2 * k, _comp_endpoints);
    for (int i = 0; i < 2 * k; ++i)
    {
        EndPoint &t = ends[i];
        if (t.is_start)
        {
            while(queue_tail >= queue_head && queue[queue_tail]->sf <= topk[t.rank].sf)
                --queue_tail;
            queue[++queue_tail] = &topk[t.rank];
            if (topk[t.rank].sf > cur_perm)
            {
                o_slabs[o_idx].s = cur_start;
                o_slabs[o_idx].e = t.p - 1;
                o_slabs[o_idx].perm = cur_perm;
                o_slabs[o_idx].dominant = cur_dominant;
                ++o_idx;
                cur_perm = topk[t.rank].sf;
                cur_dominant = &topk[t.rank];
                cur_start = t.p;
            }
        }
        else if (!t.is_start && queue[queue_head]->f == topk[t.rank].f)
        {
            ++queue_head;
            double new_perm = queue_tail < queue_head ? topk[k-1].sf : queue[queue_head]->sf;
            CertainFrame *new_dominante = queue_tail < queue_head ? &topk[k-1] : queue[queue_head];
            if (new_perm != cur_perm)
            {
                o_slabs[o_idx].s = cur_start;
                o_slabs[o_idx].e = t.p;
                o_slabs[o_idx].perm = cur_perm;
                o_slabs[o_idx].dominant = cur_dominant;
                ++o_idx;
                cur_start = t.p + 1;
                cur_perm = new_perm;
                cur_dominant = new_dominante;
            }
        }
    }
    if (cur_start < right_bound)
    {
        o_slabs[o_idx].s = cur_start;
        o_slabs[o_idx].e = right_bound-1;
        o_slabs[o_idx].perm = cur_perm;
        o_slabs[o_idx].dominant = cur_dominant;
        ++o_idx;
    }
    delete[] ends;
    delete[] queue;
    return o_idx;
}