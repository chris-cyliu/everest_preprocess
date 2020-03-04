#include "certain_list.h"
#include <string.h>
#include <iostream>
using namespace std;

bool _comp_certain_frame(const CertainFrame& a, const CertainFrame& b) {return a.sf < b.sf;}

CertainList::CertainList(int capacity, int num_scores) : sortedlist()
{
    this->capacity = capacity;
    this->num_scores = num_scores;
    bitmap = new char[capacity];
    memset(bitmap, 0, sizeof(*bitmap));
    h_data = new double[capacity * num_scores];
    memset(h_data, 0, sizeof(*h_data));
    frames = new int[capacity];
    size = 0;
}

int *binarySearch(int *l, int *r, int x) 
{ 
    int *m;
    while (l < r) { 
        m = l + (r - l - 1) / 2; 
        if (*m == x) 
            return m; 
        if (*m < x) 
            l = m + 1; 
        else
            r = m - 1; 
    } 
    return l; 
} 

list<CertainFrame>::iterator sortedlist_insert(list<CertainFrame> &sortedlist, const CertainFrame& frame) 
{
    list<CertainFrame>::iterator it;
    for (it = sortedlist.begin(); it->sf >= frame.sf && it != sortedlist.end(); ++it);
    return sortedlist.insert(it, frame);
}

void CertainList::insert(const CertainFrame& frame, double *probs)
{
    int f = frame.f;
    int *idx = binarySearch(frames, frames + size, f);
    int offset = idx - frames;
    memcpy(idx + 1, idx, frames + size - idx);
    *idx = f;

    for (int i = 0; i < num_scores; ++i)
    {
        memcpy(h_data + i * capacity + offset + 1, h_data + i * capacity + offset, (size - offset) * sizeof(double));
        h_data[i * capacity + offset] = offset ? h_data[i * capacity + offset - 1] : 0;
        #pragma omp simd
        for (int j = offset; j <= size; ++j)
        {
            h_data[i * capacity + j] += probs[i];
        }
    }
    bitmap[f] = 1;    
    sortedlist_insert(sortedlist, frame);
    ++size;
}

double CertainList::H(int f, int s)
{
    int offset = binarySearch(frames, frames + size, f) - frames;
    if (offset >= size || frames[offset] > f) 
        --offset;
    if (offset < 0)
        return 0;
    else 
        return h_data[s * capacity + offset];
}

bool CertainList::is_certain(int frame)
{
    return bitmap[frame];
}

int CertainList::topk(int k, int gap, CertainFrame *o_topk, CertainFrame *virtual_frame)
{
    int num = 0;
    list<CertainFrame>::iterator virtual_it = sortedlist.end();
    if (virtual_frame)
    {
        virtual_it = sortedlist_insert(sortedlist, *virtual_frame);
    }
    for (list<CertainFrame>::iterator it = sortedlist.begin(); it != sortedlist.end(); ++it)
    {
        bool masked = false;
        for (int i = 0; i < num; i++)
        {
            if (o_topk[i].f - gap <= it->f && o_topk[i].f + gap >= it->f)
            {
                masked = true;
                break;
            }
        }
        if (!masked)
        {
            o_topk[num++] = *it;
            if (num >= k)
                break;
        }
    }
    if (virtual_frame)
    {
        sortedlist.erase(virtual_it);
    }
    return num;
}

//int main()
//{
//    CertainList clist(50, 10);
//    double probs[3] = {0, 1, 2};
//    CertainFrame top[5];
//    
//    CertainFrame f0 = {5, 3};
//    CertainFrame f1 = {15, 4};
//    CertainFrame f2 = {16, 5};
//    CertainFrame f3 = {30, 6};
//    clist.insert(f0, probs);
//    clist.insert(f1, probs);
//    probs[0] = 1;
//    clist.insert(f2, probs);
//
//    int num = clist.topk(3, 5, top, &f3);
//
//    for (int i = 0; i < num; i++)
//    {
//        cout << top[i].f << "," << top[i].sf << " ";
//    }
//    cout << endl;
//    return 0;
//}