#ifndef SLAB_H
#define SLAB_H


typedef struct
{
    int f;
    int sf;
} CertainFrame;

typedef struct
{
    int s;
    int e;
    double perm;
    CertainFrame *dominant;
} Slab;

typedef struct
{
    int p;
    bool is_start;
    int rank;
} EndPoint;


bool _comp_endpoints(const EndPoint& a, const EndPoint& b);
/*
 * Parameters:
 *     topk: top-k frames, sorted
 *     k: k
 *     gap: size of gap
 *     left_bound: start frame
 *     right_bound: end frame
 *     o_slabs: pointer to output slab array
 * Return: number of slabs
 */
int getSlabs(CertainFrame topk[], int k, int gap, int left_bound, int right_bound, Slab* o_slabs);
#endif