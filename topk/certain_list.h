#include "slab.h"
#include <list>

bool _comp_certain_frame(const CertainFrame& a, const CertainFrame& b);
class CertainList
{
public:
    char     *bitmap;
    int      *frames;
    double   *h_data;
    int      size;
    int      capacity;
    int      num_scores;
    std::list<CertainFrame> sortedlist;

    CertainList(int capacity, int num_scores);
    void insert(const CertainFrame& frame, double *probs);
    double H(int f, int s);
    bool is_certain(int frame);
    int topk(int k, int gap, CertainFrame *o_topk, CertainFrame *virtual_frame=nullptr);
};

