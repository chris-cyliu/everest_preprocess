#include <iostream>
#include <string>
#include <vector>
#include <bitset>
#include <queue>
#include "slab.h"
#include "certain_list.h"
#include "cdf_table.h"
#include "model_client.h"

typedef struct
{
    int f;
    int L;
} BoundInfo;

class TopKGAPOp {
public:
    TopKGAPOp(int k, float confidence, int gap, CDFTable* table);
    ~TopKGAPOp();
    double computeL(int index);
    double computeE(int f, double pi);
    double uncertainProb(Slab* slabs, int num_slabs, bool use_dominant = false);
    double topkProb(CertainFrame *topk, int num_topk, Slab *slabs, int num_slabs, CertainFrame *virtual_frame=nullptr, bool use_dominant=false);
    void updateOrder(int lam, int eta);
    std::vector<int> selectIndex(double pi, int select_num, int eta);
    int forward(CertainFrame* topk);
    std::vector<int> get_result(const std::vector<std::string> &path_list);


    CertainList *certain_list;
    Slab *cur_slabs;
    int cur_num_slabs;
    CertainFrame *cur_topk;
    int cur_num_topk;
    int lam_i;
    int eta_i;


private:
    int k;
    float confidence;
    int gap;

    BoundInfo *bound_list;
    CDFTable *table;
    ModelClient model;
};

