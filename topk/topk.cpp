#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <tuple>
#include "topk.h"

using namespace std;

void printTopK(CertainFrame *topk, int num)
{
    cout << "num_topk: " << num << endl;
    for (int i = 0; i < num; i++)
    {
        cout << " (" << topk[i].f << "," << topk[i].sf << ") ";
    }
    cout << endl;
}

void printSlab(Slab *slabs, int num)
{
    cout << "num_slabs: " << num << endl;
    for (int i = 0; i < num; i++)
    {
        cout << " ((" << slabs[i].s << "," << slabs[i].e << "), " << slabs[i].dominant->sf << ") ";
    }
    cout << endl;
}

vector<int> TopKGAPOp::get_result(const vector<string>& path_list) {
    vector<int> scores;
    for (auto path: path_list)
        scores.push_back(model.infer(path));
    return scores;
}

TopKGAPOp::TopKGAPOp(int k, float confidence, int gap, CDFTable* table)
    : k(k), confidence(confidence), gap(gap), table(table) {
    lam_i = 0;
    eta_i = 0;
    cur_num_slabs = 0;
    cur_num_topk = 0;
    cur_slabs = new Slab[2*k+1]; cur_topk = new CertainFrame[k];
    bound_list = new BoundInfo[table->num_frames];
    certain_list = new CertainList(table->num_frames, table->num_scores);
    for (int i = 0; i < table->num_frames; i++)
    {
        bound_list[i].f = i;
        bound_list[i].L = 1; //maximum value
    }
}

TopKGAPOp::~TopKGAPOp() {
    delete [] cur_slabs;
    delete [] cur_topk;
    delete [] bound_list;
    delete certain_list;
}

double TopKGAPOp::computeL(int index) {
    int start = max(index-gap, 0);
    int end = min(index+gap, table->num_frames-1);
    double denominator = (table->H(end, eta_i) - table->H(start-1, eta_i)) * 2;
    denominator -= certain_list->H(end, eta_i) - certain_list->H(start-1, eta_i);
    denominator += table->F(index, lam_i);
    return exp(-denominator) - exp(table->F(index, lam_i) - denominator);
}

double TopKGAPOp::computeE(int f, double pi) {
    int perm = 0;
    int rank = 0;
    double expectation = pi;

    // get perm
    for (int i = 0; i < cur_num_slabs; i++)
    {
        if (cur_slabs[i].s <= f && cur_slabs[i].e >= f)
        {
            perm = cur_slabs[i].perm;
            break;
        }
    }

    // get perm rank
    for (rank = cur_num_topk; rank > 0; rank--)
    {
        if (cur_topk[rank-1].sf > perm)
            break;
    }

    for (int i = rank; i > 0; i--)
    {
        CertainFrame v_topk[k];
        CertainFrame v_frame = {f, cur_topk[i].sf + 1};
        int          v_num_topk = certain_list->topk(k, gap, v_topk, &v_frame);
        Slab         v_slabs[2 * k];
        int          v_num_slabs = getSlabs(v_topk, v_num_topk, gap, 0, table->num_frames, v_slabs);
        for (int s = cur_topk[i].sf + 1; s <= cur_topk[i-1].sf; s++)
        {
            v_frame.sf = s;
            v_topk[i].sf = s;
            printTopK(v_topk, v_num_topk);
            printSlab(v_slabs, v_num_slabs);
            double p = table->P(f, s) * topkProb(v_topk, v_num_topk, v_slabs, v_num_slabs, &v_frame, true);
            expectation += p;
        }
    }

    if (cur_topk[0].sf < table->num_scores - 1)
    {
        CertainFrame v_topk[k];
        CertainFrame v_frame = {f, cur_topk[0].sf + 1};
        int          v_num_topk = certain_list->topk(k, gap, v_topk, &v_frame);
        Slab         v_slabs[2 * k];
        int          v_num_slabs = getSlabs(v_topk, v_num_topk, gap, 0, table->num_frames, v_slabs);
        for (int s = cur_topk[0].sf + 1; s < table->num_scores; s++)
        {
            v_frame.sf = s;
            v_topk[0].sf = s;
            printTopK(v_topk, v_num_topk);
            printSlab(v_slabs, v_num_slabs);
            double p = table->P(f, s) * topkProb(v_topk, v_num_topk, v_slabs, v_num_slabs, &v_frame, true);
            expectation += p;
        }
    }

    return expectation;
}

double TopKGAPOp::uncertainProb(Slab* slabs, int num_slabs, bool use_dominant) {
    double numerator = 0;
    double denominator = 0;

    // compute for all parts

    if (use_dominant)
    {
        for (int i = 0; i < num_slabs; i++) {
            numerator += table->H(slabs[i].e, slabs[i].dominant->sf);
            numerator -= table->H(slabs[i].s-1, slabs[i].dominant->sf);
        }
        for (int i = 0; i < num_slabs; i++)
        {
            denominator += certain_list->H(slabs[i].e, slabs[i].dominant->sf) - certain_list->H(slabs[i].s-1, slabs[i].dominant->sf);
        }
    }
    else
    {
        for (int i = 0; i < num_slabs; i++) {
            numerator += table->H(slabs[i].e, slabs[i].perm);
            numerator -= table->H(slabs[i].s-1, slabs[i].perm);
        }
        for (int i = 0; i < num_slabs; i++)
        {
            denominator += certain_list->H(slabs[i].e, slabs[i].perm) - certain_list->H(slabs[i].s-1, slabs[i].perm);
        }
    }
    

    return numerator - denominator;
}

double TopKGAPOp::topkProb(CertainFrame *topk, int num_topk, Slab *slabs, int num_slabs, CertainFrame *virtual_frame, bool use_dominant) {
    if (num_topk < k)
        return 0;

    double pi = uncertainProb(slabs, num_slabs, use_dominant);
    if (virtual_frame)
    {
        for (int i = 0; i < num_slabs; i++)
        {
            if (slabs[i].s <= virtual_frame->f && slabs[i].e >= virtual_frame->f)
            {
                if (use_dominant)
                    pi -= table->F(virtual_frame->f, slabs[i].dominant->sf);
                else
                    pi -= table->F(virtual_frame->f, slabs[i].perm);
                break;
            }
        }
    }
    return exp(pi);
}

void TopKGAPOp::updateOrder(int lam, int eta) {
    lam_i = lam;
    eta_i = eta;

    for (int i = 0; i < table->num_frames; i++)
    {
        bound_list[i].L = computeL(i);
    }
    sort(bound_list, bound_list + table->num_frames,
        [](const BoundInfo &a, const BoundInfo &b) {return a.L > b.L;});
}

vector<int> TopKGAPOp::selectIndex(double pi, int select_num, int eta) {
    vector<int> indices;
    // in ascending order
    deque<tuple<double, int> > eq;

    for (int i = 0; i < table->num_frames; i++) {
        int idx = bound_list[i].f;
        if (certain_list->is_certain(idx))
            continue;

        double gam = exp(table->H(table->num_frames, eta_i) - certain_list->H(table->num_frames, eta_i));
        double ui = pi + gam * bound_list->L;
        if (int(eq.size()) >= select_num && ui <= get<0>(eq.back()))
            break;

        tuple<double, int> ei = make_tuple(computeE(idx, pi), idx);
        auto it = lower_bound(eq.begin(), eq.end(), ei);
        eq.insert(it, ei);
        if (int(eq.size()) > select_num)
            eq.pop_front();
    }

    while (!eq.empty()) {
        indices.push_back(get<1>(eq.back()));
        eq.pop_back();
    }
    return indices;
}

int TopKGAPOp::forward(CertainFrame* topk) {

    //todo: add initial certain set

    int select_num = 64;

    int niter = 0;
    while (true) {
        cur_num_topk = certain_list->topk(k, gap, cur_topk);
        if (niter % 10 == 0 || (cur_num_topk == k && cur_topk[k-1].sf < lam_i))
        {
            lam_i = cur_topk[k-1].sf;
            eta_i = cur_topk[0].sf;
            updateOrder(lam_i, eta_i);
        }
        double pi = topkProb(cur_topk, cur_num_topk, cur_slabs, cur_num_slabs);

        // stop cleaning
        if (cur_num_topk >= k && pi >= confidence)
            break;

        vector<int> idx_list = selectIndex(pi, select_num, eta_i);
        vector<string> path_list(idx_list.size());
        for (auto idx: idx_list)
            path_list.push_back(table->meta[idx].image);
        vector<int> scores = get_result(path_list);

        for (size_t i = 0; i < idx_list.size(); i++) {
            CertainFrame new_certain = {idx_list[i], scores[i]};
            certain_list->insert(new_certain, table->probs(idx_list[i]));
        }
        niter++;
    }
    topk = cur_topk;
    return cur_num_topk;
}



int main()
{
    CDFTable table;
    table.load("test_table.json");
    table.precomputeH();

    cout << "=======test basic Topk prob======" << endl;
    TopKGAPOp op(2, 0.8, 1, &table);
    op.certain_list->insert(CertainFrame{5, 2}, table.probs(5));
    op.certain_list->insert(CertainFrame{7, 3}, table.probs(5));

    CertainFrame topk[2];
    int num_topk = op.certain_list->topk(2, 1, topk);
    printTopK(topk, num_topk);

    Slab slabs[2 * 2];
    int num_slabs = getSlabs(topk, num_topk, 1, 0, table.num_frames, slabs);
    printSlab(slabs, num_slabs);
    double pi = op.topkProb(topk, num_topk, slabs, num_slabs); 
    cout << "topk_prob: " << pi << endl;

    cout << "=======test vf TopK prob========" << endl;
    CertainFrame vf = {4, 3};
    num_topk = op.certain_list->topk(2, 1, topk, &vf);
    printTopK(topk, num_topk);

    num_slabs = getSlabs(topk, num_topk, 1, 0, table.num_frames, slabs);
    printSlab(slabs, num_slabs);
    pi = op.topkProb(topk, num_topk, slabs, num_slabs, &vf); 
    cout << "topk_prob: " << pi << endl;

    cout << "=======test E=========" << endl;
    op.cur_num_topk = op.certain_list->topk(2, 1, op.cur_topk);
    printTopK(op.cur_topk, op.cur_num_topk);
    op.cur_num_slabs = getSlabs(op.cur_topk, op.cur_num_topk, 1, 0, table.num_frames, op.cur_slabs);
    printSlab(op.cur_slabs, op.cur_num_slabs);
    pi = op.topkProb(op.cur_topk, op.cur_num_topk, op.cur_slabs, op.cur_num_slabs); 
    cout << "topk_prob: " << pi << endl;
    double E = op.computeE(4, pi);
    cout << "E[2]: " << E << endl; 
    cout << endl;
    double ref = 0;

    for (int i = 0; i < 5; i++)
    {
        CertainFrame vf = {4, i};
        num_topk = op.certain_list->topk(2, 1, topk, &vf);
        num_slabs = getSlabs(topk, num_topk, 1, 0, table.num_frames, slabs);
        cout << "vf: (" << vf.f << "," << vf.sf << ")" << endl;
        printTopK(topk, num_topk);
        printSlab(slabs, num_slabs);
        double p = table.P(4, i);
        double topkp = op.topkProb(topk, num_topk, slabs, num_slabs, &vf);
        cout << "p: " << p << " topkp: " << topkp << endl;
        ref += p * topkp; 
    }
    cout << "ref: " << ref << endl;

    cout << "=======test L=======" << endl;
    op.lam_i = 2;
    op.eta_i = 3;
    double l = op.computeL(1);
    cout << "L[2]: " << l << endl;
    return 0;
}