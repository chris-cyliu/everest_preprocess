#include <string>
#include "cJSON.h"

typedef struct
{
    int         timestamp;
    std::string image;
} CDFTableMeta;

class CDFTable
{
public:
    CDFTableMeta  *meta;     //one dimentional array, aligned with cdf_data
    double        *cdf_data; //two dimentional array
    double        *h_data;   //two dimentional array
    int           num_frames;
    int           num_scores;
    
    CDFTable();
    ~CDFTable();
    void load(std::string path);
    void precomputeH();
    double H(int f, int s);
    double F(int f, int s);
    double P(int f, int s);
    double *probs(int f);
};