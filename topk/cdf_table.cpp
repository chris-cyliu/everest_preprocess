#include "cdf_table.h"
#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

CDFTable::CDFTable()
{
    meta = NULL;
    cdf_data = NULL;
    h_data = NULL;
    num_frames = 0;
    num_scores = 0;
}

CDFTable::~CDFTable()
{
    delete[] meta;
    delete[] cdf_data;
    delete[] h_data;
}

void CDFTable::load(string path)
{
    ifstream file(path);
    if (!file.good())
    {
        cout << "cannot open CDF table file!" << endl;
        return;
    }
    file.seekg(0, file.end);
    int length = file.tellg();
    file.seekg(0, file.beg);

    char *text = new char[length + 1];
    file.read(text, length);
    text[length] = '\0';
    file.close();

    cJSON *data_json = cJSON_Parse(text);
    if (data_json == NULL)
    {
        cout << "cannot parse CDF table file" << endl;
        return;
    }

    this->num_frames = cJSON_GetArraySize(data_json);
    this->meta = new CDFTableMeta[this->num_frames];
    for (int i = 0; i < this->num_frames; ++i)
    {
        cJSON *elem = cJSON_GetArrayItem(data_json, i);
        cJSON *timestamp_json = cJSON_GetObjectItem(elem, "timestamp");
        cJSON *image_json = cJSON_GetObjectItem(elem, "image_path");
        cJSON *prob_json = cJSON_GetObjectItem(elem, "prob");
        if (!elem || !timestamp_json || !image_json || !prob_json)
        {
            cout << "cannot parse CDF table file " <<  i << endl;
            return;
        }
        this->meta[i].timestamp = timestamp_json->valueint;
        this->meta[i].image = string(image_json->valuestring);
        if (this->cdf_data == NULL)
        {
            this->num_scores = cJSON_GetArraySize(prob_json);
            this->cdf_data = new double[this->num_frames * this->num_scores];
        }
        for (int j = 0; j < this->num_scores; ++j)
        {
            cJSON *prob_s_json = cJSON_GetArrayItem(prob_json, j);
            this->cdf_data[i * this->num_scores + j] = prob_s_json->valuedouble;
        }
    }

    // take log of probabilities
    #pragma omp parallel for simd
    for (int i = 0; i < this->num_frames * this->num_scores; ++i)
    {
        this->cdf_data[i] = log(this->cdf_data[i]);
    }
    cJSON_Delete(data_json);
}

void CDFTable::precomputeH()
{
    this->h_data = new double[this->num_frames * this->num_scores];
    #pragma omp parallel for
    for (int i = 0; i < this->num_scores; ++i)
    {
        double prefix = 0;
        for (int j = 0; j < this->num_frames; ++j)
        {
            prefix += this->cdf_data[j * this->num_scores + i];
            this->h_data[i * this->num_frames + j] = prefix;
        }
    }
}

double CDFTable::F(int f, int s)
{
    return this->cdf_data[f * this->num_scores + s];
}

double CDFTable::H(int f, int s)
{
    if (f < 0) 
        return 0;
    if (f >= num_frames)
        f = num_frames - 1;
    return this->h_data[s * this->num_frames + f];
}

double CDFTable::P(int f, int s)
{
    if (s == 0)
        return exp(this->F(f, 0));
    else
        return exp(this->F(f, s)) - exp(this->F(f, s-1));
    
}

double *CDFTable::probs(int f)
{
    return &cdf_data[f * this->num_scores];
}
