#include "model_client.h"
#include <iostream>
using namespace std;
ModelClient::ModelClient()
{
    curl = curl_easy_init();
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, ModelClient::receiver);
}

ModelClient::~ModelClient()
{
    curl_easy_cleanup(curl);
}

int ModelClient::receiver(char *data, size_t size, size_t nmemb, double* result)
{
    string content(data, size * nmemb);
    *result = stod(content);
    return size * nmemb;
}

double ModelClient::infer(string image_path)
{
    double result;
    CURLcode err;
    curl_easy_setopt(curl, CURLOPT_URL, ("http://127.0.0.1:5000/infer/" + image_path).c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result);
    err = curl_easy_perform(curl);
    if (err != CURLE_OK)
    {
        cout << "model server error: " << err << endl;
        return -1;
    }
    return result;
}

//int main()
//{
//    ModelClient client;
//    cout << client.infer("123") << endl;
//    cout << client.infer("123") << endl;
//    return 0;
//}