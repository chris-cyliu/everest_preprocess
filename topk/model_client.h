#include <curl/curl.h>
#include <curl/easy.h>
#include <string>

class ModelClient
{
private:
    CURL *curl;
    static int receiver(char *data, size_t size, size_t nmemb, double* result);
public:
    double infer(std::string image_path);

    ModelClient();
    ~ModelClient();
};