#include <iostream>
#include <set>
#include <experimental/filesystem>
#include <mxnet-cpp/MxNetCpp.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>
#include <version.h>
#include "Lenet.hpp"
#include "fileutil.hpp"
#include "Predictor.hpp"

namespace fs = std::experimental::filesystem;
using namespace mxnet::cpp;
using namespace std::chrono;

int iterators_mxnet();

void train();

int evaluate();

void printUsage();

/*The global context, change them if necessary*/
static mxnet::cpp::Context global_ctx(mxnet::cpp::kCPU, 0);
// static Context global_ctx(mxnet::cpp::kCPU,0);


/*
 * Convert the input string of number into the vector.
 */
template<typename T>
std::vector<T> createVectorFromString(const std::string &input_string) {
    std::vector<T> dst_vec;
    char *p_next;
    T elem;
    bool bFloat = std::is_same<T, float>::value;
    if (!bFloat) {
        elem = strtol(input_string.c_str(), &p_next, 10);
    } else {
        elem = strtof(input_string.c_str(), &p_next);
    }

    dst_vec.push_back(elem);
    while (*p_next) {
        if (!bFloat) {
            elem = strtol(p_next, &p_next, 10);
        } else {
            elem = strtof(p_next, &p_next);
        }
        dst_vec.push_back(elem);
    }
    return dst_vec;
}


int main(int argc, char const *argv[]) {
    int version;
    MXGetVersion(&version);

    LG << "Commit #: " << GIT_COMMIT_HASH;
    LG << "MxNet     version : " << version;
    LG << "Leptonica version : " << getLeptonicaVersion();
    LG << "OpenCV    version : " << CV_VERSION;
    LG << "MxNet Base";

    evaluate();

//    train();
    return 0;
}

int evaluate() {
    auto modelRoot = getDataDirectory({"models", "lenet"});
    auto testRoot = getDataDirectory({"mnist", "test", "standard"});
    auto dataRoot = getDataDirectory({"mnist", "standard"});

    LG << "MxNet Predicting";
    try {

        std::string synset_file = dataRoot / "synset.txt";
        std::string model_file_json = modelRoot / "lenet.json";
        std::string model_file_params = modelRoot / "lenet-4.params";
        std::string imageFile = testRoot / "3.png";


        // Generated via Python
        /*std::string model_file_json = "/home/gbugaj/dev/3rdparty/mxnet/example/image-classification/mnist_py-symbol.json";
        std::string model_file_params = "/home/gbugaj/dev/3rdparty/mxnet/example/image-classification/mnist_py-0015.params";
        */

        std::string input_rgb_mean("0 0 0");
        std::string input_rgb_std("1 1 1");

        bool use_gpu = false;
        bool enable_tensorrt = false;
        bool benchmark = false;
        int batch_size = 64;
        int num_inference_batches = 100;
        std::string data_layer_type("float32");
        std::string input_shape("1 28 28");
        int seed = 48564309;
        int shuffle_chunk_seed = 3982304;
        int data_nthreads = 10;

        if (model_file_json.empty()
            || (!benchmark && model_file_params.empty())
            || (enable_tensorrt && model_file_params.empty())) {
            LG << "ERROR: Model details such as symbol, param files are not specified";
            printUsage();
            return 1;
        }

        std::vector<index_t> input_dimensions = createVectorFromString<index_t>(input_shape);
        input_dimensions.insert(input_dimensions.begin(), batch_size);
        Shape input_data_shape(input_dimensions);

        std::vector<float> rgb_mean = createVectorFromString<float>(input_rgb_mean);
        std::vector<float> rgb_std = createVectorFromString<float>(input_rgb_std);

        // Initialize the predictor object
        Predictor predict(model_file_json, model_file_params, synset_file, input_data_shape, use_gpu, enable_tensorrt,
                          data_nthreads, data_layer_type, rgb_mean, rgb_std, shuffle_chunk_seed,
                          seed, benchmark);

        benchmark = false;
        if (benchmark) {
            predict.BenchmarkScore(num_inference_batches);
        } else {
            predict.Score(imageFile);
        }
    } catch (dmlc::Error &err) {
        LG << "Status: FAIL " << err.what();
        LG << "With Error: " << MXGetLastError();
        return 1;
    }
    return 0;
}

void train() {
    Lenet trainer;
    trainer.train(100);
}

/*!
 * Dump existing data iterators
 * @return
 */
int iterators_mxnet() {
    mx_uint num_data_iter_creators;
    DataIterCreator *data_iter_creators = nullptr;

    int r = MXListDataIters(&num_data_iter_creators, &data_iter_creators);
    CHECK_EQ(r, 0);
    LG << "num_data_iter_creators = " << num_data_iter_creators;
    //output: num_data_iter_creators = 8

    const char *name;
    const char *description;
    mx_uint num_args;
    const char **arg_names;
    const char **arg_type_infos;
    const char **arg_descriptions;

    for (mx_uint i = 0; i < num_data_iter_creators; i++) {
        r = MXDataIterGetIterInfo(data_iter_creators[i], &name, &description,
                                  &num_args, &arg_names, &arg_type_infos,
                                  &arg_descriptions);
        CHECK_EQ(r, 0);
        LG << " i: " << i << ", name: " << name;
    }

    MXNotifyShutdown();
    return 0;
}


void printUsage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "imagenet_inference --symbol_file <model symbol file in json format>" << std::endl
              << "--params_file <model params file> " << std::endl
              << "--dataset <dataset used to run inference> " << std::endl
              << "--data_nthreads <default: 60> " << std::endl
              << "--input_shape <shape of input image e.g \"3 224 224\">] " << std::endl
              << "--rgb_mean <mean value to be subtracted on RGB channel e.g \"0 0 0\">"
              << std::endl
              << "--rgb_std <standard deviation on R/G/B channel. e.g \"1 1 1\"> " << std::endl
              << "--batch_size <number of images per batch> " << std::endl
              << "--num_skipped_batches <skip the number of batches for inference> " << std::endl
              << "--num_inference_batches <number of batches used for inference> " << std::endl
              << "--data_layer_type <default: \"float32\" "
              << "--gpu  <whether to run inference on GPU, default: false>" << std::endl
              << "--enableTRT  <whether to run inference with TensorRT, "
              << "default: false>" << std::endl
              << "--benchmark <whether to use dummy data to run inference, default: false>"
              << std::endl;
}