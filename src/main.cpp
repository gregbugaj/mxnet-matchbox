#include <iostream>
#include <set>
#include <experimental/filesystem>
#include <mxnet-cpp/MxNetCpp.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>
#include <version.h>
#include "Trainer.hpp"
#include "fileutil.hpp"
#include "Predictor.hpp"
#include "vectorutils.hpp"

namespace fs = std::experimental::filesystem;
using namespace mxnet::cpp;
using namespace std::chrono;

void train();

int evaluate();

void printUsage();

/*The global context, change them if necessary*/
static mxnet::cpp::Context global_ctx(mxnet::cpp::kCPU, 0);
// static Context global_ctx(mxnet::cpp::kCPU,0);



void version() {
    int version;
    MXGetVersion(&version);
    LG << "Commit #: " << GIT_COMMIT_HASH;
    LG << "MxNet     version : " << version;
    LG << "Leptonica version : " << getLeptonicaVersion();
    LG << "OpenCV    version : " << CV_VERSION;
    LG << "MxNet Base";
}

int main(int argc, char const *argv[]) {
    version();
    evaluate();

//    train();
    return 0;
}

int evaluate() {

//    auto modelRoot = getDataDirectory({"models", "lenet"});
    auto modelRoot = getDataDirectory({"models", "py-mlp"});
    auto testRoot = getDataDirectory({"mnist", "test", "standard"});
    auto dataRoot = getDataDirectory({"mnist", "standard"});

    LG << "MxNet Predicting";
    try {

        /*
            std::string synset_file = dataRoot / "synset.txt";
            std::string model_file_json = modelRoot / "lenet-symbol.json";
            std::string model_file_params = modelRoot / "lenet-99.params";
            std::string imageFile = testRoot / "black/3_img_106.jpg";
         */
        std::string synset_file = dataRoot / "synset.txt";
        std::string model_file_json = modelRoot / "lenet-symbol.json";
        std::string model_file_params = modelRoot / "lenet-0001.params";
        std::string imageFile = testRoot / "black/2_img_1.jpg";
//        std::string imageFile = testRoot / "black/3_img_106.jpg";
//        std::string imageFile = testRoot / "black/8_img_110.jpg";

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
            predict.predict(imageFile);
        }
    } catch (dmlc::Error &err) {
        LG << "Status: FAIL " << err.what();
        LG << "With Error: " << MXGetLastError();
        return 1;
    }
    return 0;
}

void train() {
    Trainer trainer;
    trainer.train(1);
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