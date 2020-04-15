#ifndef MATCHBOX_PREDICTOR_HPP
#define MATCHBOX_PREDICTOR_HPP

#include <string>
#include <set>
#include <fstream>
#include <experimental/filesystem>
#include "mxnet/c_api.h"
#include "mxnet/tuple.h"
#include <mxnet-cpp/MxNetCpp.h>

namespace fs = std::experimental::filesystem;
using namespace mxnet::cpp;
using namespace std::chrono;

/**
 * class Predictor
 *
 * This class encapsulates the functionality to load the model, prepare dataset and run the forward pass.
 * https://mxnet.apache.org/api/cpp/docs/tutorials/cpp_inference
 */
class Predictor {
public:
    Predictor() {}

    Predictor(const std::string &model_json_file,
              const std::string &model_params_file,
              const Shape &input_shape,
              bool use_gpu,
              bool enable_tensorrt,
              const int data_nthreads,
              const std::string &data_layer_type,
              const std::vector<float> &rgb_mean,
              const std::vector<float> &rgb_std,
              int shuffle_chunk_seed,
              int seed, bool benchmark);

    /// Run benchmark against the model
    /// \param num_inference_batches
    void BenchmarkScore(int num_inference_batches);

    /// The following function runs the forward pass on the model and use real data for testing accuracy and performance.
    /// https://mxnet.apache.org/api/cpp/docs/tutorials/cpp_inference
    /// https://discuss.mxnet.io/t/run-time-is-different-between-python-and-c/4052/2
    /// \param image_file
    void Score(const std::string &image_file);

    ~Predictor();

private:
    /// Load input image we want to predict and converts it to NDArray for prediction
    /// \param image_file
    /// \return
    NDArray LoadInputImage(const std::string &image_file);

    void LoadModel(const std::string &model_json_file);

    void LoadParameters(const std::string &model_parameters_file);

    void LoadSynset(const std::string& synset_file);

    void SplitParamMap(const std::map<std::string, NDArray> &paramMap,
                       std::map<std::string, NDArray> *argParamInTargetContext,
                       std::map<std::string, NDArray> *auxParamInTargetContext,
                       Context targetContext);

    void ConvertParamMapToTargetContext(const std::map<std::string, NDArray> &paramMap,
                                        std::map<std::string, NDArray> *paramMapInTargetContext,
                                        Context targetContext);

    void InitParameters();

    inline bool FileExists(const std::string &name) {
        std::ifstream fhandle(name.c_str());
        return fhandle.good();
    }

    int GetDataLayerType();

    std::map<std::string, NDArray> args_map_;
    std::map<std::string, NDArray> aux_map_;
    std::vector<std::string> output_labels;

    Symbol net_;
    Executor *executor_;
    Shape input_shape_;
    Context global_ctx_ = Context::cpu();
    bool use_gpu_;
    bool enable_tensorrt_;
    int data_nthreads_;
    std::string data_layer_type_;
    std::vector<float> rgb_mean_;
    std::vector<float> rgb_std_;
    int shuffle_chunk_seed_;
    int seed_;
    bool benchmark_;
};


#endif //MATCHBOX_PREDICTOR_HPP
