#include <sys/time.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "Predictor.hpp"
#include "utils.hpp"

double ms_now() {
    auto timePoint = std::chrono::high_resolution_clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(timePoint).count();
}

// define the data type for NDArray, aliged with the definition in mshadow/base.h
enum TypeFlag {
    kFloat32 = 0,
    kFloat64 = 1,
    kFloat16 = 2,
    kUint8 = 3,
    kInt32 = 4,
    kInt8 = 5,
    kInt64 = 6,
};

/*
 * The constructor takes following parameters as input:
 * 1. model_json_file:  The model in json formatted file.
 * 2. model_params_file: File containing model parameters
 * 3. input_shape: Shape of input data to the model. Since this class will be running one inference at a time,
 *                 the input shape is required to be in format Shape(1, number_of_channels, height, width)
 *                 The input image will be resized to (height x width) size before running the inference.
 * 4. use_gpu: determine if run inference on GPU
 * 5. enable_tensorrt: determine if enable TensorRT
 * 6. dataset: data file (.rec) to be used for inference
 * 7. data_nthreads: number of threads for data loading
 * 8. data_layer_type: data type for data layer
 * 9. rgb_mean: mean value to be subtracted on R/G/B channel
 * 10. rgb_std: standard deviation on R/G/B channel
 * 11. shuffle_chunk_seed: shuffling chunk seed
 * 12. seed: shuffling seed
 * 13. benchmark: use dummy data for inference
 *
 * The constructor will:
 *  1. Create ImageRecordIter based on the given dataset file.
 *  2. Load the model and parameter files.
 *  3. Infer and construct NDArrays according to the input argument and create an executor.
 */
Predictor::Predictor(const std::string &model_json_file,
                     const std::string &model_params_file,
                     const std::string &synset_file,
                     const Shape &input_shape,
                     bool use_gpu,
                     bool enable_tensorrt,
                     const int data_nthreads,
                     const std::string &data_layer_type,
                     const std::vector<float> &rgb_mean,
                     const std::vector<float> &rgb_std,
                     int shuffle_chunk_seed,
                     int seed, bool benchmark)
        : input_shape_(input_shape),
          use_gpu_(use_gpu),
          enable_tensorrt_(enable_tensorrt),
          data_nthreads_(data_nthreads),
          data_layer_type_(data_layer_type),
          rgb_mean_(rgb_mean),
          rgb_std_(rgb_std),
          shuffle_chunk_seed_(shuffle_chunk_seed),
          seed_(seed),
          benchmark_(benchmark) {

    if (use_gpu) {
        global_ctx_ = Context::gpu();
    }

    // Load Synset(Labels)
    LoadSynset(synset_file);

    // Load the model
    LoadModel(model_json_file);

    // Initialize the parameters
    // benchmark=true && model_params_file.empty(), randomly initialize parameters
    // else, load parameters
    if (benchmark_ && model_params_file.empty()) {
        InitParameters();
    } else {
        LoadParameters(model_params_file);
    }

    int dtype = GetDataLayerType();
    if (dtype == -1) {
        throw std::runtime_error("Unsupported data layer type...");
    }

    Shape label_shape(input_shape_[0]);

    args_map_["data"] = NDArray(input_shape_, global_ctx_, false, dtype);
    args_map_["data_label"] = NDArray(label_shape, global_ctx_, false);

    std::vector<NDArray> arg_arrays;
    std::vector<NDArray> grad_arrays;
    std::vector<OpReqType> grad_reqs;
    std::vector<NDArray> aux_arrays;

    // infer and create ndarrays according to the given input ndarrays.
    net_.InferExecutorArrays(global_ctx_, &arg_arrays, &grad_arrays, &grad_reqs,
                             &aux_arrays, args_map_, std::map<std::string, NDArray>(),
                             std::map<std::string, OpReqType>(), aux_map_);

    for (auto &i : grad_reqs) {
        i = OpReqType::kNullOp;
    }

    // Create an executor after binding the model to input parameters.
    executor_ = new Executor(net_, global_ctx_, arg_arrays, grad_arrays, grad_reqs, aux_arrays);
    for (const auto &layer_name:net_.ListOutputs()) {
        LG << layer_name;
    }
}

/*
 * The following function is used to get the data layer type for input data
 */
int Predictor::GetDataLayerType() {
    int ret_type = -1;
    if (data_layer_type_ == "float32") {
        ret_type = kFloat32;
    } else if (data_layer_type_ == "int8") {
        ret_type = kInt8;
    } else if (data_layer_type_ == "uint8") {
        ret_type = kUint8;
    } else {
        LG << "Unsupported data layer type " << data_layer_type_ << "..."
           << "Please use one of {float32, int8, uint8}";
    }
    return ret_type;
}

/// https://answers.opencv.org/question/72564/how-can-i-convert-an-image-into-a-1d-vector-in-c/
NDArray Predictor::LoadInputImage(const std::string &image_file) {
    if (!FileExists(image_file)) {
        LG << "Image file " << image_file << " does not exist";
        throw std::runtime_error("Image file does not exist");
    }
    LG << "Loading the image " << image_file << std::endl;
    cv::Mat mat = cv::imread(image_file, cv::IMREAD_COLOR);
//    mat.convertTo(mat, CV_32F);
    mat.convertTo(mat, CV_32F, 1.f/255);
    // resize pictures to (28, 28) according to the pretrained model
    int channels = input_shape_[1];
    int height = input_shape_[2];
    int width = input_shape_[3];

    cv::resize(mat, mat, cv::Size(width, height));
    std::vector<float> array((float *) mat.data, (float *) mat.data + mat.rows * mat.cols);
    std::cout << mat;

    NDArray image_data = NDArray(input_shape_, global_ctx_, false);
    image_data.SyncCopyFromCPU(array.data(), input_shape_.Size());
    NDArray::WaitAll();
    return image_data;
}

/*
 * The following function loads the model from json file.
 */
void Predictor::LoadModel(const std::string &model_json_file) {
    if (!FileExists(model_json_file)) {
        LG << "Model file " << model_json_file << " does not exist";
        throw std::runtime_error("Model file does not exist");
    }
    LG << "Loading the model from " << model_json_file << std::endl;
    net_ = Symbol::Load(model_json_file);//.GetInternals()["softmax_output"];

    if (enable_tensorrt_) {
        net_ = net_.GetBackendSymbol("TensorRT");
    }

    LG << "-------- Net Arguments --------";
    for (const auto &args_name : net_.ListArguments()) {
        LG << args_name;
    }

    LG << "-------- Net outputs --------";
    for (const auto &layer_name : net_.ListOutputs()) {
        LG << layer_name;
    }

    LG << "-------- Input  Arguments --------";
    for (const std::string &name : net_.ListInputs()) {
        LG << name;
    }
    LG << "---------------------------";
}

/*
 * The following function loads the model parameters.
 */
void Predictor::LoadParameters(const std::string &model_parameters_file) {
    if (!FileExists(model_parameters_file)) {
        LG << "Parameter file " << model_parameters_file << " does not exist";
        throw std::runtime_error("Model parameters does not exist");
    }
    LG << "Loading the model parameters from " << model_parameters_file << std::endl;
    std::map<std::string, NDArray> parameters;
    NDArray::Load(model_parameters_file, 0, &parameters);
    if (enable_tensorrt_) {
        std::map<std::string, NDArray> intermediate_args_map;
        std::map<std::string, NDArray> intermediate_aux_map;
        SplitParamMap(parameters, &intermediate_args_map, &intermediate_aux_map, Context::cpu());
        contrib::InitTensorRTParams(net_, &intermediate_args_map, &intermediate_aux_map);
        ConvertParamMapToTargetContext(intermediate_args_map, &args_map_, global_ctx_);
        ConvertParamMapToTargetContext(intermediate_aux_map, &aux_map_, global_ctx_);
    } else {
        SplitParamMap(parameters, &args_map_, &aux_map_, global_ctx_);
    }

    /*WaitAll is needed when we copy data between GPU and the main memory*/
    NDArray::WaitAll();
}

/*
 * The following function split loaded param map into arg parm
 *   and aux param with target context
 */
void Predictor::SplitParamMap(const std::map<std::string, NDArray> &paramMap,
                              std::map<std::string, NDArray> *argParamInTargetContext,
                              std::map<std::string, NDArray> *auxParamInTargetContext,
                              Context targetContext) {
    for (const auto &pair : paramMap) {
        std::string type = pair.first.substr(0, 4);
        std::string name = pair.first.substr(4);
        LG << "ParamMap >>  " << type << " = " << name;
        if (type == "arg:") {
            (*argParamInTargetContext)[name] = pair.second.Copy(targetContext);
        } else if (type == "aux:") {
            (*auxParamInTargetContext)[name] = pair.second.Copy(targetContext);
        }
    }
}

/*
 * The following function copy the param map into the target context
 */
void Predictor::ConvertParamMapToTargetContext(const std::map<std::string, NDArray> &paramMap,
                                               std::map<std::string, NDArray> *paramMapInTargetContext,
                                               Context targetContext) {
    for (const auto &pair : paramMap) {
        (*paramMapInTargetContext)[pair.first] = pair.second.Copy(targetContext);
    }
}

/**
 * The following function randomly initializes the parameters when benchmark_ is true.
 */
void Predictor::InitParameters() {
    std::vector<mx_uint> data_shape;
    for (index_t i = 0; i < input_shape_.ndim(); i++) {
        data_shape.push_back(input_shape_[i]);
    }

    std::map<std::string, std::vector<mx_uint> > arg_shapes;
    std::vector<std::vector<mx_uint> > aux_shapes, in_shapes, out_shapes;
    arg_shapes["data"] = data_shape;
    net_.InferShape(arg_shapes, &in_shapes, &aux_shapes, &out_shapes);

    // initializer to call
    Xavier xavier(Xavier::uniform, Xavier::avg, 2.0f);

    auto arg_name_list = net_.ListArguments();
    for (index_t i = 0; i < in_shapes.size(); i++) {
        const auto &shape = in_shapes[i];
        const auto &arg_name = arg_name_list[i];
        int paramType = kFloat32;
        if (Initializer::StringEndWith(arg_name, "weight_quantize") ||
            Initializer::StringEndWith(arg_name, "bias_quantize")) {
            paramType = kInt8;
        }

        NDArray tmp_arr(shape, global_ctx_, false, paramType);
        xavier(arg_name, &tmp_arr);
        args_map_[arg_name] = tmp_arr.Copy(global_ctx_);
    }

    auto aux_name_list = net_.ListAuxiliaryStates();
    for (index_t i = 0; i < aux_shapes.size(); i++) {
        const auto &shape = aux_shapes[i];
        const auto &aux_name = aux_name_list[i];
        NDArray tmp_arr(shape, global_ctx_, false);
        xavier(aux_name, &tmp_arr);
        aux_map_[aux_name] = tmp_arr.Copy(global_ctx_);
    }

    /*WaitAll is need when we copy data between GPU and the main memory*/
    NDArray::WaitAll();
}

/**
 * The following function runs the forward pass on the model and use dummy data for benchmark.
 */
void Predictor::BenchmarkScore(int num_inference_batches) {
    // Create dummy data
    std::vector<float> dummy_data(input_shape_.Size());
    std::default_random_engine generator;
    std::uniform_real_distribution<float> val(0.0f, 1.0f);

    for (size_t i = 0; i < static_cast<size_t>(input_shape_.Size()); ++i) {
        dummy_data[i] = static_cast<float>(val(generator));
    }
    executor_->arg_dict()["data"].SyncCopyFromCPU(
            dummy_data.data(),
            input_shape_.Size());
    NDArray::WaitAll();

    LG << "Running the forward pass on model to evaluate the performance..";

    // warm up.
    for (int i = 0; i < 5; i++) {
        executor_->Forward(false);
        NDArray::WaitAll();
    }

    // Run the forward pass.
    double ms = ms_now();
    for (int i = 0; i < num_inference_batches; i++) {
        executor_->Forward(false);
        NDArray::WaitAll();
    }
    ms = ms_now() - ms;
    LG << " benchmark completed!";
    LG << " batch size: " << input_shape_[0] << " num batch: " << num_inference_batches
       << " throughput: " << 1000.0 * input_shape_[0] * num_inference_batches / ms
       << " imgs/s latency:" << ms / input_shape_[0] / num_inference_batches << " ms";
}

void Predictor::predict(const std::string &image_file) {
    // Load the input image
    NDArray image_data = LoadInputImage(image_file);
    LG << "Running the forward pass on model to predict the image";

    /*
     * The executor->arg_arrays represent the arguments to the model.
     *
     * Copying the image_data that contains the NDArray of input image
     * to the arg map of the executor. The input is stored with the key "data" in the map.
     */
    double ms = ms_now();
    image_data.CopyTo(&args_map_["data"]);
    NDArray::WaitAll();

    // Run the forward pass.
    executor_->Forward(false);
    NDArray::WaitAll();
    auto array = executor_->outputs[0].Copy(global_ctx_);
    /*
    * Find out the maximum accuracy and the index associated with that accuracy.
    * This is done by using the argmax operator on NDArray.
    */
    auto predicted = array.ArgmaxChannel();

    /*
     * Wait until all the previous write operations on the 'predicted'
     * NDArray to be complete before we read it.
     * This method guarantees that all previous write operations that pushed into the backend engine
     * for execution are actually finished.
     */
    predicted.WaitToRead();
    NDArray::WaitAll();

    auto best_idx = predicted.At(0);
    auto best_accuracy = array.At(0, best_idx);
    LG << "best_idx, best_accuracy = " << best_idx << " : " << best_accuracy;

    if (output_labels.empty()) {
        LG << "The model predicts the highest accuracy of " << best_accuracy << " at index "
           << best_idx;
    } else {
        LG << "The model predicts the input image to be a [" << output_labels[best_idx]
           << " ] with Accuracy = " << best_accuracy << std::endl;
    }

    mx_uint len = output_labels.size();
    std::vector<mx_float> pred_data(len);
    std::vector<mx_float> label_data(len);

    predicted.SyncCopyToCPU(&pred_data, len);

    // Display all candidates
    for (mx_uint i = 0; i < len; ++i) {
        auto val = pred_data[i];  // predicted
        auto label = label_data[i]; // expected

        best_idx = predicted.At(i);
        best_accuracy = array.At(0, best_idx);
        LG << "best_idx, best_accuracy = " << best_idx << " : " << best_accuracy;
        auto accuracy = array.At(0, i);
        LG << "Found, Expected, Accuracy  :: " << i << " : " << val << " = " << label << " : " << accuracy << " == "
           << best_accuracy;
    }
    ms = ms_now() - ms;
    auto args_name = net_.ListArguments();
    LG << "INFO:" << "label_name = " << args_name[args_name.size() - 1];
    LG << "INFO:" << "rgb_mean: " << "(" << rgb_mean_[0] << ", " << rgb_mean_[1]
       << ", " << rgb_mean_[2] << ")";
    LG << "INFO:" << "rgb_std: " << "(" << rgb_std_[0] << ", " << rgb_std_[1]
       << ", " << rgb_std_[2] << ")";
    LG << "INFO:" << "Image shape: " << "(" << input_shape_[1] << ", "
       << input_shape_[2] << ", " << input_shape_[3] << ")";
    LG << "INFO:" << "Batch size = " << input_shape_[0] << " for inference";
    LG << "INFO:" << "Throughput: " << (1000.0 * input_shape_[0] / ms)
       << " images per second";
}

/*
 * The following function loads the synset file.
 * This information will be used later to report the label of input image.
 */
void Predictor::LoadSynset(const std::string &synset_file) {
    LG << "Loading the synset file : " << synset_file;
    std::ifstream fi(synset_file.c_str());
    if (!fi.is_open()) {
        std::cerr << "Error opening synset file " << synset_file << std::endl;
        std::abort();
    }
    std::string lemma;
    while (getline(fi, lemma)) {
        output_labels.push_back(lemma);
    }
    fi.close();
}

Predictor::~Predictor() {
    if (executor_) {
        delete executor_;
    }
    MXNotifyShutdown();
}