
#ifndef MATCHBOX_LENET_HPP
#define MATCHBOX_LENET_HPP

#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <cstdlib>
#include <mxnet-cpp/MxNetCpp.h>
#include "utils.hpp"

using namespace mxnet::cpp;
using mxnet::cpp::Symbol;

// https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/lenet.cpp
class Lenet {

public :

    Lenet()
            : ctx_cpu(Context(DeviceType::kCPU, 0)),
#if MXNET_USE_CPU
            ctx_dev(Context(DeviceType::kCPU, 0))
#else
              ctx_dev(Context(DeviceType::kCPU, 0))
#endif
    {}

    Symbol symbol() {
        /*define the symbolic net*/
        Symbol data = Symbol::Variable("data");
        Symbol data_label = Symbol::Variable("data_label");

        Symbol conv1_w("conv1_w"), conv1_b("conv1_b");
        Symbol conv2_w("conv2_w"), conv2_b("conv2_b");
        Symbol conv3_w("conv3_w"), conv3_b("conv3_b");
        Symbol fc1_w("fc1_w"), fc1_b("fc1_b");
        Symbol fc2_w("fc2_w"), fc2_b("fc2_b");

        conv1_w.SetAttribute("kernel", "(5, 5)");
        conv1_w.SetAttribute("num_filter", "20");

        conv1_b.SetAttribute("kernel", "(5, 5)");
        conv1_b.SetAttribute("num_filter", "20");

        conv2_w.SetAttribute("kernel", "(5, 5)");
        conv2_w.SetAttribute("num_filter", "50");

        conv2_b.SetAttribute("kernel", "(5, 5)");
        conv2_b.SetAttribute("num_filter", "50");

        fc1_w.SetAttribute("num_hidden", "500");
        fc1_b.SetAttribute("num_hidden", "500");

        fc2_w.SetAttribute("num_hidden", "10");
        fc2_w.SetAttribute("num_hidden", "10");

        // first conv
        Symbol conv1 = Convolution("conv1", data, conv1_w, conv1_b, Shape(5, 5), 20);
        Symbol tanh1 = Activation("tanh1", conv1, ActivationActType::kTanh);
        Symbol pool1 = Pooling("pool1", tanh1, Shape(2, 2), PoolingPoolType::kMax,
                               false, false, PoolingPoolingConvention::kValid, Shape(2, 2));
        // second conv
        Symbol conv2 = Convolution("conv2", pool1, conv2_w, conv2_b, Shape(5, 5), 50);
        Symbol tanh2 = Activation("tanh2", conv2, ActivationActType::kTanh);
        Symbol pool2 = Pooling("pool2", tanh2, Shape(2, 2), PoolingPoolType::kMax,
                               false, false, PoolingPoolingConvention::kValid, Shape(2, 2));

        // first fullc
        Symbol flatten = Flatten("flatten", pool2);
        Symbol fc1 = FullyConnected("fc1", flatten, fc1_w, fc1_b, 500);
        Symbol tanh3 = Activation("tanh3", fc1, ActivationActType::kTanh);
        // second fullc
        Symbol fc2 = FullyConnected("fc2", tanh3, fc2_w, fc2_b, 10);
        // loss
        Symbol lenet = SoftmaxOutput("softmax", fc2, data_label);

        for (auto s : lenet.ListArguments()) {
            LG << s;
        }

        return lenet;
    }

    NDArray ResizeInput(NDArray data, const Shape new_shape) {
        NDArray pic = data.Reshape(Shape(0, 1, 28, 28));
        NDArray output;
        Operator("_contrib_BilinearResize2D")
                .SetParam("height", new_shape[2])
                .SetParam("width", new_shape[3])
                        (pic).Invoke(output);
        return output;
    }

    /// Train the network for max number of epochs
    /// \param max_epoch
    void train(int max_epoch = 10) {
        LG << "Training";
        auto net = symbol();

        /*setup basic configs*/
        int W = 28;
        int H = 28;
        int batch_size = 32;

        float learning_rate = 1e-4;
        float weight_decay = 1e-4;

        auto dev_ctx = Context::cpu();
        int num_gpu;
        MXGetGPUCount(&num_gpu);
#if !MXNET_USE_CPU
        if (num_gpu > 0) {
            dev_ctx = Context::gpu();
        }
#endif

        std::map<std::string, NDArray> args_map;
        const Shape data_shape = Shape(batch_size, 1, H, W),
                label_shape = Shape(batch_size);

        args_map["data"] = NDArray(data_shape, dev_ctx);
        args_map["data_label"] = NDArray(label_shape, dev_ctx);

        net.InferArgsMap(dev_ctx, &args_map, args_map);

        std::vector<std::string> data_files = {
                "/home/gbugaj/dev/lbp-matcher/test-deck/data/mnist_data/train-images-idx3-ubyte",
                "/home/gbugaj/dev/lbp-matcher/test-deck/data/mnist_data/train-labels-idx1-ubyte",
                "/home/gbugaj/dev/lbp-matcher/test-deck/data/mnist_data/t10k-images-idx3-ubyte",
                "/home/gbugaj/dev/lbp-matcher/test-deck/data/mnist_data/t10k-labels-idx1-ubyte"
        };

        auto train_iter = MXDataIter("MNISTIter");
        if (!setDataIter(&train_iter, "Train", data_files, batch_size)) {
            throw std::runtime_error("Unable to create Train Iterator");
        }

        auto val_iter = MXDataIter("MNISTIter");
        if (!setDataIter(&val_iter, "Label", data_files, batch_size)) {
            throw std::runtime_error("Unable to create Train Iterator");
        }

        Optimizer *opt = OptimizerRegistry::Find("sgd");
        opt->SetParam("momentum", 0.9)
                ->SetParam("rescale_grad", 1.0)
                ->SetParam("clip_gradient", 10)
                ->SetParam("lr", learning_rate)
                ->SetParam("wd", weight_decay);

        auto *exec = net.SimpleBind(dev_ctx, args_map);
        auto arg_names = net.ListArguments();

        Accuracy train_acc;
        float score = 0;

        for (int iter = 0; iter < max_epoch; ++iter) {
            int samples = 0;
            train_iter.Reset();
            train_acc.Reset();

            auto tic = std::chrono::system_clock::now();

            while (train_iter.Next()) {
                samples += batch_size;
                auto data_batch = train_iter.GetDataBatch();
                auto resized = ResizeInput(data_batch.data, data_shape);

                resized.CopyTo(&args_map["data"]);
                data_batch.label.CopyTo(&args_map["data_label"]);
                NDArray::WaitAll();

                // Compute gradients
                exec->Forward(true);
                exec->Backward();

                // Update parameters
                for (size_t i = 0; i < arg_names.size(); ++i) {
                    if (arg_names[i] == "data" || arg_names[i] == "data_label")
                        continue;

                    opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
                }
                // Update metric
                train_acc.Update(data_batch.label, exec->outputs[0]);
            }

            /*save the parameters*/
            std::string prefix = "/home/gbugaj/dev/lbp-matcher/test-deck/data/lenet";
            std::string param_path = prefix + "-" + std::to_string(iter) + ".params";

            LG << "EPOCH: " << iter << " Saving to..." << param_path;
            SaveCheckpoint(param_path, net, exec);
            net.Save("/home/gbugaj/dev/lbp-matcher/test-deck/data/lenet.json");

            // one epoch of training is finished
            auto toc = std::chrono::system_clock::now();
            float duration = std::chrono::duration_cast<std::chrono::milliseconds>
                                     (toc - tic).count() / 1000.0;

            LG << "Epoch[" << iter << "] " << samples / duration << " samples/sec " << "Train-Accuracy="
               << train_acc.Get();

            Accuracy acc;
            val_iter.Reset();

            while (val_iter.Next()) {
                auto data_batch = val_iter.GetDataBatch();
                auto resized = ResizeInput(data_batch.data, data_shape);

                resized.CopyTo(&args_map["data"]);
                data_batch.label.CopyTo(&args_map["data_label"]);
                NDArray::WaitAll();

                // Only forward pass is enough as no gradient is needed when evaluating
                exec->Forward(false);
                NDArray::WaitAll();

                acc.Update(data_batch.label, exec->outputs[0]);
                score = acc.Get();
            }
            LG << "Epoch [" << iter << "] accuracy=" << acc.Get();
        }

        LG << "Score " << score;
        std::cerr << "Saving the model" << std::endl;
        auto json = net.ToJSON();
        std::cerr << json;
        net.Save("/home/gbugaj/dev/lbp-matcher/test-deck/data/lenet.json");
        std::cerr << "Done saving the model." << std::endl;

        delete exec;
        delete opt;
        MXNotifyShutdown();
    }

private:
    Context ctx_cpu;
    Context ctx_dev;
};

#endif //MATCHBOX_LENET_HPP
