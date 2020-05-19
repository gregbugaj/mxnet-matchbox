#ifndef MATCHBOX_TRAINER_HPP
#define MATCHBOX_TRAINER_HPP

#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <cstdlib>
#include <mxnet-cpp/MxNetCpp.h>
#include <dataset/mnist/MnistDataSetIterator.hpp>
#include "utils.hpp"
#include "fileutil.hpp"

using namespace mxnet::cpp;
using mxnet::cpp::Symbol;

class Trainer {

public :
    Trainer() {
        // noop
    }

    NDArray ResizeInput(NDArray data, const Shape &new_shape) {
        int channels = new_shape[1];
        int height = new_shape[2];
        int width = new_shape[3];

        NDArray pic = data.Reshape(Shape(0, channels, height, width));
        NDArray output;
        Operator("_contrib_BilinearResize2D")
                .SetParam("height", height)
                .SetParam("width", width)
                        (pic).Invoke(output);
        return output;
    }

    /**
     * Train the network for max number of epochs
     *
     * @param net
     * @param max_epoch
     */
    void
    train(Context ctx, Symbol net, MXDataIter &train_iter, MXDataIter &validation_iter, int max_epoch,
          Shape input_shape, fs::path &destPath, std::string prefix) {

        int batch_size = input_shape[0];
        int channels = input_shape[1];
        int height = input_shape[2];
        int width = input_shape[3];

        LG << "---------- Training ----------";
        LG << "ctx : " << ctx.GetDeviceType();
        LG << "max_epoch : " << max_epoch;
        LG << "shape : " << input_shape;
        LG << "batch_size : " << batch_size;
        LG << "channels : " << channels;
        LG << "width : " << width;
        LG << "height : " << height;
        LG << "destPath : " << destPath;
        LG << "prefix : " << prefix;

        /*
        if (true)
            return;
 */
        float learning_rate = 1e-4;
        float weight_decay = 1e-4;

        /*args_map and aux_map is used for parameters' saving*/
        std::map<std::string, NDArray> args_map;
        std::map<std::string, NDArray> aux_map;
        const Shape data_shape = Shape(batch_size, channels, height, width);
        const Shape label_shape = Shape(batch_size);

        args_map["data"] = NDArray(data_shape, ctx);
        args_map["data_label"] = NDArray(label_shape, ctx);
        net.InferArgsMap(ctx, &args_map, args_map);

        Optimizer *opt = OptimizerRegistry::Find("sgd");
        opt->SetParam("momentum", 0.9)
                ->SetParam("rescale_grad", 1.0)
                ->SetParam("clip_gradient", 10)
                ->SetParam("lr", learning_rate)
                ->SetParam("wd", weight_decay);

        /*with data and label, executor can be generated automatically*/
        auto *exec = net.SimpleBind(ctx, args_map);
        auto arg_names = net.ListArguments();
        aux_map = exec->aux_dict();
        args_map = exec->arg_dict();

        //Initialize all parameters with uniform distribution U(-0.01, 0.01)
        auto xavier = Xavier();
        for (auto &arg : args_map) {
            //arg.first is parameter name, and arg.second is the value
            xavier(arg.first, &arg.second);
        }

        // Create metrics
        Accuracy train_acc, acu_val;
        LogLoss logloss_train, logloss_val;
        float score = 0;

        for (int epoch = 0; epoch < max_epoch; ++epoch) {
            LG << "Train Epoch: " << epoch;
            int samples = 0;
            /*reset the metric every epoch*/
            train_acc.Reset();
            /*reset the data iter every epoch*/
            train_iter.Reset();

            auto tic = std::chrono::system_clock::now();
            int iter = 0;
            while (train_iter.Next()) {
                samples += batch_size;
                auto data_batch = train_iter.GetDataBatch();
                /*use copyto to feed new data and label to the executor*/
                ResizeInput(data_batch.data, data_shape).CopyTo(&args_map["data"]);
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
                NDArray::WaitAll();
                // Update metrics
                train_acc.Update(data_batch.label, exec->outputs[0]);
                logloss_train.Reset();
                logloss_train.Update(data_batch.label, exec->outputs[0]);
                LG << "EPOCH: " << epoch << " ITER: " << iter
                   << " Train Accuracy: " << train_acc.Get()
                   << " Train Loss: " << logloss_train.Get();

                ++iter;
            }
            // one epoch of training is finished
            auto toc = std::chrono::system_clock::now();
            double duration = std::chrono::duration_cast<std::chrono::milliseconds>
                                      (toc - tic).count() / 1000.0;

            LG << "EPOCH [" << epoch << "] " << samples / duration << " samples/sec "
               << " Train Accuracy: " << train_acc.Get();

            LG << "Val Epoch: " << epoch;
            acu_val.Reset();
            validation_iter.Reset();
            logloss_val.Reset();
            iter = 0;

            while (validation_iter.Next()) {
                auto data_batch = validation_iter.GetDataBatch();
                ResizeInput(data_batch.data, data_shape).CopyTo(&args_map["data"]);
                data_batch.label.CopyTo(&args_map["data_label"]);
                NDArray::WaitAll();

                // Only forward pass is enough as no gradient is needed when evaluating
                exec->Forward(false);
                NDArray::WaitAll();

                acu_val.Update(data_batch.label, exec->outputs[0]);
                score = acu_val.Get();

                acu_val.Update(data_batch.label, exec->outputs[0]);
                logloss_val.Update(data_batch.label, exec->outputs[0]);
                LG << "EPOCH: " << epoch << " ITER: " << iter << " Val Accuracy: " << acu_val.Get();
                ++iter;
            }

            LG << "EPOCH: " << epoch << " ITER: " << iter
               << " Val Accuracy: " << acu_val.Get()
               << " Val Loss: " << logloss_val.Get();

            /* save the parameters, why not   sprintf(buff, "%05d", 123) ??? */
            std::string pid(std::to_string(epoch + 1));
            pid.insert(0, 4 - pid.length(), '0');
            std::string param_path = destPath / (prefix + "-" + pid + ".params");
            std::string model_path = destPath / (prefix + "-symbol.json");

            LG << "EPOCH [" << epoch << "] Saving params to..." << param_path;
            LG << "EPOCH [" << epoch << "] Saving model  to..." << model_path;
            // saving model so in case we stopped mid training we have something to work with
            SaveCheckpoint(param_path, model_path, net, exec);
        }

        LG << "Predicted score :  " << score;
        auto json = net.ToJSON();
        std::cerr << json;
        /*cleanup*/
        delete exec;
        delete opt;
        MXNotifyShutdown();
    }
};

#endif //MATCHBOX_TRAINER_HPP
