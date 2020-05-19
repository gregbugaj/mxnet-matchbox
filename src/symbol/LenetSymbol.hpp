#ifndef MATCHBOX_LENETSYMBOL_HPP
#define MATCHBOX_LENETSYMBOL_HPP

#include <mxnet-cpp/MxNetCpp.h>

using namespace mxnet::cpp;
// https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/lenet.cpp

class LenetSymbol {

public :
    /**
     * Lenet symbol
     *
     * @param num_classes
     * @return
     */
    Symbol symbol(int num_classes) {
        /*define the symbolic net*/
        Symbol data = Symbol::Variable("data");
        Symbol label = Symbol::Variable("data_label");

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
        Symbol fc2 = FullyConnected("fc2", tanh3, fc2_w, fc2_b, num_classes);
        // loss
        Symbol lenet = SoftmaxOutput("softmax", fc2, label);
        for (auto s : lenet.ListArguments()) {
            LG << s;
        }
        return lenet;
    }
};

#endif //MATCHBOX_LENETSYMBOL_HPP
