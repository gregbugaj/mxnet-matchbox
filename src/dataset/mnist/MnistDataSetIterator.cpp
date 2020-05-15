#include "MnistDataSetIterator.hpp"

MXDataIter MnistDataSetIterator::getMXDataIter() {
    return MXDataIter("MNISTIter")
            .SetParam("image", image_file)
            .SetParam("label", label_file)
            .SetParam("batch_size", batch_size)
            .SetParam("shuffle", 1)
            .SetParam("flat", 1)
            .CreateDataIter();
}