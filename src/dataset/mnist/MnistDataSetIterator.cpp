#include "MnistDataSetIterator.hpp"

MXDataIter MnistDataSetIterator::getMXDataIter() {
    return MXDataIter("MNISTIter")
            .SetParam("image", _image_file)
            .SetParam("label", _label_file)
            .SetParam("batch_size", _batch_size)
            .SetParam("shuffle", 1)
            .SetParam("flat", 1)
            .CreateDataIter();
}