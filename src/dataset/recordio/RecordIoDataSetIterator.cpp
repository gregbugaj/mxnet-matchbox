#include "RecordIoDataSetIterator.hpp"

MXDataIter RecordIoDataSetIterator::getMXDataIter() {
    auto iter = MXDataIter("ImageRecordIter");

    // set image record parser parameters
    iter.SetParam("path_imgrec_", path_imgrec_);
    iter.SetParam("label_width", 1);
    iter.SetParam("data_shape_", data_shape_);
    iter.SetParam("preprocess_threads", preprocess_threads_); // number of threads for data decoding
    iter.SetParam("shuffle_chunk_seed", shuffle_chunk_seed_);

    // set Batch parameters
    iter.SetParam("batch_size", batch_size_);

    iter.SetParam("rand_crop", false);
    iter.SetParam("rand_mirror", false);

    // image record parameters
    iter.SetParam("shuffle", shuffle_);
    iter.SetParam("seed", seed_);

    // set normalize parameters
    iter.SetParam("mean_r", rgb_mean_[0]);
    iter.SetParam("mean_g", rgb_mean_[1]);
    iter.SetParam("mean_b", rgb_mean_[2]);
    iter.SetParam("std_r", rgb_std_[0]);
    iter.SetParam("std_g", rgb_std_[1]);
    iter.SetParam("std_b", rgb_std_[2]);

    return iter.CreateDataIter();
}
