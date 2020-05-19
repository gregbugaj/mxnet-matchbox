#ifndef MATCHBOX_MNISTDATASETITERATOR_HPP
#define MATCHBOX_MNISTDATASETITERATOR_HPP

#include <dataset/DataSetIterator.hpp>

class MnistDataSetIterator : public DataSetIterator {

public :

    /**
     *
     * @param image_file the input image file
     * @param label_file the input label
     * @param batch_size  nubmer of items per batch
     * @param num_examples
     * @param shuffle
     * @param seed
     */
    MnistDataSetIterator(std::string image_file, std::string label_file, int batch_size,
                         bool shuffle, int seed)
            : _image_file(image_file), _label_file(label_file), _batch_size(batch_size) {

        validateFile(image_file);
        validateFile(label_file);
    }

    MXDataIter getMXDataIter() override;

    int batch() override {
        return _batch_size;
    };

    int totalOutcomes() override {
        return 10;
    }

    std::vector<std::string> getLabels() override {
        std::vector<std::string> labels;
        return labels;
    }

private :
    int _batch_size;
    std::string _image_file;
    std::string _label_file;
};


#endif //MATCHBOX_MNISTDATASETITERATOR_HPP
