
#ifndef MATCHBOX_MNISTDATASETITERATOR_HPP
#define MATCHBOX_MNISTDATASETITERATOR_HPP

#include <dataiter/DataSetIterator.hpp>

class MnistDataSetIterator : public DataSetIterator {

public :

    /**
     *
     * @param image_file the input image file
     * @param label_file the input label
     * @param batch_size  nubme o
     * @param num_examples
     * @param shuffle
     * @param seed
     */
    MnistDataSetIterator(std::string &image_file, std::string &label_file, int batch_size, int num_examples,
                         bool shuffle, int seed)
            : image_file(image_file), label_file(label_file), batch_size(batch_size) {

        validateFile(image_file);
        validateFile(label_file);
    }

    MXDataIter getMXDataIter() override;

private :
    int batch_size;
    std::string image_file;
    std::string label_file;
};


#endif //MATCHBOX_MNISTDATASETITERATOR_HPP
