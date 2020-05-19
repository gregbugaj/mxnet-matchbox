#ifndef MATCHBOX_DIRDATASETITERATOR_HPP
#define MATCHBOX_DIRDATASETITERATOR_HPP

#include <dataset/DataSetIterator.hpp>

class DirDataSetIterator : public DataSetIterator {

public :

    DirDataSetIterator(std::string &directory, int batch_size, int num_examples,
                       bool shuffle, int seed)
            : directory(directory), batch_size(batch_size) {

        validateFile(directory);
    }

    MXDataIter getMXDataIter() override;

private :
    int batch_size;
    std::string directory;
};


#endif //MATCHBOX_DIRDATASETITERATOR_HPP
