#ifndef TRAINING_BATCH_HPP
#define TRAINING_BATCH_HPP

#include "Matrix.hpp"
#include <vector>

class TrainingBatch{
    public:
        TrainingBatch();
        void add_sample(const Matrix& input, const Matrix& desired_output);
        std::pair<const Matrix&,const Matrix&> operator[](size_t index) const;
        size_t size() const;
    private:
        std::vector<std::pair<Matrix, Matrix>> training_cases;
};      

#endif