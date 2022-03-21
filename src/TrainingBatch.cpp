#include "TrainingBatch.hpp"

TrainingBatch::TrainingBatch()
:training_cases({})
{}

void TrainingBatch::add_sample(const Matrix& input, const Matrix& desired_output)
{
    training_cases.push_back(std::make_pair<const Matrix&,const Matrix&>(input,desired_output));
}

std::pair<const Matrix&,const Matrix&> TrainingBatch::operator[](size_t index) const
{
    return training_cases[index];
}

size_t TrainingBatch::size() const
{
    return training_cases.size();
}