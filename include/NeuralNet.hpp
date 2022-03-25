#ifndef NEURALNET_HPP
#define NEURALNET_HPP

#include "TrainingBatch.hpp"

class NeuralNet{
    public:
        NeuralNet();
        void add_layer(size_t n_neurons);
        void set_input(const Matrix& input);
        const Matrix& get_output();
        float sigmoid(float x);
        void sigmoid(Matrix& mat);
        float sigmoid_derivative(float x);
        void sigmoid_derivative(Matrix& mat);
        void feedforward();
        void randomize();
        void process_batch(float learning_rate, const TrainingBatch& batch);
        void backpropagate(const Matrix& input, const Matrix& desired);
        float calculate_cost(const Matrix& desired);
        void set_layer_neurons(size_t n_layer, const Matrix& mat);
        void set_layer_weights(size_t n_layer, const Matrix& mat);
        void set_layer_bias(size_t n_layer, const Matrix& mat);
        const Matrix& get_layer_neurons(size_t n_layer);
        const Matrix& get_layer_weights(size_t n_layer);
        const Matrix& get_layer_bias(size_t n_layer);
        std::string to_str();
        void from_str(const std::string& str);
    private:
        std::vector<Matrix> neuron_layers;
        std::vector<Matrix> weight_layers;
        std::vector<Matrix> bias_layers;
        std::vector<Matrix> error_layers;
        std::vector<Matrix> bias_delta;
        std::vector<Matrix> weight_delta;

};

#endif