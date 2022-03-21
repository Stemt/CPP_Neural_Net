#include "NeuralNet.hpp"
#include <cstdlib>

NeuralNet::NeuralNet()
:neuron_layers({}),weight_layers({}),bias_layers({})
{}

void NeuralNet::add_layer(size_t n_neurons)
{
    if(neuron_layers.size() > 0)
    {
        size_t previous_layer_height = neuron_layers.back().get_height();
        neuron_layers.emplace_back(1,n_neurons);
        weight_layers.emplace_back(previous_layer_height,n_neurons);
        bias_layers.emplace_back(1,n_neurons);
        error_layers.emplace_back(1,n_neurons);
    }else{
        neuron_layers.emplace_back(1,n_neurons);
        weight_layers.emplace_back(1,1); // not used
        bias_layers.emplace_back(1,1); // not used
        error_layers.emplace_back(1,1); // not used
    }
}

void NeuralNet::set_input(const Matrix& input)
{
    if(input.get_width() != 1){
        throw std::invalid_argument("input data must be column vector thus the width must be 1");
    }
    neuron_layers.at(0) = input;
}

const Matrix& NeuralNet::get_output()
{
    return neuron_layers.back();
}

float NeuralNet::sigmoid(float x)
{
    return 1/(1+exp(x));
}

void NeuralNet::sigmoid(Matrix& mat)
{
    for(auto& value : mat.get_data())
    {
        value = sigmoid(value);
    }
}

float NeuralNet::sigmoid_derivative(float x)
{
    return sigmoid(x)*(1-sigmoid(x));
}

void NeuralNet::sigmoid_derivative(Matrix& mat)
{
    for(auto& value : mat.get_data())
    {
        value = sigmoid_derivative(value);
    }
}

void NeuralNet::feedforward()
{
    for(size_t n_layer = 1; n_layer < neuron_layers.size(); n_layer++)
    {
        // //std::cout << "compute layer " + std::to_string(n_layer) << std::endl;

        // //std::cout << "neurons: " << neuron_layers.at(n_layer-1).str() << std::endl;
        // //std::cout << "weights: " << weight_layers.at(n_layer).str() << std::endl;
        // //std::cout << "biases: " << bias_layers.at(n_layer).str() << std::endl;
        neuron_layers.at(n_layer) = weight_layers.at(n_layer).dot(neuron_layers.at(n_layer-1)) + bias_layers.at(n_layer);
        sigmoid(neuron_layers.at(n_layer));
        // //std::cout << "output: " << bias_layers.at(n_layer-1).str() << std::endl;

    }
}

void NeuralNet::randomize()
{
    //randomize weights
    for(size_t n_layer = 1; n_layer < neuron_layers.size(); n_layer++)
    {
        std::vector<float>& data = weight_layers.at(n_layer).get_data();
        for(auto& value : data)
        {
            value = (float) (rand() % 10000);
            value /= 10000.0f;
        }
    }

    //randomize bias
    for(size_t n_layer = 1; n_layer < neuron_layers.size(); n_layer++)
    {
        std::vector<float>& data = bias_layers.at(n_layer).get_data();
        for(auto& value : data)
        {
            value = (float) (rand() % 10000);
            value /= 10000.0f;
        }
    }
}

void NeuralNet::process_batch(float learning_rate, const TrainingBatch& batch)
{
    bias_delta = bias_layers;
    weight_delta = weight_layers;

    std::vector<Matrix> bias_delta_sum = bias_layers;
    std::vector<Matrix> weight_delta_sum = weight_layers;
    for(size_t n_layer = 0; n_layer < neuron_layers.size(); n_layer++)
    {
        bias_delta_sum[n_layer] *= 0;
        weight_delta_sum[n_layer] *= 0;
    }


    for(size_t n_batch = 0; n_batch < batch.size(); n_batch++)
    {
        Matrix input = batch[n_batch].first;
        Matrix desired = batch[n_batch].second;

        backpropagate(input,desired);


        for(size_t n_layer = 0; n_layer < neuron_layers.size(); n_layer++)
        {
            bias_delta_sum[n_layer] += bias_delta[n_layer];
            weight_delta_sum[n_layer] += weight_delta[n_layer];
        }
    }

    // adjusting weights
    for(size_t n_layer = 0; n_layer < neuron_layers.size(); n_layer++)
    {
        weight_layers[n_layer] = weight_layers[n_layer] + (weight_delta_sum[n_layer] * (learning_rate/batch.size()));
        bias_layers[n_layer] = bias_layers[n_layer] + (bias_delta_sum[n_layer] * (learning_rate/batch.size()));
    }
}

void NeuralNet::backpropagate(const Matrix& input, const Matrix& desired)
{

    // feed forward
    set_input(input);
    std::vector<Matrix> intermediates({});

    for(size_t n_layer = 1; n_layer < neuron_layers.size(); n_layer++)
    {
        Matrix intermediate = weight_layers[n_layer].dot(neuron_layers[n_layer-1]) + bias_layers[n_layer];
        neuron_layers[n_layer] = intermediate;
        sigmoid(neuron_layers[n_layer]); 
        intermediates.push_back(intermediate);
    }

    for(size_t i = 0; i < intermediates.size(); i++){
        //std::cout << "z " << i << " shape: " << intermediates[i].shape_str() << std::endl;
    }

    // actual backward propagation
    Matrix sig_der(intermediates.back());
    sigmoid_derivative(sig_der);
    Matrix delta = (neuron_layers.back() - desired) * sig_der;
    //std::cout << "delta shape: " << delta.shape_str() << std::endl;

    bias_delta.back() = delta;
    //std::cout << "delta bias 2 shape: " << bias_delta.back().shape_str() << std::endl;
    weight_delta.back() = delta.dot(neuron_layers[neuron_layers.size()-2].transpose());
    //std::cout << "delta weight 2 shape: " << weight_delta.back().shape_str() << std::endl;
    
    for(size_t n_layer = neuron_layers.size()-2; n_layer >= 1; n_layer--)
    {
        //std::cout << "accesing z at " << n_layer-1 << std::endl;
        Matrix intermediate(intermediates[n_layer-1]);
        // //std::cout << "accesing z at " << n_layer << std::endl;
        // Matrix intermediate(intermediates[n_layer]);
        sig_der = intermediate;
        sigmoid_derivative(sig_der);

        //std::cout << "z " << n_layer << " shape: " << intermediate.shape_str() << std::endl;
        //std::cout << "weight " << n_layer + 1 << " shape: " << weight_layers[n_layer+1].shape_str() << std::endl;
        delta = weight_layers[n_layer+1].transpose().dot(delta) * sig_der;
        //std::cout << "delta shape: " << delta.shape_str() << std::endl;
        bias_delta[n_layer] = delta;
        //std::cout << "bias delta " << n_layer << " shape: " << bias_delta[n_layer].shape_str() << std::endl;

        weight_delta[n_layer] = delta.dot(neuron_layers[n_layer-1].transpose());
        //std::cout << "weight delta " << n_layer << " shape: " << weight_delta[n_layer].shape_str() << std::endl;
    }
}

float NeuralNet::calculate_cost(const Matrix& desired)
{
    return (neuron_layers.back() - desired).sum();
}

void NeuralNet::set_layer_neurons(size_t n_layer, const Matrix& mat)
{
    neuron_layers.at(n_layer) = mat;
}

void NeuralNet::set_layer_weights(size_t n_layer, const Matrix& mat)
{
    weight_layers.at(n_layer) = mat;
}

void NeuralNet::set_layer_bias(size_t n_layer, const Matrix& mat)
{
    bias_layers.at(n_layer) = mat;
}

const Matrix& NeuralNet::get_layer_neurons(size_t n_layer)
{
    return neuron_layers.at(n_layer);
}

const Matrix& NeuralNet::get_layer_weights(size_t n_layer)
{
    return weight_layers.at(n_layer);
}

const Matrix& NeuralNet::get_layer_bias(size_t n_layer)
{
    return bias_layers.at(n_layer);
}

std::string NeuralNet::to_str()
{
    //TODO implement
}

void NeuralNet::from_str(const std::string)
{
    //TODO implement
}