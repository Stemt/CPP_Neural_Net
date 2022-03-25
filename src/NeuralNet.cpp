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


    // actual backward propagation
    Matrix sig_der(intermediates.back());
    sigmoid_derivative(sig_der);
    Matrix delta = (neuron_layers.back() - desired) * sig_der;

    bias_delta.back() = delta;
    weight_delta.back() = delta.dot(neuron_layers[neuron_layers.size()-2].transpose());
    
    for(size_t n_layer = neuron_layers.size()-2; n_layer >= 1; n_layer--)
    {
        Matrix intermediate(intermediates[n_layer-1]);
        sig_der = intermediate;
        sigmoid_derivative(sig_der);

        delta = weight_layers[n_layer+1].transpose().dot(delta) * sig_der;
        bias_delta[n_layer] = delta;

        weight_delta[n_layer] = delta.dot(neuron_layers[n_layer-1].transpose());
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
    std::string str("");

    // structure info
    str += "structure ";
    for(const auto& layer : neuron_layers)
    {
        str += std::to_string(layer.get_height());
        if(&layer == &neuron_layers.back())
        {
            str += "\n";
        }
        else
        {
            str += " ";
        }
    }

    // weight data
    for(size_t n_layer = 0; n_layer < weight_layers.size(); n_layer++)
    {
        str += "weight " + std::to_string(n_layer);
        for(const auto& value : weight_layers.at(n_layer).get_data())
        {
            str += " " + std::to_string(value);
        }
        str += "\n";
    }

    // bias data
    for(size_t n_layer = 0; n_layer < bias_layers.size(); n_layer++)
    {
        str += "bias " + std::to_string(n_layer);
        for(const auto& value : bias_layers.at(n_layer).get_data())
        {
            str += " " + std::to_string(value);
        }
        str += "\n";
    }

    return str;
}

void NeuralNet::from_str(const std::string& str)
{
    std::string buf("");

    bool done = false;

    
    for(size_t n_char = 0; n_char < str.size(); n_char++)
    {
        buf += str.at(n_char);

        // structure info
        if(buf == "structure "){
            buf = "";
            while(str.at(n_char) != '\n'){
                if(str.at(n_char) == ' '){
                    if(buf.size() > 0){
                        size_t layer_size = std::stoi(buf);
                        add_layer(layer_size);
                    }
                    buf = "";
                }
                buf += str.at(n_char);
                n_char++;
            }

            if(buf.size() > 0){
                size_t layer_size = std::stoi(buf);
                add_layer(layer_size);
            }
            buf = "";
        }

        // weight info
        if(buf == "weight "){
            buf = "";
            n_char++;
            while(str.at(n_char) != ' '){
                buf += str.at(n_char);
                n_char++;
            }
            size_t layer_index = std::stoi(buf);
            buf = "";
            std::vector<float> layer_data;
            while(str.at(n_char) != '\n'){
                if(str.at(n_char) == ' '){
                    if(buf.size() > 0){
                        layer_data.push_back(std::stof(buf));
                    }
                    buf = "";
                }
                buf += str.at(n_char);
                n_char++;
            }
            weight_layers.at(layer_index).set_data(layer_data);
            buf = "";
        }

        // bias info
        if(buf == "bias "){
            buf = "";
            n_char++;
            while(str.at(n_char) != ' '){
                buf += str.at(n_char);
                n_char++;
            }
            size_t layer_index = std::stoi(buf);
            buf = "";
            std::vector<float> layer_data;
            while(str.at(n_char) != '\n'){
                if(str.at(n_char) == ' '){
                    if(buf.size() > 0){
                        layer_data.push_back(std::stof(buf));
                    }
                    buf = "";
                }
                buf += str.at(n_char);
                n_char++;
            }
            bias_layers.at(layer_index).set_data(layer_data);
            buf = "";
        }
    }
}