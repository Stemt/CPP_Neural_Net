#include <iostream>
#include "NeuralNet.hpp"
#include "BMP.hpp"

void bmp_to_greyscale_mat(BMP& bmp, Matrix& mat)
{
    std::vector<float> normalized;
    normalized.reserve(128*128);
    for(size_t x = 0; x < 128; x++){
        for(size_t y = 0; y < 128; y++){
            auto rgba = bmp.get_pixel(x,y);
            float total = (float)(rgba[0] + rgba[1] + rgba[2]);
            total /= 3;
            total /= 255;
            normalized.push_back(total);
        }
    }
    mat = Matrix(normalized,128,128);
}


void load_data(std::vector<std::vector<Matrix>>& data)
{
    for(size_t digit = 0; digit < 10; digit++)
    {
        size_t image_count = 0;
        bool done = false;
        std::vector<Matrix> images({});
        while(!done)
        {
            try
            {
                std::string filename = "data/"+std::to_string(digit)+"/"+std::to_string(image_count)+".bmp";
                std::cout << "loading: " << filename << std::endl;
                BMP bmp(filename.c_str());
                Matrix mat;
                bmp_to_greyscale_mat(bmp,mat);
                mat.reshape(1,128*128);
                images.push_back(mat);
                image_count++;
            }
            catch(std::exception& e)
            {
                done = true;
            }
        }
        data.push_back(images);
    }
}

void generate_batches(const std::vector<std::vector<Matrix>>& data, size_t batch_size, size_t batch_count, std::vector<TrainingBatch>& batches)
{
    for(size_t n_batch = 0; n_batch < batch_count; n_batch++)
    {
        TrainingBatch batch;
        for(size_t count = 0; count < batch_size; count++)
        {
            size_t digit = (rand() % 10);
            size_t index = (rand() % data.at(digit).size());
            Matrix desired(1,10);
            desired.set_value(0,digit,1);
            batch.add_sample(data.at(digit).at(index),desired);
            std::cout << "added batch sample digit: " << digit << " index: " << index << std::endl;
            batches.push_back(batch);
        }
    }
}

int main(){

    std::vector<std::vector<Matrix>> data({});
    std::vector<TrainingBatch> batches;

    load_data(data);
    generate_batches(data,30,10,batches);



    BMP bmp("data/0/0.bmp");
    Matrix mat;
    bmp_to_greyscale_mat(bmp,mat);
    mat.reshape(1,128*128);

    Matrix expected({1,0,0,0,0,0,0,0,0,0},1,10);
    TrainingBatch batch;
    batch.add_sample(mat,expected);
    
    NeuralNet net;
    net.add_layer(128*128);
    net.add_layer(32);
    net.add_layer(10);
    net.randomize();
  
    std::cout << "batch size: " <<  batches.size() << std::endl;

    for(size_t i = 0; i < 10000; i++){
        net.process_batch(0.1,batches.at(i%10));
        if(i%1000 == 0){
            float total = 0;
            for(size_t i = 0; i < batch.size(); i++){
                net.set_input(batch[i].first);
                net.feedforward();
                total += net.calculate_cost(batch[i].second);
            }
            std::cout << "-------------------------\naverage error: " << total / batch.size() << std::endl;
        }
    }



    return 0;
}