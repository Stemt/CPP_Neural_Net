#include "Matrix.hpp"

Matrix::Matrix()
:width(0), height(0), data({})
{}

Matrix::Matrix(const std::vector<float>& data)
:width(data.size()), height(1), data(data)
{}

Matrix::Matrix(const std::vector<float>& data, size_t width, size_t height)
:width(width), height(height), data(data)
{}

Matrix::Matrix(size_t width, size_t height)
:width(width), height(height), data({})
{
    for(size_t i = 0; i < width * height; i++){
        data.push_back(0);
    }
}

Matrix::Matrix(const Matrix& mat)
:width(mat.width), height(mat.height), data({})
{
    // std::cout << "vector max: " << data.max_size() << " vs requested: " << mat.width * mat.height << std::endl;

    data.reserve(mat.width*mat.height);

    for(size_t i = 0; i < mat.data.size(); i++)
    {
        // std::cout << "at element: " << i << std::endl;
        data.push_back(mat.data.at(i));
    }
}

Matrix::~Matrix()
{}

Matrix Matrix::operator+(const Matrix& mat) const
{
    if(mat.width != width || mat.height != height){
        std::cout << "A:" << str() << std::endl;
        std::cout << "B:" << mat.str() << std::endl;
        throw std::invalid_argument("matrices must have same dimensions to be added");
    }
    std::vector<float> result;
    result.reserve(width*height);
    for(size_t i = 0; i < data.size(); i++){
        result.push_back(data.at(i) + mat.data.at(i));
    }
    Matrix new_mat(result,width,height);
    return new_mat;
}

Matrix Matrix::operator-(const Matrix& mat) const
{
    if(mat.width != width || mat.height != height){
        std::cout << "A:" << str() << std::endl;
        std::cout << "B:" << mat.str() << std::endl;
        throw std::invalid_argument("matrices must have same dimensions to be subtracted");
    }
    std::vector<float> result;
    result.reserve(width*height);
    for(size_t i = 0; i < data.size(); i++){
        result.push_back(data.at(i) - mat.data.at(i));
    }
    Matrix new_mat(result,width,height);
    return new_mat;
}

Matrix Matrix::operator-(float value) const{
    std::vector<float> result;
    result.reserve(data.size());
    for(size_t i = 0; i < data.size(); i++){
        result.push_back(data.at(i) - value);
    }
    Matrix new_mat(result,width,height);
    return new_mat;
}

Matrix Matrix::operator*(const Matrix& mat) const
{
    if(mat.width != width || mat.height != height){
        std::cout << "A shape:" << shape_str() << std::endl;
        std::cout << "B shape:" << mat.shape_str() << std::endl;
        throw std::invalid_argument("matrices must have same dimensions to be multiplied (non dot product)");
    }
    std::vector<float> result;
    result.reserve(width*height);
    for(size_t i = 0; i < data.size(); i++){
        result.push_back(data.at(i) * mat.data.at(i));
    }
    Matrix new_mat(result,width,height);
    return new_mat;
}

Matrix Matrix::operator*(float factor) const{
    std::vector<float> result;
    result.reserve(data.size());
    for(size_t i = 0; i < data.size(); i++){
        result.push_back(data.at(i) * factor);
    }
    Matrix new_mat(result,width,height);
    return new_mat;
}

Matrix Matrix::operator/(float factor) const{
    std::vector<float> result;
    result.reserve(data.size());
    for(size_t i = 0; i < data.size(); i++){
        result.push_back(data.at(i) / factor);
    }
    Matrix new_mat(result,width,height);
    return new_mat;
}

void Matrix::operator+=(const Matrix& mat)
{
    *this = *this + mat;
}

void Matrix::operator-=(const Matrix& mat)
{
    *this = *this - mat;
}

void Matrix::operator*=(const Matrix& mat)
{
    *this = *this * mat;
}

void Matrix::operator*=(float factor)
{
    *this = *this * factor;
}

Matrix& Matrix::operator=(const Matrix& mat)
{
    if(&mat != this){
        width = mat.width;
        height = mat.height;
        data = mat.data;
    }
    return *this;
}


Matrix Matrix::dot(const Matrix& mat) const
{
    if(width != mat.height){
        std::cout << "A shape:" << shape_str() << std::endl;
        std::cout << "B shape:" << mat.shape_str() << std::endl;
        throw std::invalid_argument("the width of matrix A must be equal to the height of matrix B to calculate the dot product.\n A width = " + std::to_string(width) + ", B height = " + std::to_string(mat.height));
    }

    std::vector<float> result;
    result.reserve(mat.width*height);

    for(size_t row_n = 0; row_n < height; row_n++){
        for(size_t col_n = 0; col_n < mat.width; col_n++){
            float sub_total = 0;
            for(size_t index = 0; index < width; index++){
                sub_total += get_value(index,row_n) * mat.get_value(col_n,index);
            }
            result.push_back(sub_total);
        }
    }
    Matrix new_mat(result,mat.width,height);
    return new_mat;
}

float Matrix::sum()
{
    float result = 0;
    for(float value : data)
    {
        result += value;
    }
    return result;
}

Matrix Matrix::transpose()
{
    Matrix t(height,width);
    for(size_t x = 0; x < width; x++){
        for(size_t y = 0; y < height; y++){
            t.set_value(y,x,get_value(x,y));
        }
    }
    return t;
}

float Matrix::get_value(size_t x, size_t y) const
{
    if(x >= width || y >= height){
        throw std::invalid_argument("x and or y value out of range");
    }

    size_t index = y * width + x;
    return data.at(index);
}

void Matrix::set_value(size_t x, size_t y, float value)
{
    if(x >= width || y >= height){
        throw std::invalid_argument("x and or y value out of range");
    }

    size_t index = y * width + x;
    data.at(index) = value;
}

size_t Matrix::get_width() const
{
    return width;
}

size_t Matrix::get_height() const
{
    return height;
}

std::vector<float>& Matrix::get_data()
{
    return data;
}

void Matrix::reshape(size_t new_width, size_t new_height)
{
    if(new_width * new_height != width * height){
        throw std::invalid_argument("cannot reshape from (" + std::to_string(width) + "," + std::to_string(height) + ") to (" + std::to_string(new_width) + "," + std::to_string(new_height) + ")");
    }
    width = new_width;
    height = new_height;
}

std::string Matrix::str() const
{
    std::string str = "";
    for(size_t y = 0; y < height; y++)
    {
        for(size_t x = 0; x < width; x++)
        {
            str += std::to_string(get_value(x,y));
            if(x != width-1){
                str += " ";
            }else{
                str += "\n";
            }
        }
    }
    return str;
}


void Matrix::from_str(const std::string& str)
{
    width = 0;
    height = 0;
    data.clear();
    std::string num_buf("");

    for(size_t n_char = 0; n_char < str.size(); n_char++)
    {
        if(str.at(n_char) == ' ')
        {
            if(height == 0){
                width++;
            }
            data.push_back(std::stof(num_buf));
            num_buf = "";
        }
        else if(str.at(n_char) == '\n')
        {
            height++;
            data.push_back(std::stof(num_buf));
            num_buf = "";
        }
        else
        {
            num_buf += str.at(n_char);
        }
    }
}

std::string Matrix::shape_str() const
{
    std::string str = "(" + std::to_string(width) + ", " + std::to_string(height) + ")";
    return str;
}