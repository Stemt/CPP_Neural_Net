#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <iostream>
#include <cmath>


class Matrix{
    public:
        Matrix();
        Matrix(const std::vector<float>& data);
        Matrix(const std::vector<float>& data, size_t width, size_t height);
        Matrix(size_t width, size_t height);
        Matrix(const Matrix& mat);
        ~Matrix();
        Matrix operator+(const Matrix& mat) const;
        Matrix operator-(const Matrix& mat) const;
        Matrix operator-(float value) const;
        Matrix operator*(const Matrix& mat) const;
        Matrix operator*(float factor) const;
        Matrix operator/(float factor) const;
        void operator+=(const Matrix& mat);
        void operator-=(const Matrix& mat);
        void operator*=(const Matrix& mat);
        void operator*=(float factor);
        Matrix& operator=(const Matrix& mat);
        Matrix dot(const Matrix& mat) const;
        float sum(); // returns sum of all elements of matrix
        Matrix transpose();
        float get_value(size_t x, size_t y) const;
        void set_value(size_t x, size_t y, float value);
        size_t get_width() const;
        size_t get_height() const;
        std::vector<float>& get_data();
        void set_data(const std::vector<float>& new_data);
        // Matrix get_column(size_t x) const;
        // Matrix get_row(size_t y) const;
        void reshape(size_t new_width, size_t new_height);
        std::string str() const;
        void from_str(const std::string& str);
        std::string shape_str() const;
    private:
        size_t width;
        size_t height;
        std::vector<float> data;
};

#endif