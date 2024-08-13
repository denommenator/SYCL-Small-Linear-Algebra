//
// Created by robert-denomme on 8/5/24.
//

#include <sycl/sycl.hpp>
#include <small_la/small_matrix.hpp>

#include <iostream>


template<class mat_class>
void print_matrix(mat_class A)
{
    for(size_t i = 0; i < A.num_rows; i++)
    {
        for(size_t j = 0; j < A.num_cols; j++)
        {
            std::cout << A(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main()
{
    std::cout << "working!" << std::endl;

    sycl::mfloat2 m;
    constexpr size_t numrows = 3;
    constexpr size_t numcols = 2;

    small_la::small_matrix<double, numrows, numcols> v;

    for(size_t i = 0; i < v.num_rows; i++)
    {
        for(size_t j = 0; j < v.num_cols; j++)
        {
            v(i, j) = i + numrows * j;
        }
    }


    v = 2.0 * v;
    v = -v;

    print_matrix(v);
    /*
    for(size_t i = 0; i < numrows; i++)
    {
        for(size_t j = 0; j < numcols; j++)
        {
            std::cout << v(i, j) << " ";
        }
        std::cout << std::endl;
    }

    */


    auto w = v.transpose();
    auto vtv = w * v;
    auto vvt = v * w;

    print_matrix(w);
    std::cout << std::endl;

    print_matrix(vtv);
    std::cout << std::endl;

    print_matrix(vvt);
    std::cout << std::endl;

    small_la::small_matrix<double, 3, 1> x;
    x(0, 0) = 0.0;
    x(1, 0) = 1.0;
    x(2, 0) = 2.0;

    std::cout << small_la::dot(x, x) << std::endl;

    auto y_arr = small_la::MakeVector<double, 3>({0.0, 1.0, 2.0});
    auto M_arr = small_la::MakeMatrix<double, 2, 3>(
        {
            0.0, 1.0, 2.0,
            3.0, 4.0, 5.0
        }
        );
    print_matrix(M_arr);

}
