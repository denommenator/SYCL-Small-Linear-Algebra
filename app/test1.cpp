//
// Created by robert-denomme on 8/5/24.
//

#include <sycl/sycl.hpp>
#include <small_la/column_matrix.h>
#include <small_la/sycl_column_matrix.h>

#include <iostream>



int main()
{
    std::cout << "working!" << std::endl;

    constexpr size_t numrows = 3;
    constexpr size_t numcols = 3;

    small_la::matrix3x3d v;
    v.columns[0] = {0.0, 1.0, 2.0};
    v.columns[1] = {3.0, 4.0, 5.0};
    v.columns[2] = {6.0, 7.0, 8.0};

    v *= 2;

    for(size_t i = 0; i < numrows; i++)
    {
        for(size_t j = 0; j < numcols; j++)
        {
            std::cout << v(i, j) << " ";
        }
        std::cout << std::endl;
    }

}
