//
// Created by robert-denomme on 9/4/24.
//

#ifndef SMALL_LA_SMALL_MATRIX_TEST_HELPERS_H
#define SMALL_LA_SMALL_MATRIX_TEST_HELPERS_H

#include <catch2/catch.hpp>
#include <small_la/small_matrix.hpp>


template<class Tscalar_t, int Tnum_rows, int Tnum_cols, bool Tcol_majorA, bool Tcol_majorB>
bool ApproxEqual(const small_la::small_matrix<Tscalar_t, Tnum_rows, Tnum_cols, Tcol_majorA> A,
                 const small_la::small_matrix<Tscalar_t, Tnum_rows, Tnum_cols, Tcol_majorB> B)
{
    bool ret = true;
    for(int i = 0; i < Tnum_rows; ++i)
    {
        for (int j = 0; j < Tnum_cols; ++j)
        {
            ret = ret && A(i,j) == Approx(B(i,j)).margin(1.0E-6);
        }
    }
    return ret;
}

template<class Tscalar_t, int Tnum_rows, int Tnum_cols, bool Tcol_major>
void print(const small_la::small_matrix<Tscalar_t, Tnum_rows, Tnum_cols, Tcol_major> A)
{
    std::cout << std::endl;

    for(int i = 0; i < Tnum_rows; ++i)
    {
        for (int j = 0; j < Tnum_cols; ++j)
        {
            std::cout << A(i, j) << ", ";
        }
        std::cout << std::endl;
    }
}
#endif //SMALL_LA_SMALL_MATRIX_TEST_HELPERS_H
