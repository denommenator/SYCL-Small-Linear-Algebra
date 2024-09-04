//
// Created by robert-denomme on 8/13/24.
//
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>


#include <sycl/sycl.hpp>

#include <vector>
#include <small_la/small_matrix.hpp>
#include "small_matrix_test_helpers.h"

TEST_CASE( "PolarDecomp test", "[SVD]" )
{
    using Matrix_t = small_la::small_matrix<double, 2, 2>;
    double c_u = std::cos(1), s_u = std::sin(1);
    double c_v = std::cos(2), s_v = std::sin(2);
    double sigma_0 = 3, sigma_1 = 2;
    Matrix_t U_actual(c_u, s_u, -s_u, c_u), V_actual(c_v, s_v, -s_v, c_v);
    Matrix_t Sigma_actual(sigma_0, 0, 0, sigma_1);

    Matrix_t A = U_actual * Sigma_actual * V_actual.transpose();

    Matrix_t R_actual = U_actual * V_actual.transpose();
    Matrix_t S_actual = R_actual.transpose() * A;

    Matrix_t R, S;

    small_la::PolarDecomposition(A, R, S);

    std::cout << "R and R_actual" << std::endl;
    print(R);
    print(R_actual);

    std::cout << "S and S_actual" << std::endl;
    print(S);
    print(S_actual);


    CHECK(ApproxEqual(R, R_actual));
    CHECK(ApproxEqual(S, S_actual));


}
TEST_CASE( "SVD test", "[SVD]" )
{
    using Matrix_t = small_la::small_matrix<double, 2, 2>;
    double c_u = std::cos(1), s_u = std::sin(1);
    double c_v = std::cos(2), s_v = std::sin(2);
    double sigma_0 = 3, sigma_1 = 2;
    Matrix_t U_actual(c_u, s_u, -s_u, c_u), V_actual(c_v, s_v, -s_v, c_v);
    Matrix_t Sigma_actual(sigma_0, 0, 0, sigma_1);

    Matrix_t U, V, Sigma;
    Matrix_t A = U_actual * Sigma_actual * V_actual.transpose();
    small_la::SVD(A, U, Sigma, V);

//    std::cout << "U and U_actual" << std::endl;
//    print(U);
//    print(U_actual);
//
//    std::cout << "V and V_actual" << std::endl;
//    print(V);
//    print(V_actual);
//
//    std::cout << "Sigma and Sigma_actual" << std::endl;
//    print(Sigma);
//    print(Sigma_actual);
    CHECK(ApproxEqual(U, U_actual));
    CHECK(ApproxEqual(V, V_actual));
    CHECK(ApproxEqual(Sigma, Sigma_actual));

}