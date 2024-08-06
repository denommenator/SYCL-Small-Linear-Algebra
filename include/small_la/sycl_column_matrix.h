//
// Created by robert-denomme on 8/6/24.
//

#ifndef SYCL_COLUMN_MATRIX_H
#define SYCL_COLUMN_MATRIX_H

#include <sycl/sycl.hpp>

#include "column_matrix.h"

namespace small_la
{
template<class scalar_t>
using matrix3x3 = column_matrix<sycl::vec<scalar_t, 3>, 3>;

using matrix3x3d = column_matrix<sycl::vec<double, 3>, 3>;
}

#endif //SYCL_COLUMN_MATRIX_H
