//
// Created by robert-denomme on 8/6/24.
//



#ifndef SMALL_MATRIX_H
#define SMALL_MATRIX_H

#include <sycl/sycl.hpp>

#include <cmath>

namespace small_la
{

template<int num_rows, int num_cols, bool col_major_storage>
int flatten_index(const size_t row, const size_t col)
{
    if constexpr(col_major_storage)
    {
        return row + num_rows * col;
    }
    else
    {
        return col + num_cols * row;
    }
}

constexpr int packed_size(const int num_rows, const int num_cols)
{
    //static_assert(num_rows <= 4, "num rows must be less than 4");
    //static_assert(num_cols <= 4, "num cols must be less than 4");

    const int size_required = num_rows * num_cols;

    if(size_required <= 1)
        return 1;
    if(size_required <= 2)
        return 2;
    if(size_required <= 3)
        return 3;
    if(size_required <= 4)
        return 4;
    if(size_required <= 8)
        return 8;
    return 16;
}

//A matrix that fits into one of the
//sycl::vec<*,*> primitives
template<class Tscalar_t, int Tnum_rows, int Tnum_cols, bool col_major_storage = true>
struct small_matrix
{
public:
    using scalar_t = Tscalar_t;
    static constexpr int num_rows = Tnum_rows;
    static constexpr int num_cols = Tnum_cols;

private:
    using storage_t = sycl::vec<scalar_t, packed_size(num_rows, num_cols)>;
    using this_t = small_matrix<scalar_t, num_rows, num_cols, col_major_storage>;
    storage_t data;

public:
    template<typename... Tscalar_dummy>
    explicit small_matrix(const Tscalar_dummy&... entries) : data(0)
    {
        static_assert(sizeof...(entries) == num_rows * num_cols, "Wrong number of entries to a matrix");
        std::array<scalar_t, num_rows * num_cols> entries_arr = { std::forward<const scalar_t>(entries)... };
        for(int i = 0; i < Tnum_rows; i++)
        {
            for(int j = 0; j < Tnum_cols; j++)
            {
                data[flatten_index<num_rows, num_cols, col_major_storage>(i, j)] = entries_arr[flatten_index<Tnum_rows, Tnum_cols, false>(i, j)];
            }
        }
    }
public:
    small_matrix() = default;
    ~small_matrix() = default;
    small_matrix(const small_matrix<Tscalar_t, Tnum_rows, Tnum_cols, col_major_storage>&) = default;
    small_matrix<Tscalar_t, Tnum_rows, Tnum_cols, col_major_storage>& operator=(const small_matrix<Tscalar_t, Tnum_rows, Tnum_cols, col_major_storage>& ) = default;
    small_matrix(small_matrix<Tscalar_t, Tnum_rows, Tnum_cols, col_major_storage>&&) = default;
    small_matrix<Tscalar_t, Tnum_rows, Tnum_cols, col_major_storage>& operator=(small_matrix<Tscalar_t, Tnum_rows, Tnum_cols, col_major_storage>&& ) = default;

public:

    Tscalar_t& operator()(const size_t row, const size_t col)
    {
        return data[flatten_index<num_rows, num_cols, col_major_storage>(row, col)];
    }

    const Tscalar_t& operator()(const size_t row, const size_t col) const
    {
        return data[flatten_index<num_rows, num_cols, col_major_storage>(row, col)];
    }

    template<int num_rows_dummy = num_rows, int num_cols_dummy = num_cols>
    std::enable_if_t<num_rows_dummy == 1 || num_cols_dummy == 1, Tscalar_t&> operator()(const size_t index)
    {
        return data[index];
    }

    template<int num_rows_dummy = num_rows, int num_cols_dummy = num_cols>
    std::enable_if_t<num_rows_dummy == 1 || num_cols_dummy == 1, const Tscalar_t&> operator()(const size_t index) const
    {
        return data[index];
    }

    small_matrix<scalar_t, num_cols, num_rows, col_major_storage> transpose() const
    {
        using ret_t = small_matrix<scalar_t, num_cols, num_rows, col_major_storage>;
        ret_t ret = ret_t::Zero();
        for(int i = 0; i < num_rows; i++)
            for(int j = 0; j < num_cols; j++)
                ret(j, i) = (*this)(i, j);
        return ret;
    }

    template<class TScalar_dummy = scalar_t>
    std::enable_if_t<std::is_integral_v<TScalar_dummy>, bool> operator==(const small_matrix& other) const
    {
        for(int i = 0; i < num_rows; ++i)
        {
            for(int j = 0; j < num_cols; ++j)
            {
                if((*this)(i,j) != other(i,j))
                {
                    return false;
                }
            }
        }
        return true;
    }

    template<bool Tcol_major_storage_other>
    scalar_t dot(const small_matrix<scalar_t, num_rows, 1, Tcol_major_storage_other> other)
    {
        scalar_t ret = 0;
        for(int i = 0; i < num_rows; ++i)
        {
            ret += data[i] * other.data[i];
        }

        return ret;
    }

private:
    static this_t MakeZero()
    {
        this_t ret;
        ret.data = 0;
        return ret;
    }

    template<int Tnum_rows_dummy = num_rows, int Tnum_cols_dummy = num_cols>
    static std::enable_if_t<Tnum_rows_dummy == Tnum_cols_dummy, this_t> MakeIdentity()
    {
        this_t ret = MakeZero();
        for(int i = 0; i < num_rows; ++i)
            ret(i, i) = 1;
        return ret;
    }

    static this_t MakeOnes()
    {
        this_t ret = MakeZero();
        for(int i = 0; i < num_rows; ++i)
            for(int j = 0; j < num_cols; ++j)
                ret(i, j) = 1;
        return ret;
    }

public:
    static this_t Zero()
    {
        //TODO figure out how to make this const static
        this_t zero = MakeZero();
        return zero;
    }

    static this_t Identity()
    {
        //TODO figure out how to make this const static
        this_t I = MakeIdentity();
        return I;
    }

    static this_t Ones()
    {
        //TODO figure out how to make this const static
        this_t ones = MakeOnes();
        return ones;
    }

public:
    //Arithmetic
    this_t& operator*=(const scalar_t s)
    {
        data *= s;
        return *this;
    }

    this_t& operator+=(const this_t& other)
    {
        data += other.data;
        return *this;
    }

    this_t& operator-=(const this_t& other)
    {
        data -= other.data;
        return *this;
    }

public:
    small_matrix<int, num_rows, num_cols, col_major_storage> floor()
    {
        small_matrix<int, num_rows, num_cols, col_major_storage> ret = small_matrix<int, num_rows, num_cols, col_major_storage>::Zero();
        auto floor_data = data.template convert<int, sycl::rounding_mode::rtn>();
        for(int i = 0; i < num_rows; ++i)
            for(int j = 0; j < num_cols; ++j)
                ret(i, j) = floor_data[flatten_index<Tnum_rows, Tnum_cols, false>(i, j)];

        return ret;

    }

    //small_matrix<int, num_rows, num_cols, col_major_storage> get_floor()
    //{
    //    return floor(*this);
    //}

};

template<class Tscalar_t, int Tnum_rows, int Tnum_cols, bool col_major_storage>
small_matrix<Tscalar_t, Tnum_rows, Tnum_cols, col_major_storage> operator*(
    const Tscalar_t s,
    const small_matrix<Tscalar_t, Tnum_rows, Tnum_cols, col_major_storage>& x
    )
{
    auto ret = x;
    ret *= s;
    return ret;
}

template<class Tscalar_t, int Tnum_rows, int Tnum_cols, bool col_major_storage>
small_matrix<Tscalar_t, Tnum_rows, Tnum_cols, col_major_storage> operator+(
    const small_matrix<Tscalar_t, Tnum_rows, Tnum_cols, col_major_storage>& x,
    const small_matrix<Tscalar_t, Tnum_rows, Tnum_cols, col_major_storage>& y
    )
{
    auto ret = x;
    ret += y;
    return ret;
}

template<class Tscalar_t, int Tnum_rows, int Tnum_cols, bool col_major_storage>
small_matrix<Tscalar_t, Tnum_rows, Tnum_cols, col_major_storage> operator-(
    const small_matrix<Tscalar_t, Tnum_rows, Tnum_cols, col_major_storage>& x,
    const small_matrix<Tscalar_t, Tnum_rows, Tnum_cols, col_major_storage>& y
    )
{
    auto ret = x;
    ret -= y;
    return ret;
}

template<class Tscalar_t, int Tnum_rows, int Tnum_cols, bool col_major_storage>
small_matrix<Tscalar_t, Tnum_rows, Tnum_cols, col_major_storage> operator-(
    const small_matrix<Tscalar_t, Tnum_rows, Tnum_cols, col_major_storage>& x
    )
{
    auto ret = x;
    ret *= - 1.0;
    return ret;
}

template<class Tscalar_t, int Tnum_rowsA, int Tnum_colsA, bool Tcol_major_storageA,
 int Tnum_rowsB, int Tnum_colsB, bool Tcol_major_storageB>
small_matrix<Tscalar_t, Tnum_rowsA, Tnum_colsB, Tcol_major_storageA> operator*(
    const small_matrix<Tscalar_t, Tnum_rowsA, Tnum_colsA, Tcol_major_storageA>& A,
    const small_matrix<Tscalar_t, Tnum_rowsB, Tnum_colsB, Tcol_major_storageB>& B)
{
    static_assert(Tnum_colsA == Tnum_rowsB, "Dimensions incorrect for matrix multiplication!");
    small_matrix<Tscalar_t, Tnum_rowsA, Tnum_colsB, Tcol_major_storageA> ret = small_matrix<Tscalar_t, Tnum_rowsA, Tnum_colsB, Tcol_major_storageA>::Zero();
    for(int i = 0; i < Tnum_rowsA; i ++)
    {
        for(int j = 0; j < Tnum_colsB; j++)
        {
            Tscalar_t r_ij = 0;
            for(int k = 0; k < Tnum_colsA; k++)
            {
                r_ij += A(i, k) * B(k, j);
            }
            ret(i, j) = r_ij;
        }
    }
    return ret;

}

template<class Tscalar_t, int Tnum_rows, bool col_major_storageA, bool col_major_storageB>
Tscalar_t dot(
    const small_matrix<Tscalar_t, Tnum_rows, 1, col_major_storageA>& v,
    const small_matrix<Tscalar_t, Tnum_rows, 1, col_major_storageB>& w
    )
{
    Tscalar_t ret = 0;
    for(int i = 0; i < v.num_rows; i++)
    {
        ret += v(i, 0) * w(i, 0);
    }
    return ret;
}

template<class Tscalar_t, int Tnum_rows, bool col_major_storage>
std::enable_if_t<Tnum_rows == 2, small_matrix<Tscalar_t, Tnum_rows, Tnum_rows, col_major_storage>> inverse(
    const small_matrix<Tscalar_t, Tnum_rows, Tnum_rows, col_major_storage>& A
    )
{
    using scalar_t = Tscalar_t;
    using ret_t = small_matrix<Tscalar_t, Tnum_rows, Tnum_rows, col_major_storage>;

    ret_t ret = ret_t::Zero();

    scalar_t a = A(0,0);
    scalar_t b = A(0,1);
    scalar_t c = A(1,0);
    scalar_t d = A(1,1);

    scalar_t det = a * d - b * c;
    ret(0,0) = d / det;
    ret(0,1) = -b / det;
    ret(1,0) = -c / det;
    ret(1,1) = a / det;

    return ret;

}

template<class Tscalar_t, int Tnum_rows, bool col_major_storage>
std::enable_if_t<Tnum_rows == 2, Tscalar_t> det(
        const small_matrix<Tscalar_t, Tnum_rows, Tnum_rows, col_major_storage>& A
)
{
    using scalar_t = Tscalar_t;

    scalar_t a = A(0,0);
    scalar_t b = A(0,1);
    scalar_t c = A(1,0);
    scalar_t d = A(1,1);

    return a * d - b * c;
}

template<class Tscalar_t, int Tnum_rows, bool col_major_storage>
std::enable_if_t<Tnum_rows == 2, void> PolarDecomposition(
        const small_matrix<Tscalar_t, Tnum_rows, Tnum_rows, col_major_storage>& A,
        small_matrix<Tscalar_t, Tnum_rows, Tnum_rows, col_major_storage>& R,
        small_matrix<Tscalar_t, Tnum_rows, Tnum_rows, col_major_storage>& S
)
{
    using scalar_t = Tscalar_t;
    using Matrix_t = small_matrix<Tscalar_t, Tnum_rows, Tnum_rows, col_major_storage>;

    const scalar_t x = A(0,0) + A(1, 1);
    const scalar_t y = A(1, 0) - A(0,1);

    const scalar_t norm = std::sqrt(x * x + y * y);
    const scalar_t c = x / norm;
    const scalar_t s = -y / norm;

    R = Matrix_t(c, s, -s, c);
    S = R.transpose() * A;
}

template<class Tscalar_t, int Tnum_rows, bool col_major_storage>
std::enable_if_t<Tnum_rows == 2, void> SVD(
        const small_matrix<Tscalar_t, Tnum_rows, Tnum_rows, col_major_storage>& A,
        small_matrix<Tscalar_t, Tnum_rows, Tnum_rows, col_major_storage>& U,
        small_matrix<Tscalar_t, Tnum_rows, Tnum_rows, col_major_storage>& Sigma,
        small_matrix<Tscalar_t, Tnum_rows, Tnum_rows, col_major_storage>& V
)
{
    using scalar_t = Tscalar_t;
    using Matrix_t = small_matrix<Tscalar_t, Tnum_rows, Tnum_rows, col_major_storage>;
    Matrix_t R = Matrix_t::Zero(), S = Matrix_t::Zero();
    PolarDecomposition(A, R, S);
    scalar_t c, s, sigma_0, sigma_1;
    if(S(0,1) == 0.0)
    {
        c = 1.0;
        s = 0.0;
        sigma_0 = S(0,0);
        sigma_1 = S(1,1);
    }
    else
    {
        const scalar_t tau = 0.5 * (S(0,0) - S(1,1));
        const scalar_t w = std::sqrt(tau * tau + S(0,1) * S(0,1));
        const scalar_t t = tau > 0 ? S(0,1) / (tau + w) : S(0,1) / (tau - w);
        c = 1.0 / std::sqrt(t * t + 1);
        s = -t * c;
        sigma_0 = c * c * S(0,0) - 2 * c * s * S(0,1) + s * s * S(1,1);
        sigma_1 = s * s * S(0,0) + 2 * c * s * S(0,1) + c * c * S(1,1);
    }
    if(sigma_0 < sigma_1)
    {
        std::swap(sigma_0, sigma_1);
        V = Matrix_t(-s, c, -c, -s);
    }
    else
    {
        V = Matrix_t(c, s, -s, c);
    }
    U = R * V;
    Sigma(0,0) = sigma_0;
    Sigma(1,1) = sigma_1;

}

//template<typename... Ts, class Tscalar_t, int Tnum_rows>
//small_matrix<Tscalar_t, Tnum_rows, 1> MakeVector(Ts&&... entries)
//{
//    return small_matrix<Tscalar_t, Tnum_rows, 1>(entries);
//    //small_matrix<Tscalar_t, Tnum_rows, 1> ret;
//    //for(int i = 0; i < Tnum_rows; i++)
//    //{
//    //    ret(i, 0) = entries[i];
//    //}
//    //return ret;
//}
//
//template<typename... Ts, class Tscalar_t, int Tnum_rows, int Tnum_cols>
//small_matrix<Tscalar_t, Tnum_rows, Tnum_cols> MakeMatrix(Ts&&... entries)
//{
//    small_matrix<Tscalar_t, Tnum_rows, Tnum_cols> ret;
//    for(int i = 0; i < Tnum_rows; i++)
//    {
//        for(int j = 0; j < Tnum_cols; j++)
//        {
//            ret(i, j) = entries[flatten_index<Tnum_rows, Tnum_cols, false>(i, j)];
//        }
//    }
//    return ret;
//}

}

#endif //SMALL_MATRIX_H
