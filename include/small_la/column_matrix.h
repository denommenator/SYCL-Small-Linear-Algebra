//
// Created by robert-denomme on 8/6/24.
//

#ifndef MATRIX_BLOCK_H
#define MATRIX_BLOCK_H


namespace small_la
{
template<class col_type, size_t num_cols>
struct column_matrix
{
    using this_t = column_matrix<col_type, num_cols>;
public:
    using value_type = typename col_type::value_type;
    col_type columns[num_cols];

    value_type operator()(size_t row, size_t column)
    {
        return columns[column][row];
    }

    this_t operator+(const this_t other) const
    {
        this_t ret;
        for(size_t j = 0; j < num_cols; j++)
        {
            ret.columns[j] = columns[j] + other.columns[j];
        }
        return ret;
    }

    this_t& operator+=(const this_t other)
    {
        return *this = *this + other;
    }

    this_t operator-(const this_t other) const
    {
        this_t ret;
        for(size_t j = 0; j < num_cols; j++)
        {
            ret.columns[j] = columns[j] - other.columns[j];
        }
        return ret;
    }

    this_t operator-() const
    {
        this_t ret;
        for(size_t j = 0; j < num_cols; j++)
        {
            ret.columns[j] = -columns[j];
        }
        return ret;
    }

    this_t& operator-=(const this_t other)
    {
        return *this = *this - other;
    }

    this_t& operator*=(const value_type s)
    {
        for(size_t j = 0; j < num_cols; j++)
        {
            columns[j] *= s;
        }
        return *this;
    }







};

template<class col_type, size_t num_cols>
column_matrix<col_type, num_cols> operator*(const typename col_type::value_type s, const column_matrix<col_type, num_cols> other)
{
    column_matrix<col_type, num_cols> ret;
    for(size_t j = 0; j < num_cols; j++)
    {
        ret.columns[j] = s * other.columns[j];
    }
    return ret;
}

}

#endif //MATRIX_BLOCK_H
