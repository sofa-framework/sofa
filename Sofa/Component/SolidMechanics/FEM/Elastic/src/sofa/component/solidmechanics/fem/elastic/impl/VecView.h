#pragma once

#include <sofa/type/Vec.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template<sofa::Size N, typename ValueType>
struct VecView
{
    typedef sofa::Size       Size;
    typedef ValueType        value_type;
    typedef sofa::Size       size_type;
    typedef std::ptrdiff_t   difference_type;

    template<sofa::Size L>
    VecView(sofa::type::Vec<L, ValueType>& vec)
        : m_data(vec.data())
    {}

    template<sofa::Size L>
    VecView(sofa::type::Vec<L, ValueType>& vec, sofa::Size i)
        : m_data(&vec.elems.data()[i])
    {}

    value_type operator[](sofa::Size i) const
    {
        return m_data[i];
    }

    value_type& operator[](sofa::Size i)
    {
        return m_data[i];
    }

    sofa::type::Vec<N, ValueType> operator-() const
    {
        sofa::type::Vec<N, ValueType> res(sofa::type::NOINIT);
        for (sofa::Size i = 0; i < N; ++i)
            res[i] = -m_data[i];
        return res;
    }

    sofa::type::Vec<N, ValueType> toVec() const
    {
        sofa::type::Vec<N, ValueType> res(sofa::type::NOINIT);
        for (sofa::Size i = 0; i < N; ++i)
            res[i] = m_data[i];
        return res;
    }

    VecView& operator=(const sofa::type::Vec<N, ValueType>& vec)
    {
        for (sofa::Size i = 0; i < N; ++i)
            m_data[i] = vec[i];
        return *this;
    }

private:
    ValueType* m_data { nullptr };
};


template<sofa::Size L, sofa::Size C, typename ValueType>
sofa::type::Vec<L, ValueType> operator*(const sofa::type::Mat<L, C, ValueType>& mat, const VecView<C, ValueType>& vec)
{
    sofa::type::Vec<L, ValueType> res(sofa::type::NOINIT);

    for (sofa::Size i = 0; i < L; ++i)
    {
        for (sofa::Size j = 0; j < C; ++j)
        {
            res[i] += mat(i, j) * vec[j];
        }
    }

    return res;
}

}
