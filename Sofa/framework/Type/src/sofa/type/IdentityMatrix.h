#pragma once

#include <sofa/type/Mat.h>

namespace sofa::type
{

struct IdentityMatrix
{};

template<class real>
struct ScaledIdentityMatrix
{
    real scale;
};

template<sofa::Size N, class real>
constexpr Mat<N, N, real> operator+(const IdentityMatrix& I, const Mat<N, N, real>& M)
{
    Mat<N, N, real> res(M);
    for (sofa::Size i = 0; i < N; ++i)
    {
        res[i][i] += static_cast<real>(1);
    }
    return res;
}

template<sofa::Size N, class real>
constexpr Mat<N, N, real> operator+(const Mat<N, N, real>& M, const IdentityMatrix& I)
{
    return I + M;
}

template<sofa::Size N, class real>
constexpr Mat<N, N, real> operator-(const IdentityMatrix& I, const Mat<N, N, real>& M)
{
    Mat<N, N, real> res(-M);
    for (sofa::Size i = 0; i < N; ++i)
    {
        res[i][i] += static_cast<real>(1);
    }
    return res;
}

template<sofa::Size N, class real>
constexpr Mat<N, N, real> operator-(const Mat<N, N, real>& M, const IdentityMatrix& I)
{
    Mat<N, N, real> res(M);
    for (sofa::Size i = 0; i < N; ++i)
    {
        res[i][i] -= static_cast<real>(1);
    }
    return res;
}

template<class real>
constexpr ScaledIdentityMatrix<real> operator*(const IdentityMatrix& I, real s)
{
    return { s };
}

template<class real>
constexpr ScaledIdentityMatrix<real> operator*(real s, const IdentityMatrix& I)
{
    return { s };
}

template<sofa::Size N, class real>
constexpr Mat<N, N, real> operator-(const ScaledIdentityMatrix<real>& I, const Mat<N, N, real>& M)
{
    Mat<N, N, real> res(-M);
    for (sofa::Size i = 0; i < N; ++i)
    {
        res[i][i] += I.scale;
    }
    return res;
}

template<sofa::Size N, class real>
constexpr Mat<N, N, real> operator-(const Mat<N, N, real>& M, const ScaledIdentityMatrix<real>& I)
{
    Mat<N, N, real> res(M);
    for (sofa::Size i = 0; i < N; ++i)
    {
        res[i][i] -= I.scale;
    }
    return res;
}

template<sofa::Size N, class real>
constexpr Mat<N, N, real> operator+(const ScaledIdentityMatrix<real>& I, const Mat<N, N, real>& M)
{
    Mat<N, N, real> res(M);
    for (sofa::Size i = 0; i < N; ++i)
    {
        res[i][i] += I.scale;
    }
    return res;
}

template<sofa::Size N, class real>
constexpr Mat<N, N, real> operator+(const Mat<N, N, real>& M, const ScaledIdentityMatrix<real>& I)
{
    return I + M;
}



}
