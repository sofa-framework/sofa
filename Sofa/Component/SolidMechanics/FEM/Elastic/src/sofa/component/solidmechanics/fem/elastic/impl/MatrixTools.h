#pragma once

#include <sofa/type/Mat.h>

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * Alias for a sofa::type::Mat
 * If T is sofa::type::Mat<L,C,real> and L==1 && C==1, the alias is the scalar type.
 * Otherwise, the alias is T itself.
 *
 * Example: static_cast<ScalarOrMatrix<MatType>>(matrix)
 */
template <class T>
using ScalarOrMatrix = std::conditional_t<
    (T::nbLines==1 && T::nbCols==1), typename T::Real, T>;

template <sofa::Size N, typename real>
real determinantSquareMatrix(const sofa::type::Mat<N, N, real>& mat)
{
    if constexpr (N == 1)
    {
        return mat(0, 0);
    }
    else if constexpr (N == 2)
    {
        return mat(0, 0) * mat(1, 1) - mat(0, 1) * mat(1, 0);
    }
    else
    {
        real det = 0;
        for (size_t p = 0; p < N; ++p)
        {
            sofa::type::Mat<N - 1, N - 1, real> submat;
            for (size_t i = 1; i < N; ++i)
            {
                size_t colIndex = 0;
                for (size_t j = 0; j < N; ++j)
                {
                    if (j == p) continue;
                    submat(i - 1, colIndex++) = mat(i, j);
                }
            }
            det += ((p % 2 == 0) ? 1 : -1) * mat(0, p) * determinantSquareMatrix(submat);
        }
        return det;
    }
}

template <sofa::Size N, class real>
real determinant(const sofa::type::Mat<N, N, real>& mat)
{
    return determinantSquareMatrix(mat);
}

template <sofa::Size L, sofa::Size C, class real>
real absGeneralizedDeterminant(const sofa::type::Mat<L, C, real>& mat)
{
    if constexpr (L == C)
    {
        return std::abs(determinantSquareMatrix(mat));
    }
    else
    {
        return std::sqrt(determinantSquareMatrix(mat.multTranspose(mat)));
    }
}

template <sofa::Size L, sofa::Size C, class real>
sofa::type::Mat<C, L, real> leftPseudoInverse(const sofa::type::Mat<L, C, real>& mat)
{
    return mat.multTranspose(mat).inverted() * mat.transposed();
}

/**
 * Computes the inverse of a given matrix.
 * For square matrices (L == C), the standard matrix inverse is computed.
 * For non-square matrices, the left pseudo-inverse is returned.
 *
 * @param mat The input matrix of size LxC to be inverted or pseudo-inverted.
 * @return A matrix of size CxL representing the inverse or left pseudo-inverse of the input matrix.
 */
template <sofa::Size L, sofa::Size C, class real>
sofa::type::Mat<C, L, real> inverse(const sofa::type::Mat<L, C, real>& mat)
{
    if constexpr (L == C)
    {
        return mat.inverted();
    }
    else
    {
        return leftPseudoInverse(mat);
    }
}

}
