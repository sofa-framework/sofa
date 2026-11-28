/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once
#include <sofa/component/linearsolver/direct/config.h>

#include <sofa/linearalgebra/matrix_bloc_traits.h>
#include <type_traits>
#include <utility>

namespace sofa::component::linearsolver::direct
{

/**
 * Elementary operations on the blocks of the matrices involved in a sparse LDL^T factorization.
 *
 * The factorization is written once, for any square block type: a scalar (1x1 block) or a dense
 * NxN block (type::Mat<N,N>). The access to the block entries goes through matrix_bloc_traits,
 * so all the operations are generic. For a scalar block, they reduce exactly to the scalar
 * operations (same operations in the same order).
 *
 * The only representation difference is the storage of the diagonal factor D:
 * - for a scalar, the pivot D(k) itself is stored during the numeric factorization, and the
 *   entries of L are computed as L(k,i) = Y(i) / D(i). D is inverted at the end of the
 *   factorization (see finalizeDiagonal).
 * - for a NxN block, the inverse of the dense symmetric block D(k) is stored, and the entries
 *   of L are computed as L(k,i) = Y(i)^T * D(i)^{-1}.
 * In both cases, after the factorization, the diagonal factor contains D^{-1}.
 *
 * Note that all the functions are explicitly inline: MSVC with /Ob1 (default for
 * RelWithDebInfo) only inlines functions that are explicitly declared inline.
 */
template<class TBlock>
struct SparseLDLBlockOperations
{
    using Block = TBlock;
    using traits = linearalgebra::matrix_bloc_traits<Block, sofa::SignedIndex>;
    using Real = typename traits::Real;

    static constexpr sofa::SignedIndex N = traits::NL;
    static_assert(static_cast<int>(traits::NL) == static_cast<int>(traits::NC), "The factorization requires square blocks");

    static constexpr bool IsScalar = std::is_scalar_v<Block>;

    static inline Real& v(Block& b, sofa::SignedIndex i, sofa::SignedIndex j) { return traits::v(b, i, j); }
    static inline const Real& v(const Block& b, sofa::SignedIndex i, sofa::SignedIndex j) { return traits::v(b, i, j); }

    static inline void clear(Block& b) { traits::clear(b); }

    /// res += a
    static inline void add(Block& res, const Block& a)
    {
        for (sofa::SignedIndex i = 0; i < N; ++i)
            for (sofa::SignedIndex j = 0; j < N; ++j)
                v(res, i, j) += v(a, i, j);
    }

    /// res += a^T
    static inline void addTransposed(Block& res, const Block& a)
    {
        for (sofa::SignedIndex i = 0; i < N; ++i)
            for (sofa::SignedIndex j = 0; j < N; ++j)
                v(res, i, j) += v(a, j, i);
    }

    /// res -= a * b
    static inline void subtractProduct(Block& res, const Block& a, const Block& b)
    {
        for (sofa::SignedIndex i = 0; i < N; ++i)
            for (sofa::SignedIndex k = 0; k < N; ++k)
            {
                const Real aik = v(a, i, k);
                for (sofa::SignedIndex j = 0; j < N; ++j)
                    v(res, i, j) -= aik * v(b, k, j);
            }
    }

    /// b = b^T
    static inline void transposeInPlace(Block& b)
    {
        for (sofa::SignedIndex i = 0; i < N; ++i)
            for (sofa::SignedIndex j = i + 1; j < N; ++j)
                std::swap(v(b, i, j), v(b, j, i));
    }

    /**
     * Compute the entry L(k,i) of the factorization from Y(i), the solution of the triangular
     * system for the row k, and the diagonal factor of i (see the documentation of the struct).
     */
    static inline void computeLowerEntry(Block& lki, const Block& yi, const Block& diagonalFactor_i)
    {
        if constexpr (IsScalar)
        {
            lki = yi / diagonalFactor_i;
        }
        else
        {
            // L(k,i) = Y(i)^T * D(i)^{-1}
            for (sofa::SignedIndex r = 0; r < N; ++r)
                for (sofa::SignedIndex c = 0; c < N; ++c)
                {
                    Real s = 0;
                    for (sofa::SignedIndex m = 0; m < N; ++m)
                        s += v(yi, m, r) * v(diagonalFactor_i, m, c);
                    v(lki, r, c) = s;
                }
        }
    }

    /**
     * Store the diagonal factor of k from the pivot block D(k) (see the documentation of the struct).
     * Return false if the pivot is singular.
     */
    static inline bool setDiagonalFactor(Block& diagonalFactor_k, const Block& Dk)
    {
        if constexpr (IsScalar)
        {
            diagonalFactor_k = Dk;
            return Dk != 0;
        }
        else
        {
            return invertSymmetric(Dk, diagonalFactor_k);
        }
    }

    /// Called on each diagonal factor at the end of the factorization, so that it contains D^{-1}
    static inline void finalizeDiagonal(Block& diagonalFactor)
    {
        if constexpr (IsScalar)
        {
            diagonalFactor = 1 / diagonalFactor;
        }
    }

    /**
     * Inverse of a dense symmetric block, computed with a dense LDL^T factorization (no pivoting).
     * Only the lower triangular part of A is read.
     * As for the scalar pivots, the only failure case is an exactly zero pivot.
     * matrix_bloc_traits::invert (i.e. type::Mat::invert) is not used on purpose: it considers a
     * matrix as singular based on a threshold on its determinant, whereas the entries of a
     * system matrix can be very small (e.g. a mass matrix with small masses).
     */
    static inline bool invertSymmetric(const Block& A, Block& invA)
    {
        // A = Ld * diag(d) * Ld^T
        Real Ld[N][N] {};
        Real d[N] {};
        for (sofa::SignedIndex j = 0; j < N; ++j)
        {
            Real djj = v(A, j, j);
            for (sofa::SignedIndex k = 0; k < j; ++k)
                djj -= Ld[j][k] * Ld[j][k] * d[k];
            if (djj == 0)
            {
                return false;
            }
            d[j] = djj;

            for (sofa::SignedIndex i = j + 1; i < N; ++i)
            {
                Real lij = v(A, i, j);
                for (sofa::SignedIndex k = 0; k < j; ++k)
                    lij -= Ld[i][k] * Ld[j][k] * d[k];
                Ld[i][j] = lij / djj;
            }
        }

        // W = Ld^{-1} (unit lower triangular)
        Real W[N][N] {};
        for (sofa::SignedIndex j = 0; j < N; ++j)
        {
            W[j][j] = 1;
            for (sofa::SignedIndex i = j + 1; i < N; ++i)
            {
                Real s = 0;
                for (sofa::SignedIndex k = j; k < i; ++k)
                    s -= Ld[i][k] * W[k][j];
                W[i][j] = s;
            }
        }

        // A^{-1} = W^T * diag(1/d) * W
        for (sofa::SignedIndex i = 0; i < N; ++i)
            for (sofa::SignedIndex j = i; j < N; ++j)
            {
                Real s = 0;
                // W[k][i] == 0 for k < i, and W[k][j] == 0 for k < j (with j >= i)
                for (sofa::SignedIndex k = j; k < N; ++k)
                    s += W[k][i] * W[k][j] / d[k];
                v(invA, i, j) = s;
                v(invA, j, i) = s;
            }
        return true;
    }
};

} // namespace sofa::component::linearsolver::direct
