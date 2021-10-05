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

#include <sofa/linearalgebra/config.h>

namespace sofa::linearalgebra
{

/**
 * Given two matrices, compute the product of both matrices
 *
 * To compute the product, the method computeProduct must be called.
 *
 * The computation of the product is accelerated when the profile of both input matrices did not change.
 */
template<class TMatrix>
class SparseMatrixProduct
{
public:
    TMatrix* matrixA { nullptr };
    TMatrix* matrixB { nullptr };

    void computeProduct(bool forceComputeIntersection = true);

    const TMatrix& getProductResult() const { return matrixC; }

    SparseMatrixProduct(TMatrix* a, TMatrix* b) : matrixA(a), matrixB(b) {}
    SparseMatrixProduct() = default;

protected:
    TMatrix matrixC; /// Result of A*B

    bool m_hasComputedIntersection { false };
    void computeIntersection();
    void computeProductFromIntersection();
};

template <class TMatrix>
void SparseMatrixProduct<TMatrix>::computeProduct(bool forceComputeIntersection)
{
    if (forceComputeIntersection)
    {
        m_hasComputedIntersection = false;
    }

    if (m_hasComputedIntersection == false)
    {
        computeIntersection();
        m_hasComputedIntersection = true;
    }
    computeProductFromIntersection();
}

template <class TMatrix>
void SparseMatrixProduct<TMatrix>::computeIntersection()
{
}

template <class TMatrix>
void SparseMatrixProduct<TMatrix>::computeProductFromIntersection()
{
    matrixC = (*matrixA) * (*matrixB);
}

}// sofa::linearalgebra