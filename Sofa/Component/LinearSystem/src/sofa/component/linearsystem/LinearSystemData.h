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
#include <memory>
#include <sofa/config.h>

namespace sofa::component::linearsystem
{

template<typename TMatrix, typename TVector>
struct LinearSystemData
{
    using MatrixType = TMatrix;
    using VectorType = TVector;

    using StoredMatrixType = std::unique_ptr<MatrixType>;
    using StoredVectorType = std::unique_ptr<VectorType>;


    /// The global matrix of the linear system. If the linear system is written as Ax=b, it is A.
    StoredMatrixType matrix { nullptr };

    /// The right-hand side of the linear system. If the linear system is written as Ax=b, it is b.
    StoredVectorType rhs { nullptr };

    /// The solution of the linear system. If the linear system is written as Ax=b, it is x.
    StoredVectorType solution { nullptr };

    [[nodiscard]] MatrixType* getMatrix() const { return matrix.get(); }
    [[nodiscard]] VectorType* getRHS() const { return rhs.get(); }
    [[nodiscard]] VectorType* getSolution() const { return solution.get(); }

    void allocateSystem();

    /// Allocate the object for the global matrix
    void createSystemMatrix();

    /// Allocate the object for the RHS
    void createSystemRHSVector();

    /// Allocate the object for the solution
    void createSystemSolutionVector();

    void resizeSystem(sofa::Size n);

    void clearSystem();
};

template <typename TMatrix, typename TVector>
void LinearSystemData<TMatrix, TVector>::allocateSystem()
{
    if (!this->matrix)
    {
        this->createSystemMatrix();
    }
    if (!this->rhs)
    {
        this->createSystemRHSVector();
    }
    if (!this->solution)
    {
        this->createSystemSolutionVector();
    }
}

template <typename TMatrix, typename TVector>
void LinearSystemData<TMatrix, TVector>::createSystemMatrix()
{
    matrix = std::make_unique<TMatrix>();
}

template <typename TMatrix, typename TVector>
void LinearSystemData<TMatrix, TVector>::createSystemRHSVector()
{
    rhs = std::make_unique<TVector>();
}

template <typename TMatrix, typename TVector>
void LinearSystemData<TMatrix, TVector>::createSystemSolutionVector()
{
    solution = std::make_unique<TVector>();
}

template <typename TMatrix, typename TVector>
void LinearSystemData<TMatrix, TVector>::resizeSystem(sofa::Size n)
{
    allocateSystem();

    if (matrix)
    {
        matrix->resize(n, n);
    }

    if (rhs)
    {
        rhs->resize(n);
    }

    if (solution)
    {
        solution->resize(n);
    }
}

template <typename TMatrix, typename TVector>
void LinearSystemData<TMatrix, TVector>::clearSystem()
{
    allocateSystem();

    if (matrix)
    {
        matrix->clear();
    }

    if (rhs)
    {
        rhs->clear();
    }

    if (solution)
    {
        solution->clear();
    }
}
}
