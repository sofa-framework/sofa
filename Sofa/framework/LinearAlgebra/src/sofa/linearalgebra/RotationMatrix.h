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
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/type/vector.h>

namespace sofa::linearalgebra
{

/// Direct linear solver based on Sparse LDL^T factorization, implemented with the CSPARSE library
template<class TReal>
class SOFA_LINEARALGEBRA_API RotationMatrix : public linearalgebra::BaseMatrix
{
public:
    typedef TReal Real;

    sofa::SignedIndex rowSize(void) const override;

    /// Number of columns
    sofa::SignedIndex colSize(void) const override;

    /// Read the value of the element at row i, column j (using 0-based indices)
    SReal element(Index i, sofa::SignedIndex j) const override;

    /// Resize the matrix and reset all values to 0
    void resize(sofa::SignedIndex nbRow, sofa::SignedIndex nbCol) override;

    /// Reset all values to 0
    void clear() override;

    virtual void setIdentity();

    /// Write the value of the element at row i, column j (using 0-based indices)
    void set(sofa::SignedIndex i, sofa::SignedIndex j, double v) override;

    using BaseMatrix::add;

    /// Add v to the existing value of the element at row i, column j (using 0-based indices)
    void add(sofa::SignedIndex i, sofa::SignedIndex j, double v) override;

    virtual type::vector<Real> & getVector();

    /// Deprecated (see PR #3335). Replaced by mulVector
    void opMulV(linearalgebra::BaseVector* result, const linearalgebra::BaseVector* v) const final;
    /// Deprecated (see PR #3335). Replaced by mulTransposeVector
    void opMulTV(linearalgebra::BaseVector* result, const linearalgebra::BaseVector* v) const final;
    /// Deprecated (see PR #3335). Replaced by mulTransposeMatrix
    void opMulTM(linearalgebra::BaseMatrix * bresult,linearalgebra::BaseMatrix * bm) const final;


    void mulVector(linearalgebra::BaseVector* result, const linearalgebra::BaseVector* v) const;
    void mulTransposeVector(linearalgebra::BaseVector* result, const linearalgebra::BaseVector* v) const;

    /// multiply the transpose current matrix by m matrix and strore the result in result
    void mulTransposeMatrix(RotationMatrix<Real>* result, RotationMatrix<Real>* m) const;

    void rotateMatrix(linearalgebra::BaseMatrix * mat,const linearalgebra::BaseMatrix * Jmat);

    static const char* Name();

protected :
    type::vector<Real> data;
};

template<class Real>
std::ostream& operator << (std::ostream& out, const RotationMatrix<Real> & v );

template<> const char* RotationMatrix<float>::Name();
template<> const char* RotationMatrix<double>::Name();

#if !defined(SOFA_SOFABASELINEARSOLVER_ROTATIONMATRIX_DEFINITION)
extern template class RotationMatrix<float>;
extern template class RotationMatrix<double>;
#endif ///

} // namespace sofa::component::solver
