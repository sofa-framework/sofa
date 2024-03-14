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

template<typename TReal>
class SparseMatrix;

template<typename TReal>
class RotationMatrix;

template<class Real>
std::ostream& operator << (std::ostream& out, const RotationMatrix<Real> & v );

/**
 * \brief 3x3 block-diagonal matrix where each block is considered as a rotation.
 * \tparam TReal Type of scalar
 *
 * One of the feature of this class is to rotate another matrix: if M is a matrix and R is a
 * rotation matrix, it can compute M * R.
 */
template<class TReal>
class RotationMatrix : public linearalgebra::BaseMatrix
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

    void opMulV(linearalgebra::BaseVector* result, const linearalgebra::BaseVector* v) const override;
    void opMulTV(linearalgebra::BaseVector* result, const linearalgebra::BaseVector* v) const override;

    /// multiply the transpose current matrix by m matrix and strore the result in m
    void opMulTM(linearalgebra::BaseMatrix * bresult,linearalgebra::BaseMatrix * bm) const override;

    void rotateMatrix(linearalgebra::BaseMatrix * mat,const linearalgebra::BaseMatrix * Jmat);

    static const char* Name();

    friend std::ostream& operator << <Real> (std::ostream& out, const RotationMatrix<Real> & v );

protected :
    type::vector<Real> data;

    template<typename real2>
    void rotateSparseMatrix(
        linearalgebra::BaseMatrix * result,
        const SparseMatrix<real2>* Jmat);
};

template<> SOFA_LINEARALGEBRA_API const char* RotationMatrix<float>::Name();
template<> SOFA_LINEARALGEBRA_API const char* RotationMatrix<double>::Name();

#if !defined(SOFA_SOFABASELINEARSOLVER_ROTATIONMATRIX_DEFINITION)
extern template class SOFA_LINEARALGEBRA_API RotationMatrix<float>;
extern template class SOFA_LINEARALGEBRA_API RotationMatrix<double>;

extern template SOFA_LINEARALGEBRA_API const char* RotationMatrix<float>::Name();
extern template SOFA_LINEARALGEBRA_API const char* RotationMatrix<double>::Name();

extern template SOFA_LINEARALGEBRA_API std::ostream& operator << (std::ostream& out, const RotationMatrix<float>& v);
extern template SOFA_LINEARALGEBRA_API std::ostream& operator << (std::ostream& out, const RotationMatrix<double>& v);
#endif

} // namespace sofa::component::solver
