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
#include <gtest/gtest.h>
#include <sofa/testing/NumericTest.h>
#include <sofa/linearalgebra/BaseMatrix.h>

namespace sofa::linearalgebra::testing
{

template<class TMatrix, sofa::Index TNbRows, sofa::Index TNbCols, class TReal >
struct TestBaseMatrixTraits
{
    using Matrix = TMatrix;
    static constexpr sofa::Index NbRows = TNbRows;
    static constexpr sofa::Index NbCols = TNbCols;
    using Real = TReal;
};

/// Type-parameterized tests for classes derived from BaseMatrix
template<class T>
class TestBaseMatrix : public sofa::testing::NumericTest<typename T::Real>
{
public:
    using Matrix = typename T::Matrix;
    using Real = typename T::Real;
    using Inherit = sofa::testing::NumericTest<typename T::Real>;

    static constexpr sofa::Index NbRows = T::NbRows;
    static constexpr sofa::Index NbCols = T::NbCols;

    void onSetUp() override
    {
        m_testedMatrix = std::make_unique<Matrix>();
        m_testedMatrix->resize(NbRows, NbCols);

        m_modelMatrix.clear();
    }

    void onTearDown() override
    {
        m_testedMatrix.reset();
    }

    void checkResize()
    {
        EXPECT_EQ(m_testedMatrix->rowSize(), NbRows);
        EXPECT_EQ(m_testedMatrix->rows(), NbRows);
        EXPECT_EQ(m_modelMatrix.getNbLines(), NbRows);

        EXPECT_EQ(m_testedMatrix->colSize(), NbCols);
        EXPECT_EQ(m_testedMatrix->cols(), NbCols);
        EXPECT_EQ(m_modelMatrix.getNbCols(), NbCols);
    }

    void checkAddScalar()
    {
        m_testedMatrix->clear();

        Real value = (Real)0;
        for (sofa::Index i = 0 ; i < NbRows; ++i)
        {
            for (sofa::Index j = 0 ; j < NbCols; ++j)
            {
                m_modelMatrix(i, j) = ++value;
                m_testedMatrix->add(i, j, value);
            }
        }

        m_testedMatrix->compress(); //this operation is not required for all types of BaseMatrix

        EXPECT_LT(Inherit::matrixMaxDiff(m_modelMatrix, *m_testedMatrix), 100 * Inherit::epsilon());
    }

    /// A 3x3 matrix is added into the matrix using the corresponding function overload
    /// This assumes the matrix is big enough to contain a 3x3 matrix at the requested position
    /// @param posRow row index at which the 3x3 matrix is added
    /// @param posCol column index at which the 3x3 matrix is added
    void checkAddBloc(sofa::linearalgebra::BaseMatrix::Index posRow, sofa::linearalgebra::BaseMatrix::Index posCol)
    {
        m_testedMatrix->clear();

        sofa::type::Mat<3, 3, Real> mat;
        Real value = (Real)0;
        for (typename decltype(mat)::Size i = 0 ; i < decltype(mat)::nbLines; ++i)
        {
            for (typename decltype(mat)::Size j = 0 ; j < decltype(mat)::nbCols; ++j)
            {
                mat(i, j) = ++value;
            }
        }

        m_testedMatrix->add(posRow, posCol, mat);
        m_testedMatrix->compress();

        for (sofa::linearalgebra::BaseMatrix::Index i = 0; i < m_testedMatrix->rowSize(); ++i)
        {
            for (sofa::linearalgebra::BaseMatrix::Index j = 0; j < m_testedMatrix->colSize(); ++j)
            {
                if ( i >= posRow && i < posRow + (sofa::linearalgebra::BaseMatrix::Index)decltype(mat)::nbLines
                  && j >= posCol && j < posCol + (sofa::linearalgebra::BaseMatrix::Index)decltype(mat)::nbCols)
                {
                    EXPECT_NEAR(m_testedMatrix->operator()(i,j), mat(i-posRow,j-posCol), Inherit::epsilon())
                        << "i = " << i << ", j = " << j << ", posRow = " << posRow << ", posCol = " << posCol << "\n"
                        << "mat = " << mat << "\n"
                        << "M = " << *m_testedMatrix;
                }
                else
                {
                    EXPECT_NEAR(m_testedMatrix->operator()(i,j), 0, 100 * Inherit::epsilon())
                        << "i = " << i << ", j = " << j << ", posRow = " << posRow << ", posCol = " << posCol << "\n"
                        << "mat = " << mat << "\n"
                        << "M = " << *m_testedMatrix;
                }
            }
        }
    }

protected:

    sofa::type::Mat<NbRows, NbCols, Real> m_modelMatrix;
    std::unique_ptr<sofa::linearalgebra::BaseMatrix> m_testedMatrix {nullptr};
};

TYPED_TEST_SUITE_P(TestBaseMatrix);

TYPED_TEST_P(TestBaseMatrix, resize)
{
    this->checkResize();
}

TYPED_TEST_P(TestBaseMatrix, addScalar)
{
    this->checkAddScalar();
}

TYPED_TEST_P(TestBaseMatrix, addBloc)
{
    this->checkAddBloc(0, 0);
    this->checkAddBloc(1, 1);
    this->checkAddBloc(1, 0);
    this->checkAddBloc(0, 2);
}

REGISTER_TYPED_TEST_SUITE_P(TestBaseMatrix,
                            resize, addScalar, addBloc
);

} //namespace sofa::linearalgebra::testing
