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
#include <Eigen/Dense>
#include <sofa/type/MatSym.h>
#include <gtest/gtest.h>
#include <sofa/testing/LinearCongruentialRandomGenerator.h>
#include <sofa/testing/NumericTest.h>


namespace sofa
{

template <int D, class _Real = SReal>
struct MatSymTestParameterPack
{
    static constexpr auto Size = D;
    using Real = _Real;
};

template <class ParameterPack>
class MatSymTest : public testing::NumericTest<typename ParameterPack::Real>
{
public:
    using Real = typename ParameterPack::Real;
    static constexpr auto Size = ParameterPack::Size;

    void onSetUp() override
    {
        sofa::testing::LinearCongruentialRandomGenerator lcg(96547);

        for (sofa::Size i = 0; i < sofa::type::MatSym<Size, Real>::size(); ++i)
        {
            m_symmetricMatrix[i] = lcg.generateInRange(-10., 10.);
        }
    }

    void inversion() const
    {
        type::MatSym<Size, Real> M_inverse;
        sofa::type::invertMatrix(M_inverse, m_symmetricMatrix);

        const Eigen::Matrix<Real, Size, Size> W = convert<Eigen::Matrix<Real, Size, Size>>().inverse();

        for (int i = 0; i < Size; ++i)
        {
            for (int j = 0; j < Size; ++j)
            {
                EXPECT_NEAR(W(i, j), M_inverse(i, j), testing::NumericTest<Real>::epsilon());
            }
        }
    }

    void rightProduct() const
    {
        const auto other = getRandomMatrix<sofa::type::Mat<Size, Size, Real>>();

        const auto product = m_symmetricMatrix * other;
        const auto genericProduct = convert<sofa::type::Mat<Size, Size, Real>>() * other;

        EXPECT_EQ(product, genericProduct);
    }

    void leftProduct() const
    {
        const auto other = getRandomMatrix<sofa::type::Mat<Size, Size, Real>>();

        const auto product = other * m_symmetricMatrix;
        const auto genericProduct = other * convert<sofa::type::Mat<Size, Size, Real>>();

        EXPECT_EQ(product, genericProduct);
    }

    void trace() const
    {
        const Real expectedTrace = convert<Eigen::Matrix<Real, Size, Size>>().trace();
        EXPECT_NEAR(
            expectedTrace,
            sofa::type::trace(m_symmetricMatrix),
            testing::NumericTest<Real>::epsilon());
    }

protected:
    sofa::type::MatSym<Size, Real> m_symmetricMatrix;

    template<class MatrixType>
    MatrixType convert() const
    {
        MatrixType result;
        for (int i = 0; i < Size; ++i)
        {
            for (int j = 0; j < Size; ++j)
            {
                result(i, j) = m_symmetricMatrix(i, j);
            }
        }
        return result;
    }

    template<class MatrixType>
    static MatrixType getRandomMatrix()
    {
        MatrixType randomMatrix;

        sofa::testing::LinearCongruentialRandomGenerator lcg(783352);

        for (sofa::Size i = 0; i < Size; ++i)
        {
            for (sofa::Size j = 0; j < Size; ++j)
            {
                randomMatrix(i, j) = lcg.generateInRange(-10., 10.);
            }
        }

        return randomMatrix;
    }
};

using ::testing::Types;
typedef Types<
    MatSymTestParameterPack<2, SReal>,
    MatSymTestParameterPack<3, SReal>
> DataTypes;

TYPED_TEST_SUITE(MatSymTest, DataTypes);

TYPED_TEST(MatSymTest, inversion )
{
    ASSERT_NO_THROW (this->inversion());
}

TYPED_TEST(MatSymTest, rightProduct )
{
    ASSERT_NO_THROW (this->rightProduct());
}

TYPED_TEST(MatSymTest, leftProduct)
{
    ASSERT_NO_THROW (this->leftProduct());
}

TYPED_TEST(MatSymTest, trace)
{
    ASSERT_NO_THROW (this->trace());
}


template<class _Real>
class MatSym3x3Test : public MatSymTest<MatSymTestParameterPack<3, _Real>>
{
public:
    void elementAccessor() const
    {
        EXPECT_EQ(this->m_symmetricMatrix(0, 0), this->m_symmetricMatrix[0]);
        EXPECT_EQ(this->m_symmetricMatrix(0, 1), this->m_symmetricMatrix[1]);
        EXPECT_EQ(this->m_symmetricMatrix(1, 1), this->m_symmetricMatrix[2]);
        EXPECT_EQ(this->m_symmetricMatrix(0, 2), this->m_symmetricMatrix[3]);
        EXPECT_EQ(this->m_symmetricMatrix(1, 2), this->m_symmetricMatrix[4]);
        EXPECT_EQ(this->m_symmetricMatrix(2, 2), this->m_symmetricMatrix[5]);
    }
};

TYPED_TEST_SUITE(MatSym3x3Test, Types<SReal>);
TYPED_TEST(MatSym3x3Test, elementAccessor)
{
    ASSERT_NO_THROW (this->elementAccessor());
}

}
