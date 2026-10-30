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
#include <sofa/testing/config.h>

#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/data/Data[Eigen].h>
#include <sofa/testing/BaseTest.h>
#include <sstream>
using sofa::testing::BaseTest ;

namespace sofa {

using namespace core::objectmodel;

class DataEigen_test : public BaseTest
{
public:
    // No default value needed: the dimensions are read from the string and
    // dynamic matrices are resized accordingly.
    Data<Eigen::MatrixXf> dataMatrixXf;
    Data<Eigen::MatrixXd> dataMatrixXd;
    Data<Eigen::MatrixXi> dataMatrixXi;
    Data<Eigen::Matrix3d> dataMatrix3d;
};

TEST_F(DataEigen_test, read_dynamicFloatMatrix)
{
    EXPECT_TRUE(dataMatrixXf.read("2 2 1 2 3 4"));
    const Eigen::MatrixXf& m = dataMatrixXf.getValue();
    ASSERT_EQ(m.rows(), 2);
    ASSERT_EQ(m.cols(), 2);
    EXPECT_FLOAT_EQ(m(0, 0), 1.f);
    EXPECT_FLOAT_EQ(m(0, 1), 2.f);
    EXPECT_FLOAT_EQ(m(1, 0), 3.f);
    EXPECT_FLOAT_EQ(m(1, 1), 4.f);
}

TEST_F(DataEigen_test, read_dynamicDoubleMatrix)
{
    EXPECT_TRUE(dataMatrixXd.read("2 3 1.5 2.5 3.5 4.5 5.5 6.5"));
    const Eigen::MatrixXd& m = dataMatrixXd.getValue();
    ASSERT_EQ(m.rows(), 2);
    ASSERT_EQ(m.cols(), 3);
    EXPECT_DOUBLE_EQ(m(0, 0), 1.5);
    EXPECT_DOUBLE_EQ(m(0, 2), 3.5);
    EXPECT_DOUBLE_EQ(m(1, 0), 4.5);
    EXPECT_DOUBLE_EQ(m(1, 2), 6.5);
}

TEST_F(DataEigen_test, read_integerMatrix)
{
    EXPECT_TRUE(dataMatrixXi.read("2 2 1 2 3 4"));
    const Eigen::MatrixXi& m = dataMatrixXi.getValue();
    ASSERT_EQ(m.rows(), 2);
    ASSERT_EQ(m.cols(), 2);
    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m(0, 1), 2);
    EXPECT_EQ(m(1, 0), 3);
    EXPECT_EQ(m(1, 1), 4);
}

TEST_F(DataEigen_test, read_fixedSizeMatrix)
{
    EXPECT_TRUE(dataMatrix3d.read("3 3 1 2 3 4 5 6 7 8 9"));
    const Eigen::Matrix3d& m = dataMatrix3d.getValue();
    EXPECT_DOUBLE_EQ(m(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(m(2, 2), 9.0);
}

TEST_F(DataEigen_test, read_resizesPreviousValue)
{
    Data<Eigen::MatrixXf> data{Eigen::MatrixXf::Zero(3, 3)};
    EXPECT_TRUE(data.read("2 2 1 2 3 4"));
    const Eigen::MatrixXf& m = data.getValue();
    ASSERT_EQ(m.rows(), 2);
    ASSERT_EQ(m.cols(), 2);
    EXPECT_FLOAT_EQ(m(1, 1), 4.f);
}

TEST_F(DataEigen_test, read_fixedSizeDimensionMismatch_returnsFalse)
{
    EXPECT_FALSE(dataMatrix3d.read("2 2 1 2 3 4"));
}

TEST_F(DataEigen_test, read_invalidString_returnsFalse)
{
    EXPECT_FALSE(dataMatrixXf.read("2 2 1 2 abc 4"));
}

TEST_F(DataEigen_test, read_negativeDimensions_returnsFalse)
{
    EXPECT_FALSE(dataMatrixXf.read("-1 2 1 2"));
}

TEST_F(DataEigen_test, getValueString_format)
{
    ASSERT_TRUE(dataMatrixXd.read("2 3 1.5 2.5 3.5 4.5 5.5 6.5"));
    EXPECT_EQ(dataMatrixXd.getValueString(), "2 3 1.5 2.5 3.5 4.5 5.5 6.5");
}

TEST_F(DataEigen_test, getValueString_roundTrip)
{
    ASSERT_TRUE(dataMatrixXd.read("2 3 1.5 2.5 3.5 4.5 5.5 6.5"));
    const std::string str = dataMatrixXd.getValueString();
    EXPECT_FALSE(str.empty());

    Data<Eigen::MatrixXd> copy;
    ASSERT_TRUE(copy.read(str));
    ASSERT_EQ(copy.getValue().rows(), 2);
    ASSERT_EQ(copy.getValue().cols(), 3);
    EXPECT_TRUE(copy.getValue().isApprox(dataMatrixXd.getValue()));
}

TEST_F(DataEigen_test, getValueString_fixedSizeMatrix)
{
    ASSERT_TRUE(dataMatrix3d.read("3 3 1 2 3 4 5 6 7 8 9"));

    Data<Eigen::Matrix3d> copy;
    ASSERT_TRUE(copy.read(dataMatrix3d.getValueString()));
    EXPECT_TRUE(copy.getValue().isApprox(dataMatrix3d.getValue()));
}

TEST_F(DataEigen_test, eigenStreamFormat_isUnchanged)
{
    // Data[Eigen].h must not add any operator<< to namespace Eigen:
    // streaming a matrix directly keeps Eigen's own format.
    ASSERT_TRUE(dataMatrix3d.read("3 3 1 2 3 4 5 6 7 8 9"));
    std::ostringstream out;
    out << dataMatrix3d.getValue();
    EXPECT_EQ(out.str(), "1 2 3\n4 5 6\n7 8 9");
}

}
