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
#include <iostream>
#include <sofa/type/Mat.h>
#include <sofa/type/Quat.h>
#include <sofa/testing/NumericTest.h>
#include <sofa/helper/logging/Messaging.h>

using namespace sofa::testing ;


using namespace sofa;
using namespace sofa::type;
using namespace sofa::helper;
using namespace sofa::defaulttype;

TEST(MatTypesTest, initializerListConstructors)
{
    static constexpr sofa::type::Mat<3, 3, int> A {
        {1, 2, 3}, {4, 5, 6}, {7, 8, 9}
    };

    for (sofa::Size i = 0; i<3; ++i)
    {
        for (sofa::Size j = 0; j<3; ++j)
        {
            EXPECT_EQ(A(i, j), i * 3 + j + 1);
        }
    }

    static constexpr sofa::type::Mat<1, 3, int> B { {1, 2, 3} };
    for (sofa::Size j = 0; j<3; ++j)
    {
        EXPECT_EQ(B(0, j), j + 1);
    }

    static constexpr sofa::type::Mat<1, 3, int> C {1, 2, 3};
    for (sofa::Size j = 0; j<3; ++j)
    {
        EXPECT_EQ(C(0, j), j + 1);
    }

    static constexpr sofa::type::Mat<3, 1, int> D {1, 2, 3};
    for (sofa::Size i = 0; i<3; ++i)
    {
        EXPECT_EQ(D(i, 0), i + 1);
    }

    static constexpr sofa::type::Mat<1, 1, int> E {1};
    EXPECT_EQ(E(0, 0), 1);

    const int Evalue = E;
    EXPECT_EQ(Evalue, 1);

    static constexpr sofa::type::Mat<1, 1, int> F {{1}};
    EXPECT_EQ(F(0, 0), 1);

    const int Fvalue = F;
    EXPECT_EQ(Fvalue, 1);
}

TEST(MatTypesTest, lineAccess)
{
    Matrix2 mat2;

    mat2.x();
    mat2.y();
    //mat2.z(); // z is not available due to the size of the matrix

    Matrix3 mat3;

    mat3.x();
    mat3.y();
    mat3.z();
    //mat3.w(); // z is not available due to the size of the matrix
}

TEST(MatTypesTest, mat3x3product)
{
    static constexpr Matrix3 a{ Matrix3::Line{1., 2., 3.}, Matrix3::Line{4., 5., 6.}, Matrix3::Line{7., 8., 9.} };
    static constexpr auto a2 = a * a;

    EXPECT_FLOATINGPOINT_EQ(a2[0][0], 30_sreal)
    EXPECT_FLOATINGPOINT_EQ(a2[0][1], 36_sreal)
    EXPECT_FLOATINGPOINT_EQ(a2[0][2], 42_sreal)

    EXPECT_FLOATINGPOINT_EQ(a2[1][0], 66_sreal)
    EXPECT_FLOATINGPOINT_EQ(a2[1][1], 81_sreal)
    EXPECT_FLOATINGPOINT_EQ(a2[1][2], 96_sreal)

    EXPECT_FLOATINGPOINT_EQ(a2[2][0], 102_sreal)
    EXPECT_FLOATINGPOINT_EQ(a2[2][1], 126_sreal)
    EXPECT_FLOATINGPOINT_EQ(a2[2][2], 150_sreal)
}

TEST(MatTypesTest, multTranspose)
{
    const sofa::type::Mat<3,4, int> a
    {
        sofa::type::Mat<3,4, int>::Line{1, 2, 3, 4},
        sofa::type::Mat<3,4, int>::Line{5, 6, 7, 8},
        sofa::type::Mat<3,4, int>::Line{9, 10, 11, 12}
    };

    const sofa::type::Mat<3,2, int> b
    {
        sofa::type::Mat<3,2, int>::Line{1, 2},
        sofa::type::Mat<3,2, int>::Line{3, 4},
        sofa::type::Mat<3,2, int>::Line{5, 6}
    };

    sofa::type::Mat<4, 2, int> aTb = a.multTranspose(b);

    EXPECT_EQ(aTb[0][0], 61);
    EXPECT_EQ(aTb[0][1], 76);

    EXPECT_EQ(aTb[1][0], 70);
    EXPECT_EQ(aTb[1][1], 88);

    EXPECT_EQ(aTb[2][0], 79);
    EXPECT_EQ(aTb[2][1], 100);

    EXPECT_EQ(aTb[3][0], 88);
    EXPECT_EQ(aTb[3][1], 112);

    const sofa::type::Mat<3, 3, int> c
    {
        sofa::type::Mat<3, 3, int>::Line{1., 2., 3.},
        sofa::type::Mat<3, 3, int>::Line{4., 5., 6.},
        sofa::type::Mat<3, 3, int>::Line{7., 8., 9.}
    };
    sofa::type::Mat<3, 3, int> cTc = c.multTranspose(c);

    EXPECT_EQ(cTc[0][0], 66);
    EXPECT_EQ(cTc[0][1], 78);
    EXPECT_EQ(cTc[0][2], 90);

    EXPECT_EQ(cTc[1][0], 78);
    EXPECT_EQ(cTc[1][1], 93);
    EXPECT_EQ(cTc[1][2], 108);

    EXPECT_EQ(cTc[2][0], 90);
    EXPECT_EQ(cTc[2][1], 108);
    EXPECT_EQ(cTc[2][2], 126);


}

void test_transformInverse(Matrix4 const& M)
{
    Matrix4 M_inv;
    M_inv.transformInvert(M);
    const Matrix4 res = M*M_inv;
    Matrix4 I;I.identity();
    if constexpr (std::is_same_v <SReal, double>)
    {
        EXPECT_MAT_NEAR(I, res, 1e-12_sreal);
    }
    else
    {
        EXPECT_MAT_NEAR(I, res, 1e-6_sreal);
    }
}

TEST(MatTypesTest, transformInverse)
{
    test_transformInverse(Matrix4::Identity());
    test_transformInverse(Matrix4::transformTranslation(Vec3(1.,2.,3.)));
    test_transformInverse(Matrix4::transformScale(Vec3(1.,2.,3.)));
    test_transformInverse(Matrix4::transformRotation(Quat<SReal>::fromEuler(M_PI_4,M_PI_2,M_PI/3.)));
}

TEST(MatTypesTest, setsub_vec)
{
    Matrix3 M = Matrix3::Identity();
    const Vec2 v(1.,2.);
    M.setsub(1,2,v);
    double exp[9]={1.,0.,0.,
                   0.,1.,1.,
                   0.,0.,2.};
    const Matrix3 M_exp(exp);
    EXPECT_MAT_DOUBLE_EQ(M_exp, M);
}

TEST(MatTypesTest, isTransform)
{
    Matrix4 M;
    EXPECT_FALSE(M.isTransform());
    M.identity();
    EXPECT_TRUE(M.isTransform());
}

TEST(MatTypesTest, transpose)
{
    Matrix4 M(Matrix4::Line(16, 2, 3, 13), Matrix4::Line(5, 11, 10, 8), Matrix4::Line(9, 7, 6, 12),
              Matrix4::Line(4, 14, 15, 1));

    Matrix4 Mnew;
    Mnew.transpose(M);

    const Matrix4 Mtest(Matrix4::Line(16, 5, 9, 4), Matrix4::Line(2, 11, 7, 14), Matrix4::Line(3, 10, 6, 15),
                        Matrix4::Line(13, 8, 12, 1));

    EXPECT_EQ(Mnew, Mtest);
    EXPECT_EQ(M.transposed(), Mtest);

    M.transpose(M);
    EXPECT_EQ(M, Mtest);

    M = Matrix4(Matrix4::Line(16, 2, 3, 13), Matrix4::Line(5, 11, 10, 8), Matrix4::Line(9, 7, 6, 12),
              Matrix4::Line(4, 14, 15, 1));

    M.transpose();
    EXPECT_EQ(M, Mtest);

    M.identity();
    EXPECT_EQ(M.transposed(), M);
}

TEST(MatTypesTest, nonSquareTranspose)
{
    const Mat<3,4,double> M(Matrix4::Line(16, 2, 3, 13), Matrix4::Line(5, 11, 10, 8), Matrix4::Line(9, 7, 6, 12));

    Mat<4,3,double> Mnew;
    Mnew.transpose(M);

    const Mat<4,3,double> Mtest(Matrix3::Line(16,5,9), Matrix3::Line(2,11,7), Matrix3::Line(3,10,6), Matrix3::Line(13,8,12));

    EXPECT_EQ(Mnew, Mtest);
    EXPECT_EQ(M.transposed(), Mtest);
    EXPECT_EQ(M, Mtest.transposed());
}

TEST(MatTypesTest, invert22)
{
    Matrix2 M(Matrix2::Line(4.0, 7.0), Matrix2::Line(2.0, 6.0));
    Matrix2 Minv;
    const Matrix2 Mtest(Matrix2::Line(0.6,-0.7),
                        Matrix2::Line(-0.2,0.4));

    {
        const bool success = type::invertMatrix(Minv, M);
        EXPECT_TRUE(success);
        EXPECT_EQ(Minv, Mtest);
    }

    EXPECT_EQ(M.inverted(), Mtest);

    {
        const bool success = Minv.invert(M);
        EXPECT_TRUE(success);
        EXPECT_EQ(Minv, Mtest);
    }

    {
        const bool success = M.invert(M);
        EXPECT_TRUE(success);
        EXPECT_EQ(M, Mtest);
    }
}

TEST(MatTypesTest, invert33)
{
    Matrix3 M(Matrix3::Line(3., 0., 2.), Matrix3::Line(2., 0., -2.), Matrix3::Line(0., 1., 1.));
    Matrix3 Minv;
    const Matrix3 Mtest(Matrix3::Line(0.2, 0.2, 0.),
                        Matrix3::Line(-0.2, 0.3, 1.),
                        Matrix3::Line(0.2, -0.3, 0.));

    {
        const bool success = type::invertMatrix(Minv, M);
        EXPECT_TRUE(success);
        EXPECT_EQ(Minv, Mtest);
    }

    EXPECT_EQ(M.inverted(), Mtest);

    {
        const bool success = Minv.invert(M);
        EXPECT_TRUE(success);
        EXPECT_EQ(Minv, Mtest);
    }

    {
        const bool success = M.invert(M);
        EXPECT_TRUE(success);
        EXPECT_EQ(M, Mtest);
    }
}

TEST(MatTypesTest, invert55)
{
    Mat<5, 5, SReal> M(Mat<5, 5, SReal>::Line(-2.,  7.,  0.,  6., -2.),
                       Mat<5, 5, SReal>::Line( 1., -1.,  3.,  2.,  2.),
                       Mat<5, 5, SReal>::Line( 3.,  4.,  0.,  5.,  3.),
                       Mat<5, 5, SReal>::Line( 2.,  5., -4., -2.,  2.),
                       Mat<5, 5, SReal>::Line( 0.,  3., -1.,  1., -4.));
    Mat<5, 5, SReal> Minv;

    const Mat<5, 5, SReal> Mtest(Mat<5, 5, SReal>::Line(-289./1440., 11./90., 13./90., 31./1440., 101./360.),
                                 Mat<5, 5, SReal>::Line(37./360., 14./45., -8./45., 77./360., 7./90.),
                                 Mat<5, 5, SReal>::Line(17./288., 11./18., -5./18., 49./288., 11./72.),
                                 Mat<5, 5, SReal>::Line(1./1440., -29./90.,23./90.,-319./1440.,-29./360.),
                                 Mat<5, 5, SReal>::Line(1./16., 0., 0., 1./16., -1./4.));

    {
        const bool success = type::invertMatrix(Minv, M);
        EXPECT_TRUE(success);
        EXPECT_EQ(Minv, Mtest);
    }

    EXPECT_EQ(M.inverted(), Mtest);

    {
        const bool success = Minv.invert(M);
        EXPECT_TRUE(success);
        EXPECT_EQ(Minv, Mtest);
    }

    {
        const bool success = M.invert(M);
        EXPECT_TRUE(success);
        EXPECT_EQ(M, Mtest);
    }
}

TEST(MatTypesTest, tensorProduct)
{
    const Vec<2,SReal> v1(0.,1.), v2(1.,2.);
    const Mat<2, 2, SReal> Mtest = tensorProduct(v1,v2);

    const Mat<2, 2, SReal> M(Mat<2, 2, SReal>::Line(0.,  0.),
                             Mat<2, 2, SReal>::Line( 1., 2.));
    EXPECT_EQ(M, Mtest);
}

TEST(MatTypesTest, identity)
{
    const sofa::type::Mat<3, 3, SReal>& id = sofa::type::Mat<3, 3, SReal>::Identity();

    for (sofa::Index i = 0; i < 3; ++i)
    {
        for (sofa::Index j = 0; j < 3; ++j)
        {
            EXPECT_FLOATINGPOINT_EQ(id(i,j), static_cast<SReal>(i == j))
        }
    }
}

TEST(MatTypesTest, conversionToReal)
{
    const sofa::type::Mat<1, 1, SReal>& id = sofa::type::Mat<1, 1, SReal>::Identity();

    const SReal r = id;
    EXPECT_EQ(r, 1_sreal);

    const SReal p = id.toReal();
    EXPECT_EQ(p, 1_sreal);
}
