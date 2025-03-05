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
#include <sofa/type/Quat.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <ctime>
#include <sofa/testing/LinearCongruentialRandomGenerator.h>

using sofa::type::Quat;

double errorThreshold = 1e-6;

TEST(QuaterTest, EulerAngles)
{
    sofa::testing::LinearCongruentialRandomGenerator lcg(46515387);

    // Try to tranform a Quater (q0) to Euler angles and then back to a Quater (q1)
    // Compare the result of a rotation defined by q0 and q1 on a vector
    for (int i = 0; i < 100; ++i)
    {
        // Generate random Quater and avoid singular values
        Quat<double> q0(
                lcg.generateInRange(-1., 1.),
                lcg.generateInRange(-1., 1.),
                lcg.generateInRange(-1., 1.),
                lcg.generateInRange(-1., 1.));
        q0.normalize();

        for(sofa::Size j = 0 ; j < Quat<double>::size() ; ++j)
        {
            ASSERT_FALSE( std::isnan(q0[j]) );
        }

        const sofa::type::Vec3d eulerAngles = q0.toEulerVector();

        // make sure the angles don't lead to a singularity
        for (const auto& angle : eulerAngles)
        {
            ASSERT_GE(std::abs(angle - M_PI / 2), 1e-3);
            ASSERT_GE(std::abs(angle + M_PI / 2), 1e-3);
        }

        // Transform Euler angles back to a quaternion (q1)
        Quat<double> q1 = Quat<double>::createQuaterFromEuler(eulerAngles);

        for(sofa::Size j = 0 ; j < Quat<double>::size() ; ++j)
        {
            ASSERT_FALSE( std::isnan(q1[j]) );
        }

        // Compute a random rotation with each Quater
        sofa::type::Vec<3,double> p(
            lcg.generateInRange(1., 2.),
            lcg.generateInRange(1., 2.),
            lcg.generateInRange(1., 2.));
        sofa::type::Vec<3,double> p0 = q0.rotate(p);
        sofa::type::Vec<3,double> p1 = q1.rotate(p);
        // Compare the result of the two rotations
        EXPECT_EQ(p0, p1);

        // Specific check for a certain value of p
        sofa::type::Vec<3, double> p2(2, 1, 1);
        p0 = q0.rotate(p2);
        p1 = q1.rotate(p2);
        EXPECT_EQ(p0, p1);
    }
}

/* Following unit test results have been checked with Matlab 2016a.
 * Note: ambiguous result with rotate and invrotate.
 */


TEST(QuaterTest, QuaterdSet)
{
    Quat<double> quat;
    quat.set(0.0, 0.0, 0.0, 1.0);

    EXPECT_DOUBLE_EQ(0.0, quat[0]);
    EXPECT_DOUBLE_EQ(0.0, quat[1]);
    EXPECT_DOUBLE_EQ(0.0, quat[2]);
    EXPECT_DOUBLE_EQ(1.0, quat[3]);
}

TEST(QuaterTest, QuaterdIdentity)
{
    Quat<double> id;
    id.set(0.0, 0.0, 0.0, 1.0);

    Quat<double> quat = Quat<double>::identity();
    EXPECT_DOUBLE_EQ(id[0], quat[0]);
    EXPECT_DOUBLE_EQ(id[1], quat[1]);
    EXPECT_DOUBLE_EQ(id[2], quat[2]);
    EXPECT_DOUBLE_EQ(id[3], quat[3]);
}

TEST(QuaterTest, QuaterdConstPtr)
{
    Quat<double> quat;
    quat.set(0.0, 0.0, 0.0, 1.0);

    const double* quatptr = quat.ptr();

    EXPECT_DOUBLE_EQ(0.0, quatptr[0]);
    EXPECT_DOUBLE_EQ(0.0, quatptr[1]);
    EXPECT_DOUBLE_EQ(0.0, quatptr[2]);
    EXPECT_DOUBLE_EQ(1.0, quatptr[3]);
}

TEST(QuaterTest, QuaterdPtr)
{
    Quat<double> quat;
    quat.set(0.0, 0.0, 0.0, 1.0);

    double* quatptr = quat.ptr();

    EXPECT_DOUBLE_EQ(0.0, quatptr[0]);
    EXPECT_DOUBLE_EQ(0.0, quatptr[1]);
    EXPECT_DOUBLE_EQ(0.0, quatptr[2]);
    EXPECT_DOUBLE_EQ(1.0, quatptr[3]);

    quatptr[0] = 1.0;
    EXPECT_NE(0.0, quatptr[0]);
    EXPECT_DOUBLE_EQ(1.0, quatptr[0]);
}

TEST(QuaterTest, QuaterdNormalize)
{
    Quat<double> quat;
    quat.set(1.0, 0.0, 1.0, 0.0);

    quat.normalize();

    EXPECT_NEAR(0.707106781186548, quat[0], errorThreshold);
    EXPECT_NEAR(0.0  ,  quat[1], errorThreshold);
    EXPECT_NEAR(0.707106781186548, quat[2], errorThreshold);
    EXPECT_NEAR(0.0   , quat[3], errorThreshold);
}

TEST(QuaterTest, QuaterdClear)
{
    Quat<double> quat;
    quat.set(1.0, 2.0, 3.0, 4.0);

    quat.clear();
    EXPECT_NE(1.0, quat[0]);
    EXPECT_NE(2.0, quat[1]);
    EXPECT_NE(3.0, quat[2]);
    EXPECT_NE(4.0, quat[3]);
    EXPECT_NEAR(0.0, quat[0], errorThreshold);
    EXPECT_NEAR(0.0, quat[1], errorThreshold);
    EXPECT_NEAR(0.0, quat[2], errorThreshold);
    EXPECT_NEAR(1.0, quat[3], errorThreshold);
}

TEST(QuaterTest, QuaterdFromFrame)
{
    Quat<double> quat;
    //90 deg around Z from Identity
    const sofa::type::Vec3d xAxis(1.0, 0.0, 0.0);
    const sofa::type::Vec3d yAxis(0.0, 0.0, -1.0);
    const sofa::type::Vec3d zAxis(0.0, 1.0, 0.0);
    quat.fromFrame(xAxis, yAxis, zAxis);

    EXPECT_NEAR(-0.707106781186548, quat[0], errorThreshold);
    EXPECT_NEAR(0.0, quat[1], errorThreshold);
    EXPECT_NEAR(0.0, quat[2], errorThreshold);
    EXPECT_NEAR(0.707106781186548, quat[3], errorThreshold);
}

TEST(QuaterTest, QuaterdFromMatrix)
{
    Quat<double> quat;
    //30deg X, 30deg Y and 30deg Z
    double mat[9]  = { 0.750000000000000, -0.216506350946110, 0.625000000000000,
                       0.433012701892219, 0.875000000000000, -0.216506350946110,
                      -0.500000000000000, 0.433012701892219, 0.750000000000000 };
    quat.fromMatrix(sofa::type::Matrix3(mat));

    EXPECT_NEAR(0.176776695296637, quat[0], errorThreshold);
    EXPECT_NEAR(0.306186217847897, quat[1], errorThreshold);
    EXPECT_NEAR(0.176776695296637, quat[2], errorThreshold);
    EXPECT_NEAR(0.918558653543692, quat[3], errorThreshold);
}

TEST(QuaterTest, QuaterdToMatrix)
{
    Quat<double> quat;
    //60deg X, 30deg Y and 60deg Z
    quat.set(0.306186217847897, 0.435595740399158, 0.306186217847897, 0.789149130992431);
    sofa::type::Mat3x3d mat;
    quat.toMatrix(mat);

    //matlab results
    EXPECT_NEAR(0.433012701892219,  mat[0][0], errorThreshold);
    EXPECT_NEAR(-0.216506350946110,  mat[0][1], errorThreshold);
    EXPECT_NEAR(0.875000000000000, mat[0][2], errorThreshold);
    EXPECT_NEAR(0.750000000000000,  mat[1][0], errorThreshold);
    EXPECT_NEAR(0.625000000000000,  mat[1][1], errorThreshold);
    EXPECT_NEAR(-0.216506350946110,  mat[1][2], errorThreshold);
    EXPECT_NEAR(-0.500000000000000,  mat[2][0], errorThreshold);
    EXPECT_NEAR(0.750000000000000, mat[2][1], errorThreshold);
    EXPECT_NEAR(0.433012701892219, mat[2][2], errorThreshold);
}

TEST(QuaterTest, QuaterdRotateVec)
{
    Quat<double> quat;
    //30deg X, 15deg Y and 30deg Z
    quat.set(0.215229667288440, 0.188196807757208, 0.215229667288440, 0.933774245836654);
    const sofa::type::Vec3d p(3, 6, 9);
    sofa::type::Vec3d rp = quat.rotate(p); //equiv if inverseQuat.rotate() in matlab

    EXPECT_NEAR(4.580932858428164, rp[0], errorThreshold);
    EXPECT_NEAR(3.448650396246470, rp[1], errorThreshold);
    EXPECT_NEAR(9.649967077199914, rp[2], errorThreshold);

    // Compare with previous implementation of Quat::rotate()
    auto previousRotateImpl = [](const Quat<double>& _q, const sofa::type::Vec3d& v)
    {
        return  sofa::type::Vec3d(
            ((1.0f - 2.0f * (_q[1] * _q[1] + _q[2] * _q[2])) * v[0] + (2.0f * (_q[0] * _q[1] - _q[2] * _q[3])) * v[1] + (2.0f * (_q[2] * _q[0] + _q[1] * _q[3])) * v[2]),
            ((2.0f * (_q[0] * _q[1] + _q[2] * _q[3])) * v[0] + (1.0f - 2.0f * (_q[2] * _q[2] + _q[0] * _q[0])) * v[1] + (2.0f * (_q[1] * _q[2] - _q[0] * _q[3])) * v[2]),
            ((2.0f * (_q[2] * _q[0] - _q[1] * _q[3])) * v[0] + (2.0f * (_q[1] * _q[2] + _q[0] * _q[3])) * v[1] + (1.0f - 2.0f * (_q[1] * _q[1] + _q[0] * _q[0])) * v[2])
        );
    };

    const auto& previousRotateRes = previousRotateImpl(quat, p);

    EXPECT_NEAR(0.0, rp[0] - previousRotateRes[0], errorThreshold);
    EXPECT_NEAR(0.0, rp[1] - previousRotateRes[1], errorThreshold);
    EXPECT_NEAR(0.0, rp[2] - previousRotateRes[2], errorThreshold);
}

TEST(QuaterTest, QuaterdInvRotateVec)
{
    Quat<double> quat;
    //30deg X, 15deg Y and 30deg Z
    quat.set(0.215229667288440, 0.188196807757208, 0.215229667288440, 0.933774245836654);
    const sofa::type::Vec3d p(3, 6, 9);
    sofa::type::Vec3d rp = quat.inverseRotate(p);//equiv if quat.rotate() in matlab

    EXPECT_NEAR(3.077954984157941, rp[0], errorThreshold);
    EXPECT_NEAR(8.272072482340949, rp[1], errorThreshold);
    EXPECT_NEAR(6.935344977893669, rp[2], errorThreshold);

    // Compare with previous implementation of Quat::inverseRotate()
    auto previousInverseRotateImpl = [](const Quat<double>& _q, const sofa::type::Vec3d& v)
    {
        return sofa::type::Vec3(
            ((1.0f - 2.0f * (_q[1] * _q[1] + _q[2] * _q[2])) * v[0] + (2.0f * (_q[0] * _q[1] + _q[2] * _q[3])) * v[1] + (2.0f * (_q[2] * _q[0] - _q[1] * _q[3])) * v[2]),
            ((2.0f * (_q[0] * _q[1] - _q[2] * _q[3])) * v[0] + (1.0f - 2.0f * (_q[2] * _q[2] + _q[0] * _q[0])) * v[1] + (2.0f * (_q[1] * _q[2] + _q[0] * _q[3])) * v[2]),
            ((2.0f * (_q[2] * _q[0] + _q[1] * _q[3])) * v[0] + (2.0f * (_q[1] * _q[2] - _q[0] * _q[3])) * v[1] + (1.0f - 2.0f * (_q[1] * _q[1] + _q[0] * _q[0])) * v[2])
        );
    };

    const auto& previousInverseRotateRes = previousInverseRotateImpl(quat, p);

    EXPECT_NEAR(0.0, rp[0] - previousInverseRotateRes[0], errorThreshold);
    EXPECT_NEAR(0.0, rp[1] - previousInverseRotateRes[1], errorThreshold);
    EXPECT_NEAR(0.0, rp[2] - previousInverseRotateRes[2], errorThreshold);
}

TEST(QuaterTest, QuaterdOperatorAdd)
{
    Quat<double> quat1, quat2, quatAdd;
    //30deg X, 15deg Y and 30deg Z
    quat1.set(0.215229667288440, 0.188196807757208, 0.215229667288440, 0.933774245836654);
    //60deg X, 30deg Y and 60deg Z
    quat2.set(0.306186217847897, 0.435595740399158, 0.306186217847897, 0.789149130992431);
    quatAdd = quat1 + quat2;
    const sofa::type::Vec3d p(3, 6, 9);
    sofa::type::Vec3d rp = quatAdd.rotate(p);
    sofa::type::Vec3d rrp = quat2.rotate(quat1.rotate(p));

    //According to the comments, the addition of two quaternions in the compound of the two related rotations

    EXPECT_NEAR(rrp[0], rp[0], errorThreshold);
    EXPECT_NEAR(rrp[1], rp[1], errorThreshold);
    EXPECT_NEAR(rrp[2], rp[2], errorThreshold);
}

TEST(QuaterTest, QuaterdOperatorMultQuat)
{
    Quat<double> quat1, quat2, quatMult;
    //30deg X, 15deg Y and 30deg Z
    quat1.set(0.215229667288440, 0.188196807757208, 0.215229667288440, 0.933774245836654);
    //60deg X, 30deg Y and 60deg Z
    quat2.set(0.306186217847897, 0.435595740399158, 0.306186217847897, 0.789149130992431);
    quatMult = quat1 * quat2;

    EXPECT_NEAR(0.419627252060815, quatMult[0], errorThreshold);
    EXPECT_NEAR(0.555263431278033, quatMult[1], errorThreshold);
    EXPECT_NEAR(0.491886967061105, quatMult[2], errorThreshold);
    EXPECT_NEAR(0.523108691237932, quatMult[3], errorThreshold);
}

TEST(QuaterTest, QuaterdOperatorMultReal)
{
    Quat<double> quat, quatMult;
    const double r = 2.0;
    //30deg X, 15deg Y and 30deg Z
    quat.set(0.215229667288440, 0.188196807757208, 0.215229667288440, 0.933774245836654);

    quatMult = quat * r;

    EXPECT_NEAR(0.430459334576879, quatMult[0], errorThreshold);
    EXPECT_NEAR(0.376393615514416, quatMult[1], errorThreshold);
    EXPECT_NEAR(0.430459334576879, quatMult[2], errorThreshold);
    EXPECT_NEAR(1.867548491673308, quatMult[3], errorThreshold);
}

TEST(QuaterTest, QuaterdOperatorDivReal)
{
    Quat<double> quat, quatDiv;
    const double r = 3.0;
    //60deg X, 30deg Y and 60deg Z
    quat.set(0.306186217847897, 0.435595740399158, 0.306186217847897, 0.789149130992431);

    quatDiv = quat / r;

    EXPECT_NEAR(0.102062072615966, quatDiv[0], errorThreshold);
    EXPECT_NEAR(0.145198580133053, quatDiv[1], errorThreshold);
    EXPECT_NEAR(0.102062072615966, quatDiv[2], errorThreshold);
    EXPECT_NEAR(0.263049710330810, quatDiv[3], errorThreshold);
}

TEST(QuaterTest, QuaterdOperatorMultRealSelf)
{
    Quat<double> quat;
    const double r = 2.0;
    //30deg X, 15deg Y and 30deg Z
    quat.set(0.215229667288440, 0.188196807757208, 0.215229667288440, 0.933774245836654);

    quat *= r;

    EXPECT_NEAR(0.430459334576879, quat[0], errorThreshold);
    EXPECT_NEAR(0.376393615514416, quat[1], errorThreshold);
    EXPECT_NEAR(0.430459334576879, quat[2], errorThreshold);
    EXPECT_NEAR(1.867548491673308, quat[3], errorThreshold);
}

TEST(QuaterTest, QuaterdOperatorDivRealSelf)
{
    Quat<double> quat;
    const double r = 3.0;
    //60deg X, 30deg Y and 60deg Z
    quat.set(0.306186217847897, 0.435595740399158, 0.306186217847897, 0.789149130992431);

    quat /= r;

    EXPECT_NEAR(0.102062072615966, quat[0], errorThreshold);
    EXPECT_NEAR(0.145198580133053, quat[1], errorThreshold);
    EXPECT_NEAR(0.102062072615966, quat[2], errorThreshold);
    EXPECT_NEAR(0.263049710330810, quat[3], errorThreshold);
}

//TEST(QuaterTest, QuaterdQuatVecMult)
//{
//    Quat<double> quat, quatMult;
//    sofa::type::Vec<3, double> vec(1,2,3);
//    //30deg X, 15deg Y and 30deg Z
//    quat.set(0.215229667288440, 0.188196807757208, 0.215229667288440, 0.933774245836654);
//
//    quatMult = quat.quatVectMult(vec);
//}
//
//TEST(QuaterTest, QuaterdVecQuatMult)
//{
//    Quat<double> quat, quatMult;
//    sofa::type::Vec<3, double> vec(1, 2, 3);
//    //30deg X, 15deg Y and 30deg Z
//    quat.set(0.215229667288440, 0.188196807757208, 0.215229667288440, 0.933774245836654);
//
//    quatMult = quat.vectQuatMult(vec);
//}

TEST(QuaterTest, QuaterdInverse)
{
    Quat<double> quat, quatInv;
    //60deg X, 30deg Y and 60deg Z
    quat.set(0.306186217847897, 0.435595740399158, 0.306186217847897, 0.789149130992431);

    quatInv = quat.inverse();

    EXPECT_NEAR(-0.306186217847897, quatInv[0], errorThreshold);
    EXPECT_NEAR(-0.435595740399158, quatInv[1], errorThreshold);
    EXPECT_NEAR(-0.306186217847897, quatInv[2], errorThreshold);
    EXPECT_NEAR(0.789149130992431, quatInv[3], errorThreshold);
}

TEST(QuaterTest, QuaterdToRotationVector)
{
    Quat<double> quat;
    //60deg X, 30deg Y and 60deg Z
    quat.set(0.306186217847897, 0.435595740399158, 0.306186217847897, 0.789149130992431);
    sofa::type::Vec3d p;

    p = quat.quatToRotationVector();
    //rotationMatrixToVector(quat2rotm(quatinv(q1)))
    EXPECT_NEAR(0.659404203095883, p[0], errorThreshold);
    EXPECT_NEAR(0.938101212029587, p[1], errorThreshold);
    EXPECT_NEAR(0.659404203095883, p[2], errorThreshold);
}

TEST(QuaterTest, QuaterdToEulerVector)
{
    Quat<double> quat;
    //60deg X, 30deg Y and 60deg Z
    quat.set(0.306186217847897, 0.435595740399158, 0.306186217847897, 0.789149130992431);
    sofa::type::Vec3d p;

    p = quat.toEulerVector();

    EXPECT_NEAR(1.047197551196598, p[0], errorThreshold);
    EXPECT_NEAR(0.523598775598299, p[1], errorThreshold);
    EXPECT_NEAR(1.047197551196598, p[2], errorThreshold);
}

TEST(QuaterTest, QuaterdSlerpExt)
{
    Quat<double> quat1, quat2, quatinterp;
    //60deg X, 30deg Y and 60deg Z
    quat1.set(0.306186217847897, 0.435595740399158, 0.306186217847897, 0.789149130992431);
    //30deg X, 15deg Y and 30deg Z
    quat2.set(0.215229667288440, 0.188196807757208, 0.215229667288440, 0.933774245836654);

    quatinterp.slerp(quat1, quat2, 0.0);

    EXPECT_NEAR(quat1[0], quatinterp[0], errorThreshold);
    EXPECT_NEAR(quat1[1], quatinterp[1], errorThreshold);
    EXPECT_NEAR(quat1[2], quatinterp[2], errorThreshold);
    EXPECT_NEAR(quat1[3], quatinterp[3], errorThreshold);

    quatinterp.slerp(quat1, quat2, 1.0);

    EXPECT_NEAR(quat2[0], quatinterp[0], errorThreshold);
    EXPECT_NEAR(quat2[1], quatinterp[1], errorThreshold);
    EXPECT_NEAR(quat2[2], quatinterp[2], errorThreshold);
    EXPECT_NEAR(quat2[3], quatinterp[3], errorThreshold);

    quatinterp.slerp(quat1, quat2, 0.5);
    // quatinterp(q1,q2, 0.5, 'slerp')
    EXPECT_NEAR(0.263984148784687, quatinterp[0], errorThreshold);
    EXPECT_NEAR(0.315815742361266, quatinterp[1], errorThreshold);
    EXPECT_NEAR(0.263984148784687, quatinterp[2], errorThreshold);
    EXPECT_NEAR(0.8722873123333, quatinterp[3], errorThreshold);
}

TEST(QuaterTest, QuaterdBuildRotationMatrix)
{
    Quat<double> quat;
    //60deg X, 30deg Y and 60deg Z
    quat.set(0.306186217847897, 0.435595740399158, 0.306186217847897, 0.789149130992431);
    double mat[4][4];
    quat.buildRotationMatrix(mat);

    //matlab results
    EXPECT_NEAR(0.433012701892219, mat[0][0], errorThreshold);
    EXPECT_NEAR(-0.216506350946110, mat[0][1], errorThreshold);
    EXPECT_NEAR(0.875000000000000, mat[0][2], errorThreshold);
    EXPECT_NEAR(0.750000000000000, mat[1][0], errorThreshold);
    EXPECT_NEAR(0.625000000000000, mat[1][1], errorThreshold);
    EXPECT_NEAR(-0.216506350946110, mat[1][2], errorThreshold);
    EXPECT_NEAR(-0.500000000000000, mat[2][0], errorThreshold);
    EXPECT_NEAR(0.750000000000000, mat[2][1], errorThreshold);
    EXPECT_NEAR(0.433012701892219, mat[2][2], errorThreshold);
}

TEST(QuaterTest, QuaterdWriteOpenglMatrix)
{
    Quat<double> quat;
    //60deg X, 30deg Y and 60deg Z
    quat.set(0.306186217847897, 0.435595740399158, 0.306186217847897, 0.789149130992431);
    double mat[16];
    quat.writeOpenGlMatrix(mat);

    //matlab results
    EXPECT_NEAR(0.433012701892219, mat[0], errorThreshold);
    EXPECT_NEAR(-0.216506350946110, mat[4], errorThreshold);
    EXPECT_NEAR(0.875000000000000, mat[8], errorThreshold);
    EXPECT_NEAR(0.0, mat[12], errorThreshold);
    EXPECT_NEAR(0.750000000000000, mat[1], errorThreshold);
    EXPECT_NEAR(0.625000000000000, mat[5], errorThreshold);
    EXPECT_NEAR(-0.216506350946110, mat[9], errorThreshold);
    EXPECT_NEAR(0.0, mat[13], errorThreshold);
    EXPECT_NEAR(-0.500000000000000, mat[2], errorThreshold);
    EXPECT_NEAR(0.750000000000000, mat[6], errorThreshold);
    EXPECT_NEAR(0.433012701892219, mat[10], errorThreshold);
    EXPECT_NEAR(0.0, mat[14], errorThreshold);
    EXPECT_NEAR(0.0, mat[3], errorThreshold);
    EXPECT_NEAR(0.0, mat[7], errorThreshold);
    EXPECT_NEAR(0.0, mat[11], errorThreshold);
    EXPECT_NEAR(1.0, mat[15], errorThreshold);
}

TEST(QuaterTest, QuaterdAxisToQuat)
{
    Quat<double> quat;
    //axang2quat([0 2 4 pi/3])
    const sofa::type::Vec3d axis(0, 2, 4);
    const double phi = 1.047197551196598; //60deg

    quat = quat.axisToQuat(axis, phi);

    EXPECT_NEAR(0.0, quat[0], errorThreshold);
    EXPECT_NEAR(0.223606797749979, quat[1], errorThreshold);
    EXPECT_NEAR(0.447213595499958, quat[2], errorThreshold);
    EXPECT_NEAR(0.866025403784439, quat[3], errorThreshold);
}

TEST(QuaterTest, QuaterdQuatToAxis)
{
    Quat<double> quat;
    //60deg X, 30deg Y and 60deg Z
    quat.set(0.306186217847897, 0.435595740399158, 0.306186217847897, 0.789149130992431);
    sofa::type::Vec3d axis;
    double phi = 0.0;

    quat.quatToAxis(axis, phi);
    //quat2axang(quat)
    EXPECT_NE(0.0, phi);
    EXPECT_NEAR(1.322747780240970, phi, errorThreshold);

    EXPECT_NEAR(0.498510912621420, axis[0], errorThreshold);
    EXPECT_NEAR(0.709206415646897, axis[1], errorThreshold);
    EXPECT_NEAR(0.498510912621420, axis[2], errorThreshold);
}

TEST(QuaterTest, QuaterdCreateQuaterFromFrame)
{
    //90 deg around Z from Identity
    const sofa::type::Vec3d xAxis(1.0, 0.0, 0.0);
    const sofa::type::Vec3d yAxis(0.0, 0.0, -1.0);
    const sofa::type::Vec3d zAxis(0.0, 1.0, 0.0);
    Quat<double> quat = Quat<double>::createQuaterFromFrame(xAxis, yAxis, zAxis);

    EXPECT_NEAR(-0.707106781186548, quat[0], errorThreshold);
    EXPECT_NEAR(0.0, quat[1], errorThreshold);
    EXPECT_NEAR(0.0, quat[2], errorThreshold);
    EXPECT_NEAR(0.707106781186548, quat[3], errorThreshold);
}

TEST(QuaterTest, QuaterdCreateFromRotationVector)
{
    //axang2quat([1 2 5 pi / 12])
    constexpr double phi = 0.261799387799149; //15deg
    const sofa::type::Vec3d axis = sofa::type::Vec3d(1, 2, 5).normalized() * phi;

    const auto quat = Quat<double>::createFromRotationVector(axis);

    EXPECT_NEAR(0.023830713274726, quat[0], errorThreshold);
    EXPECT_NEAR(0.047661426549452, quat[1], errorThreshold);
    EXPECT_NEAR(0.119153566373629, quat[2], errorThreshold);
    EXPECT_NEAR(0.991444861373810, quat[3], errorThreshold);
}

TEST(QuaterTest, QuaterdCreateQuaterFromEuler)
{
    //90deg X, 30deg Y and 60deg Z
    const sofa::type::Vec3d euler(1.570796326794897, 0.523598775598299, 1.047197551196598);
    Quat<double> quat = Quat<double>::createQuaterFromEuler(euler);

    EXPECT_NEAR(0.500000000000000, quat[0], errorThreshold);
    EXPECT_NEAR(0.500000000000000, quat[1], errorThreshold);
    EXPECT_NEAR(0.183012701892219, quat[2], errorThreshold);
    EXPECT_NEAR(0.683012701892219, quat[3], errorThreshold);
}

TEST(QuaterTest, QuaterdQuatDiff)
{
    Quat<double> quat1, quat2, quatres;
    //60deg X, 30deg Y and 60deg Z
    quat1.set(0.306186217847897, 0.435595740399158, 0.306186217847897, 0.789149130992431);
    //30deg X, 15deg Y and 30deg Z
    quat2.set(0.215229667288440, 0.188196807757208, 0.215229667288440, 0.933774245836654);

    //quatmultiply(quatinv(q2),q1)
    quatres = quatres.quatDiff(quat1, quat2);

    EXPECT_NEAR(0.152190357252180, quatres[0], errorThreshold);
    EXPECT_NEAR(0.258232736683732, quatres[1], errorThreshold);
    EXPECT_NEAR(0.079930642251891, quatres[2], errorThreshold);
    EXPECT_NEAR(0.950665578052285, quatres[3], errorThreshold);
}

//TEST(QuaterTest, QuaterdAngularDisplacement)
//{
//    Quat<double> quat1, quat2;
//    sofa::type::Vec3d displacement;
//    //60deg X, 30deg Y and 60deg Z
//    quat1.set(0.306186217847897, 0.435595740399158, 0.306186217847897, 0.789149130992431);
//    //30deg X, 15deg Y and 30deg Z
//    quat2.set(0.215229667288440, 0.188196807757208, 0.215229667288440, 0.933774245836654);
//
//    //quatmultiply(quatinv(q2),q1)
//    displacement = quat1.angularDisplacement(quat1, quat2);
//
//    EXPECT_NEAR(0.263750186248169, displacement[0], errorThreshold);
//    EXPECT_NEAR(0.485506742293049, displacement[1], errorThreshold);
//    EXPECT_NEAR(0.383154507999762, displacement[2], errorThreshold);
//}
//
//TEST(QuaterTest, QuaterdSlerp)
//{
//    Quat<double> quat1, quat2, quatinterp;
//    //60deg X, 30deg Y and 60deg Z
//    quat1.set(0.306186217847897, 0.435595740399158, 0.306186217847897, 0.789149130992431);
//    //30deg X, 15deg Y and 30deg Z
//    quat2.set(0.215229667288440, 0.188196807757208, 0.215229667288440, 0.933774245836654);
//
//    quatinterp = quat1.slerp(quat2, 0.0);
//
//    EXPECT_NEAR(quat1[0], quatinterp[0], errorThreshold);
//    EXPECT_NEAR(quat1[1], quatinterp[1], errorThreshold);
//    EXPECT_NEAR(quat1[2], quatinterp[2], errorThreshold);
//    EXPECT_NEAR(quat1[3], quatinterp[3], errorThreshold);
//
//    quatinterp = quat1.slerp(quat2, 1.0);
//
//    EXPECT_NEAR(quat2[0], quatinterp[0], errorThreshold);
//    EXPECT_NEAR(quat2[1], quatinterp[1], errorThreshold);
//    EXPECT_NEAR(quat2[2], quatinterp[2], errorThreshold);
//    EXPECT_NEAR(quat2[3], quatinterp[3], errorThreshold);
//
//    quatinterp = quat1.slerp(quat2, 0.5);
//    // quatinterp(q1,q2, 0.5, 'slerp')
//    EXPECT_NEAR(0.263984148784687, quatinterp[0], errorThreshold);
//    EXPECT_NEAR(0.315815742361266, quatinterp[1], errorThreshold);
//    EXPECT_NEAR(0.263984148784687, quatinterp[2], errorThreshold);
//    EXPECT_NEAR(0.8722873123333, quatinterp[3], errorThreshold);
//}

TEST(QuaterTest, QuaterdSlerp2)
{
    Quat<double> quat1, quat2, quatinterp;
    //60deg X, 30deg Y and 60deg Z
    quat1.set(0.306186217847897, 0.435595740399158, 0.306186217847897, 0.789149130992431);
    //30deg X, 15deg Y and 30deg Z
    quat2.set(0.215229667288440, 0.188196807757208, 0.215229667288440, 0.933774245836654);

    quatinterp = quat1.slerp2(quat2, 0.0);

    EXPECT_NEAR(quat1[0], quatinterp[0], errorThreshold);
    EXPECT_NEAR(quat1[1], quatinterp[1], errorThreshold);
    EXPECT_NEAR(quat1[2], quatinterp[2], errorThreshold);
    EXPECT_NEAR(quat1[3], quatinterp[3], errorThreshold);

    quatinterp = quat1.slerp2(quat2, 1.0);

    EXPECT_NEAR(quat2[0], quatinterp[0], errorThreshold);
    EXPECT_NEAR(quat2[1], quatinterp[1], errorThreshold);
    EXPECT_NEAR(quat2[2], quatinterp[2], errorThreshold);
    EXPECT_NEAR(quat2[3], quatinterp[3], errorThreshold);

    quatinterp = quat1.slerp2(quat2, 0.5);
    // quatinterp(q1,q2, 0.5, 'slerp')
    EXPECT_NEAR(0.263984148784687, quatinterp[0], errorThreshold);
    EXPECT_NEAR(0.315815742361266, quatinterp[1], errorThreshold);
    EXPECT_NEAR(0.263984148784687, quatinterp[2], errorThreshold);
    EXPECT_NEAR(0.872287312333299, quatinterp[3], errorThreshold);
}

TEST(QuaterTest, QuaterdFromUnitVectors)
{
    sofa::type::Vec<3, double> vFrom(1, 0, 0);
    sofa::type::Vec<3, double> vTo(0, 1, 0);

    Quat<double> quat1;
    quat1.setFromUnitVectors(vFrom, vTo);

    EXPECT_NEAR(0, quat1[0], errorThreshold);
    EXPECT_NEAR(0, quat1[1], errorThreshold);
    EXPECT_NEAR(0.7071067811865475, quat1[2], errorThreshold);
    EXPECT_NEAR(0.7071067811865475, quat1[3], errorThreshold);

    vFrom = sofa::type::Vec<3, double>(0.5, 0.4, 0.3);
    vTo = sofa::type::Vec<3, double>(0, 0.2, -1);
    vFrom.normalize();
    vTo.normalize();
    quat1.setFromUnitVectors(vFrom, vTo);

    EXPECT_NEAR(-0.5410972822985042, quat1[0], errorThreshold);
    EXPECT_NEAR(0.5881492198896785, quat1[1], errorThreshold);
    EXPECT_NEAR(0.11762984397793572, quat1[2], errorThreshold);
    EXPECT_NEAR(0.5894552112230939, quat1[3], errorThreshold);
}

TEST(QuaterTest, StructuredBindings)
{
    Quat<double> quat1;
    const auto& [a,b,c,d] = quat1;
    EXPECT_NEAR(0., a, errorThreshold);
    EXPECT_NEAR(0., b, errorThreshold);
    EXPECT_NEAR(0., c, errorThreshold);
    EXPECT_NEAR(1., d, errorThreshold);
}
