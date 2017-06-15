/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/helper/Quater.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include <time.h>      

using sofa::helper::Quater;

double errorThreshold = 1e-6;

TEST(QuaterTest, EulerAngles)
{
    // Try to tranform a quat (q0) to euler angles and then back to quat (q1)
    // compare the result of the rotations define by q0 and q1 on a vector 
    srand (time(NULL));
    for (int i = 0; i < 100; ++i)
    {   // Generate a test vector p
        sofa::defaulttype::Vec<3,double> p(((rand()%101)/100)+1.f, ((rand()%101)/100)+1.f, ((rand()%101)/100)+1.f);

        //Generate a test quaternion
        Quater<double> q0 (((rand()%201)/100)-1.f, ((rand()%201)/100)-1.f, ((rand()%201)/100)-1.f, ((rand()%201)/100)-1.f);
        q0.Quater<double>::normalize();
        //if(q0[3]<0)q0*=(-1);

        //Avoid singular values
        while(fabs(q0[0])==0.5 && fabs(q0[1])==0.5 &&fabs(q0[2])==0.5 && fabs(q0[3])==0.5)
        {
            Quater<double> q2 (((rand()%201)/100)-1.f, ((rand()%201)/100)-1.f, ((rand()%201)/100)-1.f, ((rand()%201)/100)-1.f);
            q2.Quater<double>::normalize();
            //if(q2[3]<0)q2*=(-1);
            q0=q2;
        }

        //Rotate p with q0
        sofa::defaulttype::Vec<3,double>  p0 = q0.Quater<double>::rotate(p);       

        //Transform q0 into euler angles and back to a quaternion (q1)
        Quater<double> q1 = Quater<double>::createQuaterFromEuler(q0.Quater<double>::toEulerVector());     
        //if(q1[3]<0)q1*=(-1);

        //std::cout << "Q0 " << q0 << std::endl << "Q1 " << q1 << std::endl;

        //Rotate p with q1
        sofa::defaulttype::Vec<3,double> p1 = q1.Quater<double>::rotate(p);

        //Compare the result of the two rotations on p
        EXPECT_EQ(p0,p1);

        // Specific check for a certain value of p
        sofa::defaulttype::Vec<3,double> p2(2,1,1);
        p0 = q0.Quater<double>::rotate(p2);
        p1 = q1.Quater<double>::rotate(p2);
        EXPECT_EQ(p0,p1);
    }

}

/* Following unit test results have been checked with Matlab 2016a.
 * Note: ambiguous result with rotate and invrotate.
 */


TEST(QuaterTest, QuaterdSet)
{
    Quater<double> quat;
    quat.set(0.0, 0.0, 0.0, 1.0);

    EXPECT_DOUBLE_EQ(0.0, quat[0]);
    EXPECT_DOUBLE_EQ(0.0, quat[1]);
    EXPECT_DOUBLE_EQ(0.0, quat[2]);
    EXPECT_DOUBLE_EQ(1.0, quat[3]);
}

TEST(QuaterTest, QuaterdIdentity)
{
    Quater<double> id;
    id.set(0.0, 0.0, 0.0, 1.0);

    Quater<double> quat = Quater<double>::identity();
    EXPECT_DOUBLE_EQ(id[0], quat[0]);
    EXPECT_DOUBLE_EQ(id[1], quat[1]);
    EXPECT_DOUBLE_EQ(id[2], quat[2]);
    EXPECT_DOUBLE_EQ(id[3], quat[3]);
}

TEST(QuaterTest, QuaterdConstPtr)
{
    Quater<double> quat;
    quat.set(0.0, 0.0, 0.0, 1.0);

    const double* quatptr = quat.ptr();

    EXPECT_DOUBLE_EQ(0.0, quatptr[0]);
    EXPECT_DOUBLE_EQ(0.0, quatptr[1]);
    EXPECT_DOUBLE_EQ(0.0, quatptr[2]);
    EXPECT_DOUBLE_EQ(1.0, quatptr[3]);
}

TEST(QuaterTest, QuaterdPtr)
{
    Quater<double> quat;
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
    Quater<double> quat;
    quat.set(1.0, 0.0, 1.0, 0.0);

    quat.normalize();
    
    EXPECT_NEAR(0.707106781186548, quat[0], errorThreshold);
    EXPECT_NEAR(0.0  ,  quat[1], errorThreshold);
    EXPECT_NEAR(0.707106781186548, quat[2], errorThreshold);
    EXPECT_NEAR(0.0   , quat[3], errorThreshold);
}

TEST(QuaterTest, QuaterdClear)
{
    Quater<double> quat;
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
    Quater<double> quat;
    //90 deg around Z from Identity
    sofa::defaulttype::Vec3d xAxis(1.0, 0.0, 0.0);
    sofa::defaulttype::Vec3d yAxis(0.0, 0.0, -1.0);
    sofa::defaulttype::Vec3d zAxis(0.0, 1.0, 0.0);
    quat.fromFrame(xAxis, yAxis, zAxis);

    EXPECT_NEAR(-0.707106781186548, quat[0], errorThreshold);
    EXPECT_NEAR(0.0, quat[1], errorThreshold);
    EXPECT_NEAR(0.0, quat[2], errorThreshold);
    EXPECT_NEAR(0.707106781186548, quat[3], errorThreshold);
}

TEST(QuaterTest, QuaterdFromMatrix)
{
    Quater<double> quat;
    //30deg X, 30deg Y and 30deg Z
    double mat[9]  = { 0.750000000000000, -0.216506350946110, 0.625000000000000,
                       0.433012701892219, 0.875000000000000, -0.216506350946110,
                      -0.500000000000000, 0.433012701892219, 0.750000000000000 };
    quat.fromMatrix(sofa::defaulttype::Matrix3(mat));

    EXPECT_NEAR(0.176776695296637, quat[0], errorThreshold);
    EXPECT_NEAR(0.306186217847897, quat[1], errorThreshold);
    EXPECT_NEAR(0.176776695296637, quat[2], errorThreshold);
    EXPECT_NEAR(0.918558653543692, quat[3], errorThreshold);
}

TEST(QuaterTest, QuaterdToMatrix)
{
    Quater<double> quat;
    //60deg X, 30deg Y and 60deg Z
    quat.set(0.306186217847897, 0.435595740399158, 0.306186217847897, 0.789149130992431);
    sofa::defaulttype::Mat3x3d mat;
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
    Quater<double> quat;
    //30deg X, 15deg Y and 30deg Z
    quat.set(0.215229667288440, 0.188196807757208, 0.215229667288440, 0.933774245836654);
    sofa::defaulttype::Vec3d p(3, 6, 9);
    sofa::defaulttype::Vec3d rp = quat.rotate(p); //equiv if inverseQuat.rotate() in matlab

    EXPECT_NEAR(4.580932858428164, rp[0], errorThreshold);
    EXPECT_NEAR(3.448650396246470, rp[1], errorThreshold);
    EXPECT_NEAR(9.649967077199914, rp[2], errorThreshold);
}

TEST(QuaterTest, QuaterdInvRotateVec)
{
    Quater<double> quat;
    //30deg X, 15deg Y and 30deg Z
    quat.set(0.215229667288440, 0.188196807757208, 0.215229667288440, 0.933774245836654);
    sofa::defaulttype::Vec3d p(3, 6, 9);
    sofa::defaulttype::Vec3d rp = quat.inverseRotate(p);//equiv if quat.rotate() in matlab

    EXPECT_NEAR(3.077954984157941, rp[0], errorThreshold);
    EXPECT_NEAR(8.272072482340949, rp[1], errorThreshold);
    EXPECT_NEAR(6.935344977893669, rp[2], errorThreshold);
}

TEST(QuaterTest, QuaterdOperatorAdd)
{
    Quater<double> quat1, quat2, quatAdd;
    //30deg X, 15deg Y and 30deg Z
    quat1.set(0.215229667288440, 0.188196807757208, 0.215229667288440, 0.933774245836654);
    //60deg X, 30deg Y and 60deg Z
    quat2.set(0.306186217847897, 0.435595740399158, 0.306186217847897, 0.789149130992431);
    quatAdd = quat1 + quat2;
    sofa::defaulttype::Vec3d p(3, 6, 9);
    sofa::defaulttype::Vec3d rp = quatAdd.rotate(p);
    sofa::defaulttype::Vec3d rrp = quat2.rotate(quat1.rotate(p));

    //According to the comments, the addition of two quaternions in the compound of the two related rotations

    EXPECT_NEAR(rrp[0], rp[0], errorThreshold);
    EXPECT_NEAR(rrp[1], rp[1], errorThreshold);
    EXPECT_NEAR(rrp[2], rp[2], errorThreshold);
}

TEST(QuaterTest, QuaterdOperatorMultQuat)
{
    Quater<double> quat1, quat2, quatMult;
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
    Quater<double> quat, quatMult;
    double r = 2.0;
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
    Quater<double> quat, quatDiv;
    double r = 3.0;
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
    Quater<double> quat;
    double r = 2.0;
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
    Quater<double> quat;
    double r = 3.0;
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
//    Quater<double> quat, quatMult;
//    sofa::defaulttype::Vec<3, double> vec(1,2,3);
//    //30deg X, 15deg Y and 30deg Z
//    quat.set(0.215229667288440, 0.188196807757208, 0.215229667288440, 0.933774245836654);
//
//    quatMult = quat.quatVectMult(vec);
//}
//
//TEST(QuaterTest, QuaterdVecQuatMult)
//{
//    Quater<double> quat, quatMult;
//    sofa::defaulttype::Vec<3, double> vec(1, 2, 3);
//    //30deg X, 15deg Y and 30deg Z
//    quat.set(0.215229667288440, 0.188196807757208, 0.215229667288440, 0.933774245836654);
//
//    quatMult = quat.vectQuatMult(vec);
//}

TEST(QuaterTest, QuaterdInverse)
{
    Quater<double> quat, quatInv;
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
    Quater<double> quat;
    //60deg X, 30deg Y and 60deg Z
    quat.set(0.306186217847897, 0.435595740399158, 0.306186217847897, 0.789149130992431);
    sofa::defaulttype::Vec3d p;

    p = quat.quatToRotationVector();
    //rotationMatrixToVector(quat2rotm(quatinv(q1)))
    EXPECT_NEAR(0.659404203095883, p[0], errorThreshold);
    EXPECT_NEAR(0.938101212029587, p[1], errorThreshold);
    EXPECT_NEAR(0.659404203095883, p[2], errorThreshold);
}

TEST(QuaterTest, QuaterdToEulerVector)
{
    Quater<double> quat;
    //60deg X, 30deg Y and 60deg Z
    quat.set(0.306186217847897, 0.435595740399158, 0.306186217847897, 0.789149130992431);
    sofa::defaulttype::Vec3d p;

    p = quat.toEulerVector();
    std::cout << p << std::endl;

    EXPECT_NEAR(1.047197551196598, p[0], errorThreshold);
    EXPECT_NEAR(0.523598775598299, p[1], errorThreshold);
    EXPECT_NEAR(1.047197551196598, p[2], errorThreshold);
}

TEST(QuaterTest, QuaterdSlerpExt)
{
    Quater<double> quat1, quat2, quatinterp;
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
    Quater<double> quat;
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
    Quater<double> quat;
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
    Quater<double> quat;
    //axang2quat([0 2 4 pi/3])
    sofa::defaulttype::Vec3d axis(0, 2, 4);
    double phi = 1.047197551196598; //60deg

    quat = quat.axisToQuat(axis, phi);

    EXPECT_NEAR(0.0, quat[0], errorThreshold);
    EXPECT_NEAR(0.223606797749979, quat[1], errorThreshold);
    EXPECT_NEAR(0.447213595499958, quat[2], errorThreshold);
    EXPECT_NEAR(0.866025403784439, quat[3], errorThreshold);
}

TEST(QuaterTest, QuaterdQuatToAxis)
{
    Quater<double> quat;
    //60deg X, 30deg Y and 60deg Z
    quat.set(0.306186217847897, 0.435595740399158, 0.306186217847897, 0.789149130992431);
    sofa::defaulttype::Vec3d axis;
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
    sofa::defaulttype::Vec3d xAxis(1.0, 0.0, 0.0);
    sofa::defaulttype::Vec3d yAxis(0.0, 0.0, -1.0);
    sofa::defaulttype::Vec3d zAxis(0.0, 1.0, 0.0);
    Quater<double> quat = Quater<double>::createQuaterFromFrame(xAxis, yAxis, zAxis);

    EXPECT_NEAR(-0.707106781186548, quat[0], errorThreshold);
    EXPECT_NEAR(0.0, quat[1], errorThreshold);
    EXPECT_NEAR(0.0, quat[2], errorThreshold);
    EXPECT_NEAR(0.707106781186548, quat[3], errorThreshold);
}

//TEST(QuaterTest, QuaterdCreateFromRotationVector)
//{
//    //axang2quat([1 2 5 pi / 12])
//    sofa::defaulttype::Vec3d axis(1, 2, 5);
//    double phi = 0.261799387799149; //15deg
//
//    Quater<double> quat = Quater<double>::createFromRotationVector(axis * phi);
//
//    EXPECT_NEAR(0.023830713274726, quat[0], errorThreshold);
//    EXPECT_NEAR(0.047661426549452, quat[1], errorThreshold);
//    EXPECT_NEAR(0.119153566373629, quat[2], errorThreshold);
//    EXPECT_NEAR(0.991444861373810, quat[3], errorThreshold);
//}
//////////////////////////////////////////
TEST(QuaterTest, QuaterdCreateQuaterFromEuler)
{
    //90deg X, 30deg Y and 60deg Z
    sofa::defaulttype::Vec3d euler(1.570796326794897, 0.523598775598299, 1.047197551196598);
    Quater<double> quat = Quater<double>::createQuaterFromEuler(euler);

    EXPECT_NEAR(0.500000000000000, quat[0], errorThreshold);
    EXPECT_NEAR(0.500000000000000, quat[1], errorThreshold);
    EXPECT_NEAR(0.183012701892219, quat[2], errorThreshold);
    EXPECT_NEAR(0.683012701892219, quat[3], errorThreshold);

}
/////////////////////////////////
//TEST(QuaterTest, QuaterdCreateFromRotationVectorT)
//{
//    //axang2quat([1 2 5 pi / 12])
//    sofa::defaulttype::Vec3d axis(1, 2, 5);
//    double phi = 0.261799387799149; //15deg
//
//    Quater<double> quat = Quater<double>::createFromRotationVector(axis * phi);
//
//    EXPECT_NEAR(0.023830713274726, quat[0], errorThreshold);
//    EXPECT_NEAR(0.047661426549452, quat[1], errorThreshold);
//    EXPECT_NEAR(0.119153566373629, quat[2], errorThreshold);
//    EXPECT_NEAR(0.991444861373810, quat[3], errorThreshold);
//}

TEST(QuaterTest, QuaterdQuatDiff)
{
    Quater<double> quat1, quat2, quatres;
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
//    Quater<double> quat1, quat2;
//    sofa::defaulttype::Vec3d displacement;
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
//    Quater<double> quat1, quat2, quatinterp;
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
    Quater<double> quat1, quat2, quatinterp;
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
