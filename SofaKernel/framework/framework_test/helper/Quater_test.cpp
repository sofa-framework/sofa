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
    for (int i = 0; i < 1000; ++i)
    {   // Generate a test vector p
        sofa::defaulttype::Vec<3,double> p(((rand()%101)/100)+1.f, ((rand()%101)/100)+1.f, ((rand()%101)/100)+1.f);

        //Generate a test quaternion
        Quater<double> q0 (((rand()%201)/100)-1.f, ((rand()%201)/100)-1.f, ((rand()%201)/100)-1.f, ((rand()%201)/100)-1.f);
        q0.Quater<double>::normalize();
        if(q0[3]<0)q0*=(-1);

        //Avoid singular values
        while(fabs(q0[0])==0.5 && fabs(q0[1])==0.5 &&fabs(q0[2])==0.5 && fabs(q0[3])==0.5)
        {
            Quater<double> q2 (((rand()%201)/100)-1.f, ((rand()%201)/100)-1.f, ((rand()%201)/100)-1.f, ((rand()%201)/100)-1.f);
            q2.Quater<double>::normalize();
            if(q2[3]<0)q2*=(-1);
            q0=q2;
        }

        //Rotate p with q0
        sofa::defaulttype::Vec<3,double>  p0 = q0.Quater<double>::rotate(p);       

        //Transform q0 into euler angles and back to a quaternion (q1)
        Quater<double> q1 = Quater<double>::createQuaterFromEuler(q0.Quater<double>::toEulerVector());        
        if(q1[3]<0)q1*=(-1);

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
    sofa::defaulttype::Vec3d rp = quat.rotate(p);

    EXPECT_NEAR(3.077954984157941, rp[0], errorThreshold);
    EXPECT_NEAR(8.272072482340949, rp[1], errorThreshold);
    EXPECT_NEAR(6.935344977893669, rp[2], errorThreshold);
}