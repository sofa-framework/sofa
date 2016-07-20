#include <sofa/helper/Quater.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include <time.h>      

using sofa::helper::Quater;

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
