/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaTest/Sofa_test.h>
#include <SofaTest/TestMessageHandler.h>


#include <SofaMeshCollision/MeshNewProximityIntersection.inl>

#include <iostream>
#include <sstream>
#include <fstream>


namespace sofa{

    struct MeshNewProximityIntersectionTest : public Sofa_test<>
    {
        typedef sofa::defaulttype::Vector3 Vec3;
        typedef sofa::defaulttype::Vector2 Vec2;
        typedef sofa::component::collision::MeshNewProximityIntersection ProximityIntersection;

        MeshNewProximityIntersectionTest(){
        }

        bool checkOutput(sofa::core::collision::DetectionOutput& o, Vec3 pc)
        {
            if (Sofa_test::vectorMaxDiff<3,SReal>(pc, o.point[0])>1e-6)
            {
                ADD_FAILURE() <<"wrong collision point: "<<o.point[0]<<", expected: "<<pc;
                return false;
            }


            return true;
        }

        bool pointTriangle()
        {
            sofa::helper::vector<sofa::core::collision::DetectionOutput> outputVector;
            unsigned nbTest = 100;
            int flag = 0xffff;


            for(unsigned i=0; i<nbTest; i++)
            {
                // create a random triangle
                Vec3 p1( Real(helper::drand(1.0)), Real(helper::drand(1.0)), Real(helper::drand(1.0)) );
                Vec3 p2( Real(helper::drand(1.0)), Real(helper::drand(1.0)), Real(helper::drand(1.0)) );
                Vec3 p3( Real(helper::drand(1.0)), Real(helper::drand(1.0)), Real(helper::drand(1.0)) );

                Vec3 p1p2 = p2 - p1;
                Vec3 p1p3 = p3 - p1;

                Vec3 n = p1p2.cross(p1p3);
                n.normalize();

                // random points inside the triangle
                Vec2 bary(helper::drand(),helper::drand());
                Vec3 bary2;
                bary/=bary.sum();

                Vec3 pc = p1 + bary[0]*(p1p2) + bary[1]*(p1p3);
                SReal maxDist = 0.1;
                SReal dist = (2*helper::drand() - 1)*maxDist;
                Vec3 q = pc + dist*n;
                if(!ProximityIntersection::doIntersectionTrianglePoint(maxDist,flag, p1,p2,p3,n,q,&outputVector,i,true))
                {
                     ADD_FAILURE() << "intersection point in triangle failed! : \n   p1: "<<p1<<"\n   p2: "<<p2<<"\n   p3: "<<p3<<"\n   n: "<<n<<"\n    q: "<<q<<"\n distance: "<<dist;
                     return false;
                }


                // point on triangle corners
                Vec3 n2( Real(helper::drand(1.0)), Real(helper::drand(1.0)), Real(helper::drand(1.0)) );
                for(unsigned j =0; j<3; j++)
                {
                    n2.normalize();
                    bary2 = Vec3(0,0,0);
                    bary2[j] = 1;

                    pc = bary2[0]*(p1) + bary2[1]*(p2) + bary2[2]*(p3);
                    q = pc + dist*n2;
                    if(!ProximityIntersection::doIntersectionTrianglePoint(maxDist,flag, p1,p2,p3,n,q,&outputVector,i))
                    {
                        ADD_FAILURE() << "intersection point on triangle corner failed! : \n   p1: "<<p1<<"\n   p2: "<<p2<<"\n   p3: "<<p3<<"\n   n: "<<n<<"\n    q: "<<q<<"\n distance: "<<dist;
                        return false;
                    }
                }

                // points on edges
                for(unsigned j =0; j<3; j++)
                {
                    bary2 = Vec3(0,0,0);
                    bary2[j] = 0.5;
                    bary2[(j+1)%3] = 0.5;

                    pc = bary2[0]*(p1) + bary2[1]*(p2) + bary2[2]*(p3);
                    q = pc + dist*n2;
                    if(!ProximityIntersection::doIntersectionTrianglePoint(maxDist,flag, p1,p2,p3,n,q,&outputVector,i))
                    {
                        ADD_FAILURE() << "intersection point on triangle edges failed! : \n   p1: "<<p1<<"\n   p2: "<<p2<<"\n   p3: "<<p3<<"\n   n: "<<n<<"\n    q: "<<q<<"\n distance: "<<dist;
                        return false;
                    }
                }
            }

            {
                // custom test for triangle with an angle > 90 degrees

                Vec3 p1(0,0,0);
                Vec3 p2(-0.1,1,0);
                Vec3 p3(1,0,0);

                Vec3 p1p2 = p2 - p1;
                Vec3 p1p3 = p3 - p1;

                Vec3 n = p1p2.cross(p1p3);
                n.normalize();

                Vec2 bary(-2,0);
                Vec3 pc =  p1 + bary[0]*(p1p2) + bary[1]*(p1p3);
                if(ProximityIntersection::doIntersectionTrianglePoint(0.1,flag, p1,p2,p3,n,pc,&outputVector,0))
                {
                    ADD_FAILURE() << "intersection point in triangle failed (false positive)! : \n   p1: "<<p1<<"\n   p2: "<<p2<<"\n   p3: "<<p3<<"\n   n: "<<n<<"\n    q: "<<pc;
                    return false;
                }

                bary = Vec2(0,-2);
                pc =  p1 + bary[0]*(p1p2) + bary[1]*(p1p3);
                if(ProximityIntersection::doIntersectionTrianglePoint(0.1,flag, p1,p2,p3,n,pc,&outputVector,0))
                {
                    ADD_FAILURE() << "intersection point in triangle failed (false positive)! : \n   p1: "<<p1<<"\n   p2: "<<p2<<"\n   p3: "<<p3<<"\n   n: "<<n<<"\n    q: "<<pc;
                    return false;
                }
                outputVector.clear();
            }
            return true;
        }

    };


TEST_F(MeshNewProximityIntersectionTest, pointTriangle ) {
    EXPECT_MSG_NOEMIT(Error) ;
    ASSERT_TRUE( pointTriangle());
}

}
