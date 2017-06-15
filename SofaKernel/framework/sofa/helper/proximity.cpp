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
#include "proximity.h"
#include <sofa/helper/LCPSolver.inl>

namespace sofa
{

namespace helper
{
using namespace sofa::helper;
using namespace sofa::defaulttype;

//------------------------
// DISTANCETRITRI
/*
 aP = alphaP, bP = betaP are the barycentric coordinates of point P on triangle P
 aQ = alphaQ, bQ = betaQ are the barycentric coordinates of point Q on triangle Q

 the computation of the distance between P and Q can be done with:

QP^2 = [aP bP aQ bQ] * [P1P2*P1P2 P1P2*P1P3 -P1P2*Q1Q2 -P1P2*Q1Q3 * [aP bP aQ bQ]^T
								   P1P3*P1P3 -P1P3*Q1Q2 -P1P3*Q1Q3
						sym					 Q1Q2*Q1Q2  Q1Q2*Q1Q3
													    Q1Q3*Q1Q3]
  + [Q1P1*P1P2 Q1P1*P1P3 -Q1P1*P1Q2 -Q1P1*P1Q3] *[aP bP aQ bQ]^T + (Q1P1)^2

  P and Q are on triangles, so we have the constraints:
aP>=0, bP>=0, aQ>=0, bQ>=0
aP+bP<=1, aQ+bQ<=1 <=> -aP-bP>=-1 , -aQ-bQ>=-1

To perform min(QP^2) we use LCP formalism

*/
//------------------------

DistanceTriTri::
DistanceTriTri()
{
}

DistanceTriTri::
~DistanceTriTri()
{
}



void DistanceTriTri::
NewComputation(const Vector3& P1, const Vector3& P2, const Vector3& P3, const Vector3& Q1, const Vector3& Q2, const Vector3& Q3, Vector3 &Presult, Vector3 &Qresult)
{
    //Vector3 P1, P2, P3, Q1, Q2, Q3;
    //P1 = triP->p1();
    //P2 = triP->p2();
    //P3 = triP->p3();
    //Q1 = triQ->p1();
    //Q2 = triQ->p2();
    //Q3 = triQ->p3();

    double alphaP, betaP, alphaQ, betaQ;
    int i;
    Vector3 P1P2, P1P3, Q1Q2, Q1Q3, P1Q1, PQ;
    LCPSolver<6>::Matrix  A;
    double          b[6], result[12];
    LCPSolver<6>          lcp;

    // clear result
    for (i=0; i<12; i++)
    {
        result[i] = 0.0;
    }


    P1P2=P2 - P1;
    P1P3=P3 - P1;
    Q1Q2=Q2 - Q1;
    Q1Q3=Q3 - Q1;
    P1Q1=Q1 - P1;

    A[0][4] = 1.0; A[4][0] = -1.0;
    A[1][4] = 1.0; A[4][1] = -1.0;
    A[2][4] = 0.0; A[4][2] = 0.0;
    A[3][4] = 0.0; A[4][3] = 0.0;

    A[0][5] = 0.0; A[5][0] = 0.0;
    A[1][5] = 0.0; A[5][1] = 0.0;
    A[2][5] = 1.0; A[5][2] = -1.0;
    A[3][5] = 1.0; A[5][3] = -1.0;

    A[4][4] = 0.0; A[5][5] = 0.0;
    A[4][5] = 0.0; A[5][4] = 0.0;

    A[0][0] = dot(P1P2,P1P2);   A[0][1] = dot(P1P3,P1P2);   A[0][2] =-dot(Q1Q2,P1P2);  A[0][3] =-dot(Q1Q3,P1P2);
    A[1][0] = dot(P1P2,P1P3);   A[1][1] = dot(P1P3,P1P3);   A[1][2] =-dot(Q1Q2,P1P3);  A[1][3] =-dot(Q1Q3,P1P3);
    A[2][0] = -dot(P1P2,Q1Q2);  A[2][1] = -dot(P1P3,Q1Q2);  A[2][2] = dot(Q1Q2,Q1Q2);  A[2][3] = dot(Q1Q3,Q1Q2);
    A[3][0] = -dot(P1P2,Q1Q3);  A[3][1] = -dot(P1P3,Q1Q3);  A[3][2] = dot(Q1Q2,Q1Q3);  A[3][3] = dot(Q1Q3,Q1Q3);

    b[0]=-dot(P1Q1,P1P2);  b[1]=-dot(P1Q1,P1P3);  b[2]= dot(P1Q1,Q1Q2); b[3]= dot(P1Q1,Q1Q3);
    b[4]=1.0;  b[5]=1.0;

    if(lcp.solve(b, A, result)==1)
    {
        alphaP=result[6];
        betaP=result[7];
        alphaQ=result[8];
        betaQ=result[9];
        Presult = P1 + P1P2*alphaP + P1P3*betaP;
        Qresult = Q1 + Q1Q2*alphaQ + Q1Q3*betaQ;
    }
    else
    {
        printf(" no result from LCP !\n");
        Presult = P1;
        Qresult = Q1;
    }


}


//------------------------
// DISTANCESEGTRI
//------------------------

DistanceSegTri::
DistanceSegTri()
{
}

DistanceSegTri::
~DistanceSegTri()
{
}

//void DistanceSegTri::
//NewComputation(Triangle *tri, const Vector3 &Q1, const Vector3 &Q2, Vector3 &Presult, Vector3 &Qresult)
//{
//	NewComputation( tri->p1(), tri->p2(), tri->p3(), Q1, Q2,Presult, Qresult );
//}

void DistanceSegTri::
NewComputation(const Vector3 &P1, const Vector3 &P2, const Vector3 &P3, const Vector3 &Q1, const Vector3 &Q2, Vector3 &Presult, Vector3 &Qresult)
{
    double alpha, beta, gamma;
    int i;


    Vector3 P1P2, P1P3, Q1Q2, P1Q1, PQ;
    LCPSolver<5>::Matrix  A;
    double          b[5], result[10];
    LCPSolver<5>          lcp;

    for (i=0; i<10; i++)
    {
        result[i] = 0.0;
    }

    Q1Q2 = Q2 - Q1;
    P1P2 = P2 - P1;
    P1P3 = P3 - P1;
    P1Q1 = Q1 - P1;


    // initialize A
    A[0][3] = 1.0; A[0][4] = 0.0;
    A[1][3] = 1.0; A[1][4] = 0.0;
    A[2][3] = 0.0; A[2][4] = 1.0;
    A[3][0] = -1.0; A[3][1] = -1.0; A[3][2] = 0.0;	A[3][3] = 0.0; A[3][4] = 0.0;
    A[4][0] = 0.0; A[4][1] = 0.0;	A[4][2] = -1.0;	A[4][3] = 0.0; A[4][4] = 0.0;

    A[0][0] = dot(P1P2,P1P2);   A[0][1] = dot(P1P3,P1P2);   A[0][2] = -dot(Q1Q2,P1P2);
    A[1][0] = dot(P1P2,P1P3);   A[1][1] = dot(P1P3,P1P3);   A[1][2] = -dot(Q1Q2,P1P3);
    A[2][0] = -dot(P1P2,Q1Q2);  A[2][1] = -dot(P1P3,Q1Q2);  A[2][2] = dot(Q1Q2,Q1Q2);

    // initialize b
    b[3]=1.0;  b[4]=1.0;
    b[0]=-dot(P1Q1,P1P2);  b[1]=-dot(P1Q1,P1P3);  b[2]=dot(P1Q1,Q1Q2);

    if(lcp.solve(b, A, result))
    {
        alpha=result[5];
        beta=result[6];
        gamma=result[7];

        Presult = P1 + P1P2*alpha + P1P3*beta;
        Qresult = Q1 + Q1Q2*gamma;
    }
    else
    {
        printf(" no result from LCP !\n");
//        alpha=0;
//        beta=0;
//        gamma=0;
        Presult = P1;
        Qresult = Q1;
    }

}

//------------------------------------------------------------------------------------------------------ //
// ---------------------------------------- DISTANCEPOINTTRI ------------------------------------------- //
//------------------------------------------------------------------------------------------------------ //


DistancePointTri::
DistancePointTri()
{
}

DistancePointTri::
~DistancePointTri()
{
}

void DistancePointTri::
NewComputation(const Vector3 &P1, const Vector3 &P2, const Vector3 &P3, const Vector3 &Q, Vector3 &Presult)
{
    double alpha, beta;
    int i;

    //Vector3 P1, P2, P3;

    Vector3 P1P2, P1P3, P1Q;
    LCPSolver<3>::Matrix  A;
    double          b[3], result[6];
    LCPSolver<3>    lcp;

    //P1 = tri->p1();
    //P2 = tri->p2();
    //P3 = tri->p3();


    for (i=0; i<6; i++)
    {
        result[i] = 0.0;
    }

    P1P2 = P2 - P1;
    P1P3 = P3 - P1;
    P1Q = Q - P1;

    // initialize A
    A[0][2] = 1.0;
    A[1][2] = 1.0;
    A[2][0] = -1.0; A[2][1] = -1.0; A[2][2] = 0.0;
    A[0][0] = dot(P1P2,P1P2);   A[0][1] = dot(P1P3,P1P2);
    A[1][0] = dot(P1P2,P1P3);   A[1][1] = dot(P1P3,P1P3);

    // initialize b
    b[2]=1.0;
    b[0]=-dot(P1Q,P1P2);  b[1]=-dot(P1Q,P1P3);

    if(lcp.solve(b, A, result))
    {
        alpha=result[3];
        beta=result[4];

        Presult = P1 + P1P2*alpha + P1P3*beta;
    }
    else
    {
        printf(" no result from LCP !\n");
//        alpha=0;
//        beta=0;
        Presult = P1;
    }
}


} // namespace helper

} // namespace sofa
