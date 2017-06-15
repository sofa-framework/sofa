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
#include <SofaBaseCollision/DiscreteIntersection.h>
#include <SofaConstraint/LMDNewProximityIntersection.inl>
#include <SofaMeshCollision/LineLocalMinDistanceFilter.h>
#include <SofaMeshCollision/PointLocalMinDistanceFilter.h>
#include <SofaMeshCollision/TriangleLocalMinDistanceFilter.h>
#include <SofaConstraint/LMDNewProximityIntersection.inl>
#include <SofaConstraint/LMDNewProximityIntersection.inl>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/Intersection.inl>


#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>

#include <sofa/helper/system/config.h>

#include <iostream>
#include <algorithm>

#include <sofa/helper/AdvancedTimer.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace helper;


void GetPosOfEdgeVertexOnTriangle(Vector3& pv1, Vector3& pv2, int edge_number, Triangle &t)
{
    sofa::core::topology::BaseMeshTopology::Edge edge = t.getCollisionModel()->getTopology()->getEdge(edge_number);
    core::behavior::MechanicalState<Vec3Types>* mState= t.getCollisionModel()->getMechanicalState();
    //core::behavior::MechanicalState<Vec3Types>::VecCoord* x =
    pv1= (mState->read(core::ConstVecCoordId::position())->getValue())[edge[0]];
    pv2= (mState->read(core::ConstVecCoordId::position())->getValue())[edge[1]];
}

SOFA_DECL_CLASS(LMDNewProximityIntersection)

int LMDNewProximityIntersectionClass = core::RegisterObject("Filtered optimized proximity intersection.")
        .add< LMDNewProximityIntersection >()
        ;

LMDNewProximityIntersection::LMDNewProximityIntersection()
    : BaseProximityIntersection()
    , useLineLine(initData(&useLineLine, true, "useLineLine", "Line-line collision detection enabled"))
{
}

void LMDNewProximityIntersection::init()
{
    intersectors.add<CubeModel, CubeModel, LMDNewProximityIntersection>(this);
    intersectors.add<PointModel, PointModel, LMDNewProximityIntersection>(this);
    intersectors.add<SphereModel, PointModel, LMDNewProximityIntersection>(this);
    intersectors.add<SphereModel, SphereModel, LMDNewProximityIntersection>(this);
    intersectors.add<LineModel, PointModel, LMDNewProximityIntersection>(this);
    intersectors.add<LineModel, SphereModel, LMDNewProximityIntersection>(this);
    intersectors.add<LineModel, LineModel, LMDNewProximityIntersection>(this);
    intersectors.add<TriangleModel, PointModel, LMDNewProximityIntersection>(this);
    intersectors.add<TriangleModel, SphereModel, LMDNewProximityIntersection>(this);
    intersectors.add<TriangleModel, LineModel, LMDNewProximityIntersection>(this);
    intersectors.add<TriangleModel, TriangleModel, LMDNewProximityIntersection>(this);

    intersectors.ignore<RayModel, PointModel>();
    intersectors.ignore<RayModel, LineModel>();
    intersectors.add<RayModel, TriangleModel, LMDNewProximityIntersection>(this);

	BaseProximityIntersection::init();
}

bool LMDNewProximityIntersection::testIntersection(Cube &cube1, Cube &cube2)
{
//     sofa::helper::AdvancedTimer::StepVar("testIntersectionCubeCube");
    const Vector3& minVect1 = cube1.minVect();
    const Vector3& minVect2 = cube2.minVect();
    const Vector3& maxVect1 = cube1.maxVect();
    const Vector3& maxVect2 = cube2.maxVect();

    const double alarmDist = getAlarmDistance() + cube1.getProximity() + cube2.getProximity();

    for (int i = 0; i < 3; i++)
    {
        if ( minVect1[i] > maxVect2[i] + alarmDist || minVect2[i] > maxVect1[i] + alarmDist )
            return false;
    }

    return true;
}


int LMDNewProximityIntersection::computeIntersection(Cube&, Cube&, OutputVector* /*contacts*/)
{
    return 0; /// \todo
}


bool LMDNewProximityIntersection::testIntersection(Point& e1, Point& e2)
{
    OutputVector contacts;
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();

    int n = doIntersectionPointPoint(alarmDist*alarmDist, e1.p(), e2.p(), &contacts, -1, e1.getIndex(), e2.getIndex(), *(e1.getCollisionModel()->getFilter()), *(e2.getCollisionModel()->getFilter()));

    return (n > 0);
}


int LMDNewProximityIntersection::computeIntersection(Point& e1, Point& e2, OutputVector* contacts)
{
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();
    std::cout<<"computeIntersection(Point& e1, Point& e2... is called"<<std::endl;
    int n = doIntersectionPointPoint(alarmDist*alarmDist, e1.p(), e2.p(), contacts
            , (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex()
            , e1.getIndex(), e2.getIndex(), *(e1.getCollisionModel()->getFilter())
            , *(e2.getCollisionModel()->getFilter()));

//	int n = doIntersectionPointPoint(alarmDist*alarmDist, e1.p(), e2.p(), contacts
//		, (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex());

    if ( n > 0)
    {
        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
        for (OutputVector::iterator detection = contacts->end()-n; detection != contacts->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->value -= contactDist;
        }
    }
    return n;
}


bool LMDNewProximityIntersection::testIntersection(Line&, Point&)
{
    serr << "Unnecessary call to NewProximityIntersection::testIntersection(Line,Point)."<<sendl;
    return true;
}


int LMDNewProximityIntersection::computeIntersection(Line& e1, Point& e2, OutputVector* contacts)
{
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();

    std::cout<<"computeIntersection(Line& e1, Point& e2... is called"<<std::endl;
    int id = e2.getIndex();
    int n = doIntersectionLinePoint(alarmDist*alarmDist, e1.p1(), e1.p2(), e2.p(), contacts, id
            , e1.getIndex(), e2.getIndex(), *(e1.getCollisionModel()->getFilter())
            , *(e2.getCollisionModel()->getFilter()));

    //int n = doIntersectionLinePoint(alarmDist*alarmDist, e1.p1(), e1.p2(), e2.p(), contacts, e2.getIndex());

    if ( n > 0)
    {
        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
        for (OutputVector::iterator detection = contacts->end()-n; detection != contacts->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->value -= contactDist;
        }
    }
    return n;
}


bool LMDNewProximityIntersection::testIntersection(Line&, Line&)
{
    serr << "Unnecessary call to NewProximityIntersection::testIntersection(Line,Line)."<<sendl;
    return true;
}


int LMDNewProximityIntersection::computeIntersection(Line& e1, Line& e2, OutputVector* contacts)
{
    std::cout<<"computeIntersection(Line& e1, Line& e2... is called"<<std::endl;
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const double dist2 = alarmDist*alarmDist;
    const int id = (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex();

    int n = doIntersectionLineLine(dist2, e1.p1(), e1.p2(), e2.p1(), e2.p2(), contacts, id
            , e1.getIndex(), e2.getIndex(), *(e1.getCollisionModel()->getFilter())
            , *(e2.getCollisionModel()->getFilter()));


    if ( n > 0)
    {
        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
        for (OutputVector::iterator detection = contacts->end()-n; detection != contacts->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->value -= contactDist;
        }
    }
    return n;
}


bool LMDNewProximityIntersection::testIntersection(Triangle&, Point&)
{
    serr << "Unnecessary call to NewProximityIntersection::testIntersection(Triangle,Point)."<<sendl;
    return true;
}


int LMDNewProximityIntersection::computeIntersection(Triangle& e1, Point& e2, OutputVector* contacts)
{

// index of lines:
    const fixed_array<unsigned int,3>& edgesInTriangle1 = e1.getCollisionModel()->getTopology()->getEdgesInTriangle(e1.getIndex());
    unsigned int E1edge1verif, E1edge2verif, E1edge3verif;
    E1edge1verif=0; E1edge2verif=0; E1edge3verif=0;

    // verify the edge ordering //
    sofa::core::topology::BaseMeshTopology::Edge edge[3];
    //std::cout<<"E1 & E2 verif: ";
    for (int i=0; i<3; i++)
    {
        // Verify for E1: convention: Edge1 = P1 P2    Edge2 = P2 P3    Edge3 = P3 P1
        edge[i] = e1.getCollisionModel()->getTopology()->getEdge(edgesInTriangle1[i]);
        if(((int)edge[i][0]==e1.p1Index() && (int)edge[i][1]==e1.p2Index()) || ((int)edge[i][0]==e1.p2Index() && (int)edge[i][1]==e1.p1Index()))
        {
            E1edge1verif = edgesInTriangle1[i]; /*std::cout<<"- e1 1: "<<i ;*/
        }
        if(((int)edge[i][0]==e1.p2Index() && (int)edge[i][1]==e1.p3Index()) || ((int)edge[i][0]==e1.p3Index() && (int)edge[i][1]==e1.p2Index()))
        {
            E1edge2verif = edgesInTriangle1[i]; /*std::cout<<"- e1 2: "<<i ;*/
        }
        if(((int)edge[i][0]==e1.p1Index() && (int)edge[i][1]==e1.p3Index()) || ((int)edge[i][0]==e1.p3Index() && (int)edge[i][1]==e1.p1Index()))
        {
            E1edge3verif = edgesInTriangle1[i]; /*std::cout<<"- e1 3: "<<i ;*/
        }
    }

    unsigned int e1_edgesIndex[3];
    e1_edgesIndex[0]=E1edge1verif; e1_edgesIndex[1]=E1edge2verif; e1_edgesIndex[2]=E1edge3verif;


    std::cout<<"computeIntersection(Triangle& e1, Point& e2... is called"<<std::endl;
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const double dist2 = alarmDist*alarmDist;

    int id = e2.getIndex();
    int n = doIntersectionTrianglePoint(dist2, e1.flags(), e1.p1(), e1.p2(), e1.p3(), e1.n(), e2.p(), contacts, id
            , e1, e1_edgesIndex, e2.getIndex() , *(e1.getCollisionModel()->getFilter())
            , *(e2.getCollisionModel()->getFilter()));

//	int n = doIntersectionTrianglePoint(dist2, e1.flags(), e1.p1(), e1.p2(), e1.p3(), e1.n(), e2.p(), contacts, e2.getIndex());

    if ( n > 0)
    {
        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
        for (OutputVector::iterator detection = contacts->end()-n; detection != contacts->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->value -= contactDist;
        }
    }
    return n;
}


bool LMDNewProximityIntersection::testIntersection(Triangle&, Line&)
{
    serr << "Unnecessary call to NewProximityIntersection::testIntersection(Triangle& e1, Line& e2)."<<sendl;
    return true;
}


int LMDNewProximityIntersection::computeIntersection(Triangle& e1, Line& e2, OutputVector* contacts)
{

// index of lines:
    const fixed_array<unsigned int,3>& edgesInTriangle1 = e1.getCollisionModel()->getTopology()->getEdgesInTriangle(e1.getIndex());
    unsigned int E1edge1verif, E1edge2verif, E1edge3verif;
    E1edge1verif=0; E1edge2verif=0; E1edge3verif=0;

    // verify the edge ordering //
    sofa::core::topology::BaseMeshTopology::Edge edge[3];
    //std::cout<<"E1 & E2 verif: ";
    for (int i=0; i<3; i++)
    {
        // Verify for E1: convention: Edge1 = P1 P2    Edge2 = P2 P3    Edge3 = P3 P1
        edge[i] = e1.getCollisionModel()->getTopology()->getEdge(edgesInTriangle1[i]);
        if(((int)edge[i][0]==e1.p1Index() && (int)edge[i][1]==e1.p2Index()) || ((int)edge[i][0]==e1.p2Index() && (int)edge[i][1]==e1.p1Index()))
        {
            E1edge1verif = edgesInTriangle1[i]; /*std::cout<<"- e1 1: "<<i ;*/
        }
        if(((int)edge[i][0]==e1.p2Index() && (int)edge[i][1]==e1.p3Index()) || ((int)edge[i][0]==e1.p3Index() && (int)edge[i][1]==e1.p2Index()))
        {
            E1edge2verif = edgesInTriangle1[i]; /*std::cout<<"- e1 2: "<<i ;*/
        }
        if(((int)edge[i][0]==e1.p1Index() &&(int) edge[i][1]==e1.p3Index()) || ((int)edge[i][0]==e1.p3Index() && (int)edge[i][1]==e1.p1Index()))
        {
            E1edge3verif = edgesInTriangle1[i]; /*std::cout<<"- e1 3: "<<i ;*/
        }
    }

    unsigned int e1_edgesIndex[3];
    e1_edgesIndex[0]=E1edge1verif; e1_edgesIndex[1]=E1edge2verif; e1_edgesIndex[2]=E1edge3verif;



    std::cout<<"computeIntersection(Triangle& e1, Line& e2... is called"<<std::endl;
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const double dist2 = alarmDist*alarmDist;
    const Vector3& p1 = e1.p1();
    const Vector3& p2 = e1.p2();
    const Vector3& p3 = e1.p3();
    const Vector3& pn = e1.n();
    const Vector3& q1 = e2.p1();
    const Vector3& q2 = e2.p2();

    const int f1 = e1.flags();

    int n = 0;
    int id= e2.getIndex();

    if (f1&TriangleModel::FLAG_P1)
    {
        n += doIntersectionLinePoint(dist2, q1, q2, p1, contacts, id, e1.getIndex(), e2.getIndex() , *(e1.getCollisionModel()->getFilter()), *(e2.getCollisionModel()->getFilter()), true);

    }
    if (f1&TriangleModel::FLAG_P2)
    {
        n += doIntersectionLinePoint(dist2, q1, q2, p2, contacts, id, e1.getIndex(), e2.getIndex() , *(e1.getCollisionModel()->getFilter()), *(e2.getCollisionModel()->getFilter()), true);

    }
    if (f1&TriangleModel::FLAG_P3)
    {
        n += doIntersectionLinePoint(dist2, q1, q2, p3, contacts, id, e1.getIndex(), e2.getIndex() , *(e1.getCollisionModel()->getFilter()), *(e2.getCollisionModel()->getFilter()), true);

    }

    n += doIntersectionTrianglePoint(dist2, f1, p1, p2, p3, pn, q1, contacts, id ,e1, e1_edgesIndex, e2.getIndex(), *(e1.getCollisionModel()->getFilter()), *(e2.getCollisionModel()->getFilter()), false);


    n += doIntersectionTrianglePoint(dist2, f1, p1, p2, p3, pn, q2, contacts, id, e1, e1_edgesIndex, e2.getIndex(),*(e1.getCollisionModel()->getFilter()), *(e2.getCollisionModel()->getFilter()), false);


    if (useLineLine.getValue())
    {
        if (f1&TriangleModel::FLAG_E12)
            n += doIntersectionLineLine(dist2, p1, p2, q1, q2, contacts, id, e1.getIndex(), e2.getIndex(), *(e1.getCollisionModel()->getFilter()), *(e2.getCollisionModel()->getFilter()));

        if (f1&TriangleModel::FLAG_E23)
            n += doIntersectionLineLine(dist2, p2, p3, q1, q2, contacts, id, e1.getIndex(), e2.getIndex(), *(e1.getCollisionModel()->getFilter()), *(e2.getCollisionModel()->getFilter()));

        if (f1&TriangleModel::FLAG_E31)
            n += doIntersectionLineLine(dist2, p3, p1, q1, q2, contacts, id, e1.getIndex(), e2.getIndex(), *(e1.getCollisionModel()->getFilter()), *(e2.getCollisionModel()->getFilter()));

    }

    if ( n > 0)
    {
        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
        for (OutputVector::iterator detection = contacts->end()-n; detection != contacts->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->value -= contactDist;
        }
    }
    return n;
}


bool LMDNewProximityIntersection::testIntersection(Triangle&, Triangle&)
{
    serr << "Unnecessary call to NewProximityIntersection::testIntersection(Triangle& e1, Triangle& e2)."<<sendl;
    return true;
}


int LMDNewProximityIntersection::computeIntersection(Triangle& e1, Triangle& e2, OutputVector* contacts)
{
    //sofa::helper::AdvancedTimer::StepVar("ComputeIntersectionOnTri");


    //std::cout<<"computeIntersection(Triangle "<<e1.getIndex()<<", Triangle"<< e2.getIndex()<<" ... is called"<<std::endl;

    if (e1.getIndex() >= e1.getCollisionModel()->getSize())
    {
        serr << "NewProximityIntersection::computeIntersection(Triangle, Triangle): ERROR invalid e1 index "
                << e1.getIndex() << " on CM " << e1.getCollisionModel()->getName() << " of size " << e1.getCollisionModel()->getSize()<<sendl;
        return 0;
    }

    if (e2.getIndex() >= e2.getCollisionModel()->getSize())
    {
        serr << "NewProximityIntersection::computeIntersection(Triangle, Triangle): ERROR invalid e2 index "
                << e2.getIndex() << " on CM " << e2.getCollisionModel()->getName() << " of size " << e2.getCollisionModel()->getSize()<<sendl;
        return 0;
    }



    // index of lines:
    const fixed_array<unsigned int,3>& edgesInTriangle1 = e1.getCollisionModel()->getTopology()->getEdgesInTriangle(e1.getIndex());
    const fixed_array<unsigned int,3>& edgesInTriangle2 = e2.getCollisionModel()->getTopology()->getEdgesInTriangle(e2.getIndex());

    unsigned int E1edge1verif, E1edge2verif, E1edge3verif;
    unsigned int E2edge1verif, E2edge2verif, E2edge3verif;
    E1edge1verif=0; E1edge2verif=0; E1edge3verif=0;
    E2edge1verif=0; E2edge2verif=0; E2edge3verif=0;


    // verify the edge ordering //
    sofa::core::topology::BaseMeshTopology::Edge edge[3];
    //std::cout<<"E1 & E2 verif: ";
    for (int i=0; i<3; i++)
    {
        // Verify for E1: convention: Edge1 = P1 P2    Edge2 = P2 P3    Edge3 = P3 P1
        edge[i] = e1.getCollisionModel()->getTopology()->getEdge(edgesInTriangle1[i]);
        if(((int)edge[i][0]==e1.p1Index() && (int)edge[i][1]==e1.p2Index()) || ((int)edge[i][0]==e1.p2Index() && (int)edge[i][1]==e1.p1Index()))
        {
            E1edge1verif = edgesInTriangle1[i]; /*std::cout<<"- e1 1: "<<i ;*/
        }
        if(((int)edge[i][0]==e1.p2Index() && (int)edge[i][1]==e1.p3Index()) || ((int)edge[i][0]==e1.p3Index() && (int)edge[i][1]==e1.p2Index()))
        {
            E1edge2verif = edgesInTriangle1[i]; /*std::cout<<"- e1 2: "<<i ;*/
        }
        if(((int)edge[i][0]==e1.p1Index() && (int)edge[i][1]==e1.p3Index()) || ((int)edge[i][0]==e1.p3Index() && (int)edge[i][1]==e1.p1Index()))
        {
            E1edge3verif = edgesInTriangle1[i]; /*std::cout<<"- e1 3: "<<i ;*/
        }
        // Verify for E2: convention: Edge1 = P1 P2    Edge2 = P2 P3    Edge3 = P3 P1
        edge[i] = e2.getCollisionModel()->getTopology()->getEdge(edgesInTriangle2[i]);
        if(((int)edge[i][0]==e2.p1Index() && (int)edge[i][1]==e2.p2Index()) || ((int)edge[i][0]==e2.p2Index() && (int)edge[i][1]==e2.p1Index()))
        {
            E2edge1verif = edgesInTriangle2[i];/*std::cout<<"- e2 1: "<<i ;*/
        }
        if(((int)edge[i][0]==e2.p2Index() && (int)edge[i][1]==e2.p3Index()) || ((int)edge[i][0]==e2.p3Index() && (int)edge[i][1]==e2.p2Index()))
        {
            E2edge2verif = edgesInTriangle2[i];/*std::cout<<"- e2 2: "<<i ;*/
        }
        if(((int)edge[i][0]==e2.p1Index() && (int)edge[i][1]==e2.p3Index()) || ((int)edge[i][0]==e2.p3Index() && (int)edge[i][1]==e2.p1Index()))
        {
            E2edge3verif = edgesInTriangle2[i]; /*std::cout<<"- e2 3: "<<i ;*/
        }
    }

    unsigned int e1_edgesIndex[3],e2_edgesIndex[3];
    e1_edgesIndex[0]=E1edge1verif; e1_edgesIndex[1]=E1edge2verif; e1_edgesIndex[2]=E1edge3verif;
    e2_edgesIndex[0]=E2edge1verif; e2_edgesIndex[1]=E2edge2verif; e2_edgesIndex[2]=E2edge3verif;



    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity();
    const double dist2 = alarmDist*alarmDist;
    const Vector3& p1 = e1.p1();
    const Vector3& p2 = e1.p2();
    const Vector3& p3 = e1.p3();
    const Vector3& pn = e1.n();
    const Vector3& q1 = e2.p1();
    const Vector3& q2 = e2.p2();
    const Vector3& q3 = e2.p3();
    const Vector3& qn = e2.n();



    const int f1 = e1.flags();
    const int f2 = e2.flags();

    const int id1 = e1.getIndex()*3; // index of contacts involving points in e1
    const int id2 = e1.getCollisionModel()->getSize()*3 + e2.getIndex()*12; // index of contacts involving points in e2


    int n = 0;






    if (f1&TriangleModel::FLAG_P1)
        n += doIntersectionTrianglePoint(dist2, f2, q1, q2, q3, qn, p1, contacts, id1+0, e2, e2_edgesIndex, e1.p1Index(), *(e2.getCollisionModel()->getFilter()), *(e1.getCollisionModel()->getFilter()), true);
    //	n += doIntersectionTrianglePoint(dist2, f2, q1, q2, q3, qn, p1, contacts, id1+0, true);
    if (f1&TriangleModel::FLAG_P2)
        n += doIntersectionTrianglePoint(dist2, f2, q1, q2, q3, qn, p2, contacts, id1+1, e2, e2_edgesIndex, e1.p2Index(), *(e2.getCollisionModel()->getFilter()), *(e1.getCollisionModel()->getFilter()), true);
    //	n += doIntersectionTrianglePoint(dist2, f2, q1, q2, q3, qn, p2, contacts, id1+1, true);
    if (f1&TriangleModel::FLAG_P3)
        n += doIntersectionTrianglePoint(dist2, f2, q1, q2, q3, qn, p3, contacts, id1+2, e2, e2_edgesIndex, e1.p3Index(), *(e2.getCollisionModel()->getFilter()), *(e1.getCollisionModel()->getFilter()), true);
    //	n += doIntersectionTrianglePoint(dist2, f2, q1, q2, q3, qn, p3, contacts, id1+2, true);

    if (f2&TriangleModel::FLAG_P1)
        n += doIntersectionTrianglePoint(dist2, f1, p1, p2, p3, pn, q1, contacts, id2+0, e1, e1_edgesIndex, e2.p1Index(), *(e1.getCollisionModel()->getFilter()), *(e2.getCollisionModel()->getFilter()), false);
    //	n += doIntersectionTrianglePoint(dist2, f1, p1, p2, p3, pn, q1, contacts, id2+0, false);
    if (f2&TriangleModel::FLAG_P2)
        n += doIntersectionTrianglePoint(dist2, f1, p1, p2, p3, pn, q2, contacts, id2+1, e1, e1_edgesIndex, e2.p2Index(), *(e1.getCollisionModel()->getFilter()), *(e2.getCollisionModel()->getFilter()), false);
    //	n += doIntersectionTrianglePoint(dist2, f1, p1, p2, p3, pn, q1, contacts, id2+0, false);
    if (f2&TriangleModel::FLAG_P3)
        n += doIntersectionTrianglePoint(dist2, f1, p1, p2, p3, pn, q3, contacts, id2+2, e1, e1_edgesIndex, e2.p3Index(), *(e1.getCollisionModel()->getFilter()), *(e2.getCollisionModel()->getFilter()), false);
    //	n += doIntersectionTrianglePoint(dist2, f1, p1, p2, p3, pn, q1, contacts, id2+0, false);

    if (useLineLine.getValue())
    {

        //std::cout<<" "<<std::endl;

        //<<E1edge1verif<<" "<<E1edge2verif<<" "<<E1edge3verif<<"  - E2verif: "<<E2edge1verif<<" "<<E2edge2verif<<" "<<E2edge3verif<<std::endl;


        //if(e1.getIndex()==23 || e2.getIndex()==23 || e1.getIndex()==46 || e2.getIndex()==46)
        //	std::cout <<"triangle "<<e1.getIndex() <<" is tested with triangle"<< e2.getIndex()<<"-   f1 = "<<f1<<"  - f2 = "<<f2<<std::endl;

        Vector3 e1_p1, e1_p2, e1_p3,e2_q1 , e2_q2,e2_q3;

        if (f1&TriangleModel::FLAG_E12)
        {
            GetPosOfEdgeVertexOnTriangle(e1_p1,e1_p2,edgesInTriangle1[0],e1);

            if (f2&TriangleModel::FLAG_E12)
            {
                // look for the first edge of the triangle (given by edgesInTriangle1[0] )
                GetPosOfEdgeVertexOnTriangle(e2_q1,e2_q2,edgesInTriangle2[0],e2);
                n += doIntersectionLineLine(dist2, e1_p1, e1_p2, e2_q1, e2_q2, contacts, id2+3, edgesInTriangle1[0], edgesInTriangle2[0], *(e1.getCollisionModel()->getFilter()), *(e2.getCollisionModel()->getFilter()));
            }
            if (f2&TriangleModel::FLAG_E23)
            {
                GetPosOfEdgeVertexOnTriangle(e2_q2,e2_q3,edgesInTriangle2[1],e2);
                n += doIntersectionLineLine(dist2, e1_p1, e1_p2, e2_q2, e2_q3, contacts, id2+4, edgesInTriangle1[0], edgesInTriangle2[1], *(e1.getCollisionModel()->getFilter()), *(e2.getCollisionModel()->getFilter()));
            }
            if (f2&TriangleModel::FLAG_E31)
            {
                GetPosOfEdgeVertexOnTriangle(e2_q3,e2_q1,edgesInTriangle2[2],e2);
                n += doIntersectionLineLine(dist2, e1_p1, e1_p2, e2_q3, e2_q1, contacts, id2+5, edgesInTriangle1[0], edgesInTriangle2[2], *(e1.getCollisionModel()->getFilter()), *(e2.getCollisionModel()->getFilter()));
            }
        }

        if (f1&TriangleModel::FLAG_E23)
        {
            GetPosOfEdgeVertexOnTriangle(e1_p2,e1_p3,edgesInTriangle1[1],e1);

            if (f2&TriangleModel::FLAG_E12)
            {
                GetPosOfEdgeVertexOnTriangle(e2_q1,e2_q2,edgesInTriangle2[0],e2);
                n += doIntersectionLineLine(dist2, e1_p2, e1_p3, e2_q1, e2_q2, contacts, id2+6, edgesInTriangle1[1], edgesInTriangle2[0], *(e1.getCollisionModel()->getFilter()), *(e2.getCollisionModel()->getFilter()));
            }
            if (f2&TriangleModel::FLAG_E23)
            {
                GetPosOfEdgeVertexOnTriangle(e2_q2,e2_q3,edgesInTriangle2[1],e2);
                n += doIntersectionLineLine(dist2, e1_p2, e1_p3, e2_q2, e2_q3, contacts, id2+7, edgesInTriangle1[1], edgesInTriangle2[1], *(e1.getCollisionModel()->getFilter()), *(e2.getCollisionModel()->getFilter()));
            }
            if (f2&TriangleModel::FLAG_E31)
            {
                GetPosOfEdgeVertexOnTriangle(e2_q3,e2_q1,edgesInTriangle2[2],e2);
                n += doIntersectionLineLine(dist2, e1_p2, e1_p3, e2_q3, e2_q1, contacts, id2+8, edgesInTriangle1[1], edgesInTriangle2[2], *(e1.getCollisionModel()->getFilter()), *(e2.getCollisionModel()->getFilter()));
            }
        }

        if (f1&TriangleModel::FLAG_E31)
        {
            GetPosOfEdgeVertexOnTriangle(e1_p3,e1_p1,edgesInTriangle1[2],e1);
            if (f2&TriangleModel::FLAG_E12)
            {
                GetPosOfEdgeVertexOnTriangle(e2_q1,e2_q2,edgesInTriangle2[0],e2);
                n += doIntersectionLineLine(dist2, e1_p3, e1_p1, e2_q1, e2_q2, contacts, id2+9, edgesInTriangle1[2], edgesInTriangle2[0], *(e1.getCollisionModel()->getFilter()), *(e2.getCollisionModel()->getFilter()));
            }
            if (f2&TriangleModel::FLAG_E23)
            {
                GetPosOfEdgeVertexOnTriangle(e2_q2,e2_q3,edgesInTriangle2[1],e2);
                n += doIntersectionLineLine(dist2, e1_p3, e1_p1, e2_q2, e2_q3, contacts, id2+10, edgesInTriangle1[2], edgesInTriangle2[1], *(e1.getCollisionModel()->getFilter()), *(e2.getCollisionModel()->getFilter()));
            }
            if (f2&TriangleModel::FLAG_E31)
            {
                GetPosOfEdgeVertexOnTriangle(e2_q3,e2_q1,edgesInTriangle2[2],e2);
                n += doIntersectionLineLine(dist2, e1_p3, e1_p1, e2_q3, e2_q1, contacts, id2+11, edgesInTriangle1[2], edgesInTriangle2[2], *(e1.getCollisionModel()->getFilter()), *(e2.getCollisionModel()->getFilter()));
            }
        }
    }

    if (n > 0)
    {
        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity();
        for (int i = 0; i < n; ++i)
        {
            (*contacts)[contacts->size()-n+i].elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            (*contacts)[contacts->size()-n+i].value -= contactDist;
        }
    }
    return n;
}


bool LMDNewProximityIntersection::testIntersection(Ray &t1,Triangle &t2)
{
    Vector3 P,Q,PQ;
    static DistanceSegTri proximitySolver;
    const double alarmDist = getAlarmDistance() + t1.getProximity() + t2.getProximity();

    if (fabs(t2.n() * t1.direction()) < 0.000001)
        return false; // no intersection for edges parallel to the triangle

    Vector3 A = t1.origin();
    Vector3 B = A + t1.direction() * t1.l();

    proximitySolver.NewComputation( t2.p1(), t2.p2(), t2.p3(), A, B,P,Q);
    PQ = Q-P;

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        //sout<<"Collision between Line - Triangle"<<sendl;
        return true;
    }
    else
        return false;
}


int LMDNewProximityIntersection::computeIntersection(Ray &t1, Triangle &t2, OutputVector* contacts)
{
    const double alarmDist = getAlarmDistance() + t1.getProximity() + t2.getProximity();

    if (fabs(t2.n() * t1.direction()) < 0.000001)
        return false; // no intersection for edges parallel to the triangle

    Vector3 A = t1.origin();
    Vector3 B = A + t1.direction() * t1.l();

    Vector3 P,Q,PQ;
    static DistanceSegTri proximitySolver;

    proximitySolver.NewComputation( t2.p1(), t2.p2(), t2.p3(), A,B,P,Q);
    PQ = Q-P;

    if (PQ.norm2() >= alarmDist*alarmDist)
        return 0;

    const double contactDist = alarmDist;
    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(t1, t2);
    detection->point[1]=P;
    detection->point[0]=Q;
#ifdef DETECTIONOUTPUT_FREEMOTION
    detection->freePoint[1] = P;
    detection->freePoint[0] = Q;
#endif
    detection->normal=-t2.n();
    detection->value = PQ.norm();
    detection->value -= contactDist;
    return 1;
}





} // namespace collision

} // namespace component

} // namespace sofa

