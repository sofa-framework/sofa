/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/collision/FrictionContact.inl>
#include <sofa/component/collision/BarycentricContactMapper.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace defaulttype;
using namespace sofa::helper;
using simulation::Node;

sofa::core::collision::DetectionOutput::ContactId Identifier::cpt=0;
std::list<sofa::core::collision::DetectionOutput::ContactId> Identifier::availableId;

SOFA_DECL_CLASS(FrictionContact)
SOFA_DECL_CLASS(ContinuousFrictionContact)
Creator<Contact::Factory, ContinuousFrictionContact<PointModel, PointModel> > PointPointContinuousFrictionContactClass("ContinuousFrictionContact",true);
Creator<Contact::Factory, ContinuousFrictionContact<LineModel, PointModel> > LinePointContinuousFrictionContactClass("ContinuousFrictionContact",true);
Creator<Contact::Factory, ContinuousFrictionContact<LineModel, LineModel> > LineLineContinuousFrictionContactClass("ContinuousFrictionContact",true);
Creator<Contact::Factory, ContinuousFrictionContact<TriangleModel, PointModel> > TrianglePointContinuousFrictionContactContactClass("ContinuousFrictionContact",true);


Creator<Contact::Factory, FrictionContact<PointModel, PointModel> > PointPointFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<LineModel, SphereModel> > LineSphereFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<LineModel, PointModel> > LinePointFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<LineModel, LineModel> > LineLineFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TriangleModel, SphereModel> > TriangleSphereFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TriangleModel, PointModel> > TrianglePointFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TriangleModel, LineModel> > TriangleLineFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TriangleModel, TriangleModel> > TriangleTriangleFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronModel, SphereModel> > TetrahedronSphereFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronModel, PointModel> > TetrahedronPointFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronModel, LineModel> > TetrahedronLineFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronModel, TriangleModel> > TetrahedronTriangleFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronModel, TetrahedronModel> > TetrahedronTetrahedronFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<SphereModel, SphereModel> > SphereSphereFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<SphereModel, PointModel> > SpherePointFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<SphereTreeModel, SphereTreeModel> > SphereTreeSphereTreeFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<SphereTreeModel, TriangleModel> > SphereTreeTriangleFrictionContactClass("FrictionContact", true);

Creator<Contact::Factory, FrictionContact<TetrahedronModel, SphereModel> > TetrahedronSpherePenalityFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronModel, PointModel> > TetrahedronPointPenalityFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronModel, LineModel> > TetrahedronLinePenalityFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronModel, TriangleModel> > TetrahedronTrianglePenalityFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronModel, TetrahedronModel> > TetrahedronTetrahedronPenalityFrictionContactClass("FrictionContact",true);

Creator<Contact::Factory, FrictionContact<RigidDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > DistanceGridDistanceGridFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<RigidDistanceGridCollisionModel, PointModel> > DistanceGridPointFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<RigidDistanceGridCollisionModel, SphereModel> > DistanceGridSphereFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<RigidDistanceGridCollisionModel, TriangleModel> > DistanceGridTriangleFrictionContactClass("FrictionContact", true);

Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, FFDDistanceGridCollisionModel> > FFDDistanceGridFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > FFDDistanceGridRigidDistanceGridFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, PointModel> > FFDDistanceGridPointFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, SphereModel> > FFDDistanceGridSphereFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, TriangleModel> > FFDDistanceGridTriangleFrictionContactClass("FrictionContact", true);


template<>
int ContinuousFrictionContact<PointModel, PointModel>::mapTheContinuousContact(Vector3 &/*baryCoord*/, int index, Vector3 &pos, bool case1)
{
    std::vector<std::pair<int, double> > barycentricData;

    if (case1)
    {
        barycentricData.push_back( std::pair<int, double> (index,1.0) );
        return map1->addContactPointFromInputMapping(pos, barycentricData);
    }
    else
    {
        barycentricData.push_back( std::pair<int, double> (index,1.0) );
        return map2->addContactPointFromInputMapping(pos, barycentricData);
    }
}

template<>
int ContinuousFrictionContact<LineModel, PointModel>::mapTheContinuousContact(Vector3 &baryCoord, int index, Vector3 &pos, bool case1)
{
    std::vector<std::pair<int, double> > barycentricData;

    if (case1)
    {
        Line *l=new Line(this->model1, index);
        barycentricData.push_back( std::pair<int, double> (l->i1(),1.0-baryCoord[0]) );
        barycentricData.push_back( std::pair<int, double> (l->i2(),baryCoord[0]) );
        return map1->addContactPointFromInputMapping(pos, barycentricData);
    }
    else
    {
        barycentricData.push_back( std::pair<int, double> (index,1.0) );
        return map2->addContactPointFromInputMapping(pos, barycentricData);
    }
}

template<>
int ContinuousFrictionContact<LineModel, LineModel>::mapTheContinuousContact(Vector3 & baryCoord, int index, Vector3 &pos, bool case1)
{
    std::vector<std::pair<int, double> > barycentricData;

    if (case1)
    {
        Line *l=new Line(this->model1, index);
        barycentricData.push_back( std::pair<int, double> (l->i1(),1.0-baryCoord[0]) );
        barycentricData.push_back( std::pair<int, double> (l->i2(),baryCoord[0]) );
        return map1->addContactPointFromInputMapping(pos, barycentricData);
    }
    else
    {
        Line *l=new Line(this->model2, index);
        barycentricData.push_back( std::pair<int, double> (l->i1(),1.0-baryCoord[0]) );
        barycentricData.push_back( std::pair<int, double> (l->i2(),baryCoord[0]) );
        return map2->addContactPointFromInputMapping(pos, barycentricData);
    }
}

template<>
int ContinuousFrictionContact<TriangleModel, PointModel>::mapTheContinuousContact(Vector3 & baryCoord, int index, Vector3 &pos, bool case1)
{
    std::vector<std::pair<int, double> > barycentricData;

    if (case1)
    {
        Triangle *t=new Triangle(this->model1, index);
        barycentricData.push_back( std::pair<int, double> (t->p1Index(),1.0-baryCoord[0]-baryCoord[1]) );
        barycentricData.push_back( std::pair<int, double> (t->p2Index(),baryCoord[0]) );
        barycentricData.push_back( std::pair<int, double> (t->p3Index(),baryCoord[1]) );
        return map1->addContactPointFromInputMapping(pos, barycentricData);
    }
    else
    {
        barycentricData.push_back( std::pair<int, double> (index,1.0) );
        return map2->addContactPointFromInputMapping(pos, barycentricData);
    }
}

} // namespace collision

} // namespace component

} // namespace sofa
