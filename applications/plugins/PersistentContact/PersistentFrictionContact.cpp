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
#include "PersistentFrictionContact.inl"

namespace sofa
{

namespace component
{

namespace collision
{

using namespace defaulttype;
using namespace sofa::helper;

SOFA_DECL_CLASS(PersistentFrictionContact)

Creator<Contact::Factory, PersistentFrictionContact<PointModel, PointModel> > PointPointPersistentFrictionContactClass("PersistentFrictionContact",true);
Creator<Contact::Factory, PersistentFrictionContact<LineModel, PointModel> > LinePointPersistentFrictionContactClass("PersistentFrictionContact",true);
Creator<Contact::Factory, PersistentFrictionContact<LineModel, LineModel> > LineLinePersistentFrictionContactClass("PersistentFrictionContact",true);
Creator<Contact::Factory, PersistentFrictionContact<TriangleModel, PointModel> > TrianglePointPersistentFrictionContactContactClass("PersistentFrictionContact",true);


template<>
int PersistentFrictionContact<PointModel, PointModel>::mapThePersistentContact(Vector3 &/*baryCoord*/, int index, Vector3 &pos, bool case1)
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
int PersistentFrictionContact<LineModel, PointModel>::mapThePersistentContact(Vector3 &baryCoord, int index, Vector3 &pos, bool case1)
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
int PersistentFrictionContact<LineModel, LineModel>::mapThePersistentContact(Vector3 & baryCoord, int index, Vector3 &pos, bool case1)
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
int PersistentFrictionContact<TriangleModel, PointModel>::mapThePersistentContact(Vector3 & baryCoord, int index, Vector3 &pos, bool case1)
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
