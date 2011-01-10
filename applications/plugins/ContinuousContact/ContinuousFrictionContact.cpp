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
#include "ContinuousFrictionContact.inl"

namespace sofa
{

namespace component
{

namespace collision
{

using namespace defaulttype;
using namespace sofa::helper;

SOFA_DECL_CLASS(ContinuousFrictionContact)

Creator<Contact::Factory, ContinuousFrictionContact<PointModel, PointModel> > PointPointContinuousFrictionContactClass("ContinuousFrictionContact",true);
Creator<Contact::Factory, ContinuousFrictionContact<LineModel, PointModel> > LinePointContinuousFrictionContactClass("ContinuousFrictionContact",true);
Creator<Contact::Factory, ContinuousFrictionContact<LineModel, LineModel> > LineLineContinuousFrictionContactClass("ContinuousFrictionContact",true);
Creator<Contact::Factory, ContinuousFrictionContact<TriangleModel, PointModel> > TrianglePointContinuousFrictionContactContactClass("ContinuousFrictionContact",true);


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
