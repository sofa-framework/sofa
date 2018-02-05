/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
    std::vector< std::pair<int, double> > barycentricData(1);

    if (case1)
    {
        barycentricData[0] = std::pair<int, double> (index,1.0);

        return map1->addContactPointFromInputMapping(pos, barycentricData);
    }
    else
    {
        barycentricData[0] = std::pair<int, double> (index,1.0);

        return map2->addContactPointFromInputMapping(pos, barycentricData);
    }
}

template<>
int PersistentFrictionContact<LineModel, PointModel>::mapThePersistentContact(Vector3 &baryCoord, int index, Vector3 &pos, bool case1)
{
    if (case1)
    {
        Line l(this->model1, index);

        std::vector< std::pair<int, double> > barycentricData(2);
        barycentricData[0] = std::pair<int, double> (l.i1(), 1.0 - baryCoord[0]);
        barycentricData[1] = std::pair<int, double> (l.i2(), baryCoord[0]);

        return map1->addContactPointFromInputMapping(pos, barycentricData);
    }
    else
    {
        std::vector< std::pair<int, double> > barycentricData(1);
        barycentricData[0] = std::pair<int, double> (index,1.0);

        return map2->addContactPointFromInputMapping(pos, barycentricData);
    }
}

template<>
int PersistentFrictionContact<LineModel, LineModel>::mapThePersistentContact(Vector3 & baryCoord, int index, Vector3 &pos, bool case1)
{
    std::vector< std::pair<int, double> > barycentricData(2);

    if (case1)
    {
        Line l(this->model1, index);

        barycentricData[0] = std::pair<int, double> (l.i1(), 1.0 - baryCoord[0]);
        barycentricData[1] = std::pair<int, double> (l.i2(), baryCoord[0]);

        return map1->addContactPointFromInputMapping(pos, barycentricData);
    }
    else
    {
        Line l(this->model2, index);
        barycentricData[0] = std::pair<int, double> (l.i1(), 1.0 - baryCoord[0]);
        barycentricData[1] = std::pair<int, double> (l.i2(), baryCoord[0]);

        return map2->addContactPointFromInputMapping(pos, barycentricData);
    }
}

template<>
int PersistentFrictionContact<TriangleModel, PointModel>::mapThePersistentContact(Vector3 & baryCoord, int index, Vector3 &pos, bool case1)
{
    std::vector<std::pair<int, double> > barycentricData;

    if (case1)
    {
        Triangle t(this->model1, index);

        std::vector<std::pair<int, double> > barycentricData(3);
        barycentricData[0] = std::pair<int, double> (t.p1Index(), 1.0 - baryCoord[0] - baryCoord[1]);
        barycentricData[1] = std::pair<int, double> (t.p2Index(), baryCoord[0]);
        barycentricData[2] = std::pair<int, double> (t.p3Index(), baryCoord[1]);

        return map1->addContactPointFromInputMapping(pos, barycentricData);
    }
    else
    {
        std::vector<std::pair<int, double> > barycentricData(1);
        barycentricData[0] = std::pair<int, double> (index,1.0);

        return map2->addContactPointFromInputMapping(pos, barycentricData);
    }
}

} // namespace collision

} // namespace component

} // namespace sofa
