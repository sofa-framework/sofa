/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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

Creator<Contact::Factory, PersistentFrictionContact<geometry::PointCollisionModel<sofa::defaulttype::Vec3Types>, geometry::PointCollisionModel<sofa::defaulttype::Vec3Types>> > PointPointPersistentFrictionContactClass("PersistentFrictionContact",true);
Creator<Contact::Factory, PersistentFrictionContact<geometry::LineCollisionModel<sofa::defaulttype::Vec3Types>, geometry::PointCollisionModel<sofa::defaulttype::Vec3Types>> > LinePointPersistentFrictionContactClass("PersistentFrictionContact",true);
Creator<Contact::Factory, PersistentFrictionContact<geometry::LineCollisionModel<sofa::defaulttype::Vec3Types>, geometry::LineCollisionModel<sofa::defaulttype::Vec3Types>> > LineLinePersistentFrictionContactClass("PersistentFrictionContact",true);
Creator<Contact::Factory, PersistentFrictionContact<geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>, geometry::PointCollisionModel<sofa::defaulttype::Vec3Types>> > TrianglePointPersistentFrictionContactContactClass("PersistentFrictionContact",true);


template<>
int PersistentFrictionContact<geometry::PointCollisionModel<sofa::defaulttype::Vec3Types>, geometry::PointCollisionModel<sofa::defaulttype::Vec3Types>>::mapThePersistentContact(type::Vec3 &/*baryCoord*/, int index, type::Vec3 &pos, bool case1)
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
int PersistentFrictionContact<geometry::LineCollisionModel<sofa::defaulttype::Vec3Types>, geometry::PointCollisionModel<sofa::defaulttype::Vec3Types>>::mapThePersistentContact(type::Vec3 &baryCoord, int index, type::Vec3 &pos, bool case1)
{
    if (case1)
    {
        geometry::Line l(this->model1, index);

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
int PersistentFrictionContact<geometry::LineCollisionModel<sofa::defaulttype::Vec3Types>, geometry::LineCollisionModel<sofa::defaulttype::Vec3Types>>::mapThePersistentContact(type::Vec3 & baryCoord, int index, type::Vec3 &pos, bool case1)
{
    std::vector< std::pair<int, double> > barycentricData(2);

    if (case1)
    {
        geometry::Line l(this->model1, index);

        barycentricData[0] = std::pair<int, double> (l.i1(), 1.0 - baryCoord[0]);
        barycentricData[1] = std::pair<int, double> (l.i2(), baryCoord[0]);

        return map1->addContactPointFromInputMapping(pos, barycentricData);
    }
    else
    {
        geometry::Line l(this->model2, index);
        barycentricData[0] = std::pair<int, double> (l.i1(), 1.0 - baryCoord[0]);
        barycentricData[1] = std::pair<int, double> (l.i2(), baryCoord[0]);

        return map2->addContactPointFromInputMapping(pos, barycentricData);
    }
}

template<>
int PersistentFrictionContact<geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>, geometry::PointCollisionModel<sofa::defaulttype::Vec3Types>>::mapThePersistentContact(type::Vec3 & baryCoord, int index, type::Vec3 &pos, bool case1)
{
    if (case1)
    {
        geometry::Triangle t(this->model1, index);

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
