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
#include <sofa/component/collision/detection/algorithm/DSAPBox.h>
#include <sofa/component/collision/detection/algorithm/EndPoint.h>

namespace sofa::component::collision::detection::algorithm
{

void DSAPBox::update(int axis, double alarmDist)
{
    min->value = (cube.minVect())[axis] - alarmDist;
    max->value = (cube.maxVect())[axis] + alarmDist;
}

double DSAPBox::squaredDistance(const DSAPBox &other) const
{
    double dist2 = 0;

    for (int axis = 0; axis < 3; ++axis)
    {
        dist2 += squaredDistance(other, axis);
    }

    return dist2;
}

double DSAPBox::squaredDistance(const DSAPBox &other, int axis) const
{
    const type::Vec3 &min0 = this->cube.minVect();
    const type::Vec3 &max0 = this->cube.maxVect();
    const type::Vec3 &min1 = other.cube.minVect();
    const type::Vec3 &max1 = other.cube.maxVect();

    if (min0[axis] > max1[axis])
    {
        return std::pow(min0[axis] - max1[axis], 2);
    }

    if (min1[axis] > max0[axis])
    {
        return std::pow(min1[axis] - max0[axis], 2);
    }

    return 0;
}
}