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
#pragma once

#include <sofa/component/collision/detection/algorithm/config.h>

#include <sofa/component/collision/geometry/CubeCollisionModel.h>

namespace sofa::component::collision::detection::algorithm
{

class EndPoint;

/**
 * SAPBox is a simple bounding box. It contains a Cube which contains only one final
 * CollisionElement and pointers to min and max EndPoints. min and max end points
 * are respectively min and max coordinates of the cube on a coordinate axis.
 * min and max are updated with the method update(int i), so min and max have
 * min/max values on the i-th axis after the method update(int i).
 */
class SOFA_COMPONENT_COLLISION_DETECTION_ALGORITHM_API DSAPBox
{
public:
    explicit DSAPBox(const collision::geometry::Cube &c, EndPoint *mi = nullptr, EndPoint *ma = nullptr) : cube(c), min(mi), max(ma)
    {}

    void update(int axis, double alarmDist);

    [[nodiscard]]
    double squaredDistance(const DSAPBox &other) const;

    /// Compute the squared distance from this to other on a specific axis
    [[nodiscard]]
    double squaredDistance(const DSAPBox &other, int axis) const;

    void show() const;

    collision::geometry::Cube cube;
    EndPoint *min{nullptr};
    EndPoint *max{nullptr};
};

}
