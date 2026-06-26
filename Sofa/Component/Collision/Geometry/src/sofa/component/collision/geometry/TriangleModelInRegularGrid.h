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
#include <sofa/component/collision/geometry/config.h>

#include <sofa/component/collision/geometry/TriangleCollisionModel.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa::component::collision::geometry
{


class SOFA_COMPONENT_COLLISION_GEOMETRY_API TriangleModelInRegularGrid : public TriangleCollisionModel<sofa::defaulttype::Vec3Types>
{
public:
    SOFA_CLASS(TriangleModelInRegularGrid, TriangleCollisionModel<sofa::defaulttype::Vec3Types>);

    void init() override;
    void computeBoundingTree ( int maxDepth=0 ) override;

    sofa::core::topology::BaseMeshTopology* _topology;
    sofa::core::topology::BaseMeshTopology* _higher_topo;
    core::behavior::MechanicalState<defaulttype::Vec3Types>* _higher_mstate;

protected:
    TriangleModelInRegularGrid();
    ~TriangleModelInRegularGrid();
};

}  // namespace sofa::component::collision::geometry
