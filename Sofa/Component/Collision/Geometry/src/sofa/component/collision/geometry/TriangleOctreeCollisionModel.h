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

#include <sofa/core/CollisionModel.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/TriangleOctree.h>
#include <sofa/component/collision/geometry/TriangleCollisionModel.h>


namespace sofa::component::collision::geometry
{

class SOFA_COMPONENT_COLLISION_GEOMETRY_API TriangleOctreeCollisionModel : public  TriangleCollisionModel<sofa::defaulttype::Vec3Types>, public helper::TriangleOctreeRoot
{
public:
    SOFA_CLASS(TriangleOctreeCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>);
protected:
    TriangleOctreeCollisionModel();
    void drawCollisionModel(const core::visual::VisualParams* vparams) override;
public:
    Data<bool> d_drawOctree;  ///< draw the octree

    /// the normals for each point
    type::vector<type::Vec3> pNorms;
    void computeBoundingTree(int maxDepth=0) override;
    void computeContinuousBoundingTree(SReal dt, int maxDepth=0) override;
    /// init the octree creation
    void buildOctree ();
};


using TriangleOctreeModel SOFA_ATTRIBUTE_DEPRECATED__RENAMED_TRIANGLEOCTREEMODEL() = TriangleOctreeCollisionModel;


} // namespace sofa::component::collision::geometry
