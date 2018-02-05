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
#ifndef SOFA_COMPONENT_COLLISION_TRIANGLEOCTREEMODEL_H
#define SOFA_COMPONENT_COLLISION_TRIANGLEOCTREEMODEL_H
#include "config.h"

#include <sofa/core/CollisionModel.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaGeneralMeshCollision/TriangleOctree.h>


namespace sofa
{

namespace component
{

namespace collision
{

class SOFA_GENERAL_MESH_COLLISION_API TriangleOctreeModel : public  TriangleModel, public TriangleOctreeRoot
{
public:
    SOFA_CLASS(TriangleOctreeModel, TriangleModel);
protected:
    TriangleOctreeModel();
public:
#if 0
    /// the triangles associated to a point
    helper::vector<helper::vector<int> > pTri;
#endif

    /// the normals for each point
    helper::vector<defaulttype::Vector3> pNorms;
    //vector < defaulttype::Vector4 > octreeVec;
    void draw(const core::visual::VisualParams* vparams) override;
    virtual void computeBoundingTree(int maxDepth=0) override;
    virtual void computeContinuousBoundingTree(double dt, int maxDepth=0) override;
    /// init the octree creation
    void buildOctree ();
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
