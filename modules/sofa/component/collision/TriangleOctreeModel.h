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
#ifndef SOFA_COMPONENT_COLLISION_TRIANGLEOCTREEMODEL_H
#define SOFA_COMPONENT_COLLISION_TRIANGLEOCTREEMODEL_H

#include <sofa/core/CollisionModel.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/TriangleOctree.h>


namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

class SOFA_BASE_COLLISION_API TriangleOctreeModel : public  TriangleModel, public TriangleOctreeRoot
{
public:
    SOFA_CLASS(TriangleOctreeModel, TriangleModel);
protected:
    TriangleOctreeModel();
public:
#if 0
    /// the triangles associated to a point
    vector<vector<int> > pTri;
#endif

    /// the normals for each point
    vector<Vector3> pNorms;
    //vector < Vector4 > octreeVec;
    void draw(const core::visual::VisualParams* vparams);
    virtual void computeBoundingTree(int maxDepth=0);
    virtual void computeContinuousBoundingTree(double dt, int maxDepth=0);
    /// init the octree creation
    void buildOctree ();
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
