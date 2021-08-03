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

#include <SofaGeneralMeshCollision/config.h>

#include <sofa/core/collision/NarrowPhaseDetection.h>

namespace sofa::component::collision
{

class CubeCollisionModel;

/**
 *  \brief It is a Ray Trace based collision detection algorithm
 *
 *   For each point in one object, we trace a ray following the oposite of the point's normal
 *   up to find a triangle in the other object. Both triangles are tested to evaluate if they are in
 *   colliding state. It must be used with a TriangleOctreeModel,as an octree is used to traverse the object.
 */
class SOFA_SOFAGENERALMESHCOLLISION_API RayTraceNarrowPhase : public core::collision::NarrowPhaseDetection
{
public:
    SOFA_CLASS(RayTraceNarrowPhase, core::collision::NarrowPhaseDetection);

private:
    Data < bool > bDraw;

protected:
    RayTraceNarrowPhase();

public:
    void addCollisionPair (const std::pair < core::CollisionModel *,
            core::CollisionModel * >&cmPair) override;

    void findPairsVolume (CubeCollisionModel * cm1, CubeCollisionModel * cm2);

    void draw (const core::visual::VisualParams* vparams) override;
    void setDraw (bool val)
    {
        bDraw.setValue (val);
    }
};

}