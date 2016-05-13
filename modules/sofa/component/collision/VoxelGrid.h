/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_COLLISION_VOXELGRID_H
#define SOFA_COMPONENT_COLLISION_VOXELGRID_H

#include <sofa/core/collision/BroadPhaseDetection.h>
#include <sofa/component/collision/NarrowPhaseDetection.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/simulation/common/Node.h>
#include <vector>


namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

class VoxelGrid;

class GridCell
{
private:
    //Vector3 minVect, maxVect; // minx, miny, minz; maxx, maxy, maxz
    sofa::helper::vector<core::CollisionElementIterator> collisElems; // elements wich are added at each iteration
    sofa::helper::vector<core::CollisionElementIterator> collisElemsImmobile[2]; // elements which are added only once

    Vector3 minCell, maxCell;
    int timeStamp;
public:
    // Adding a sphere in a cell of the voxel grid.
    // When adding a sphere, we test if there are collision with the sphere in the cell
    // then we add it in the vector sphere
    void add(VoxelGrid* grid, core::CollisionElementIterator collElem, sofa::helper::vector<core::CollisionElementIterator> &vectCollis, int phase);
    void eraseAll(int timeStampMethod);
    GridCell();

    void draw(const core::visual::VisualParams*,int timeStampMethod);
    void setMinMax(const Vector3 &minimum, const Vector3& maximum);
};

class VoxelGrid : public BroadPhaseDetection, public NarrowPhaseDetection
{
private:
    Vector3 nbSubDiv;
    GridCell ***grid;
    bool bDraw;
    Vector3 minVect, maxVect, step;
    void posToIdx (const Vector3& pos, Vector3 &indices);
    simulation::Node* timeLogger;
    simulation::Visitor::ctime_t timeInter;
    friend class GridCell;
public:
    VoxelGrid (Vector3 minVect = Vector3(-20.0, -20.0, -20.0), Vector3 maxVect = Vector3(-20.0, -20.0, -20.0), Vector3 nbSubdivision = Vector3(5.0, 5.0, 5.0), bool draw=false)
    {
        createVoxelGrid (minVect, maxVect, nbSubdivision);
        timeStamp = 0;
        bDraw = draw;
        timeLogger = NULL;
        timeInter = 0;
    }

    ~VoxelGrid () {}

    // Create a voxel grid define by minx, miny, minz, maxx, maxy, maxz and the number of subdivision on x, y, z
    void createVoxelGrid (const Vector3 &min, const Vector3 &max, const Vector3 &nbSubdivision);

    void draw(const core::visual::VisualParams* vparams);

    void add(core::CollisionModel *cm, int phase);

    void addCollisionModel(core::CollisionModel *cm);
    void addCollisionPair(const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair);
    void add (core::CollisionModel *cm);

    void clearBroadPhase()
    {
        BroadPhaseDetection::clearBroadPhase();
        timeStamp++;
    }
    void clearNarrowPhase()
    {
        NarrowPhaseDetection::clearNarrowPhase();
        timeStamp++;
    }

};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
