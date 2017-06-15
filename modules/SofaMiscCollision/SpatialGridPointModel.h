/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_COLLISION_SPATIALGRIDPOINTMODEL_H
#define SOFA_COMPONENT_COLLISION_SPATIALGRIDPOINTMODEL_H
#include "config.h"

#include <SofaMeshCollision/PointModel.h>
#include <SofaSphFluid/SpatialGridContainer.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <vector>

namespace sofa
{

namespace component
{

namespace collision
{


class SOFA_MISC_COLLISION_API SpatialGridPointModel : public PointModel
{
public:
    SOFA_CLASS(SpatialGridPointModel, PointModel);

    typedef container::SpatialGridContainer<defaulttype::Vec3Types> GridContainer;
    typedef GridContainer::Grid Grid;

    Data<int> d_leafScale;
protected:
    SpatialGridPointModel();
public:
    virtual void init();

    // -- CollisionModel interface

    virtual void computeBoundingTree(int maxDepth=0);

    Grid* getGrid() { return grid->getGrid(); }
protected:

    GridContainer* grid;
    class OctreeCell
    {
    public:
        Grid::Key k;
        int pfirst;
        int plast;
        OctreeCell( Grid::Key k = Grid::Key(), int pfirst = 0, int plast = -1)
            : k(k), pfirst(pfirst), plast(plast)
        {
        }
    };
    class OctreeSorter
    {
    public:
        int root_shift;
        OctreeSorter(int root_shift=8) : root_shift(root_shift) {}
        bool operator()(const Grid::Key& k1, const Grid::Key &k2);
        bool operator()(const OctreeCell& c1, const OctreeCell &c2)
        {
            return (*this)(c1.k,c2.k);
        }
    };
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
