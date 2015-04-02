/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_COLLISION_GRIDMODEL_H
#define SOFA_COMPONENT_COLLISION_GRIDMODEL_H

#include <sofa/core/CollisionModel.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

class GridModel;

class GridCell : public core::TCollisionElementIterator<GridModel>
{
public:
    GridCell(GridModel* model=NULL, int index=0);

    explicit GridCell(const core::CollisionElementIterator& i);

    const Vector3& minVect() const;

    const Vector3& maxVect() const;

    //const std::pair<GridCell,GridCell>& subcells() const;
};

class GridModel : public core::CollisionModel
{
public:
    SOFA_CLASS(GridModel,sofa::core::CollisionModel);

public:
    typedef core::CollisionElementIterator ChildIterator;
    typedef Vec3Types DataTypes;
    typedef GridCell Element;
    friend class GridCell;

    GridModel();

    virtual void resize(int size);

    // -- CollisionModel interface

    virtual void computeBoundingTree(int maxDepth=0);

    virtual std::pair<core::CollisionElementIterator,core::CollisionElementIterator> getInternalChildren(int index) const;

    virtual std::pair<core::CollisionElementIterator,core::CollisionElementIterator> getExternalChildren(int index) const;

    virtual bool isLeaf( int index ) const;

    void draw(const core::visual::VisualParams*,int index);

    void draw(const core::visual::VisualParams* vparams);
};

inline GridCell::GridCell(GridModel* model, int index)
    : core::TCollisionElementIterator<GridModel>(model, index)
{
}

inline GridCell::GridCell(const core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<GridModel>(static_cast<GridModel*>(i.getCollisionModel()), i.getIndex())
{
}

//inline const Vector3& GridCell::minVect() const
//{
//	return model->elems[index].minBBox;
//}

//inline const Vector3& GridCell::maxVect() const
//{
//	return model->elems[index].maxBBox;
//}

//inline const std::pair<GridCell,GridCell>& GridCell::subcells() const
//{
//	return model->elems[index].subcells;
//}

} // namespace collision

} // namespace component

} // namespace sofa

#endif
