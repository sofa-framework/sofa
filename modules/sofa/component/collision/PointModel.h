/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_POINTMODEL_H
#define SOFA_COMPONENT_COLLISION_POINTMODEL_H

#include <sofa/core/CollisionModel.h>
#include <sofa/core/VisualModel.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <vector>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

class PointModel;

class Point : public core::TCollisionElementIterator<PointModel>
{
public:
    Point(PointModel* model, int index);

    explicit Point(core::CollisionElementIterator& i);

    const Vector3& p() const;
    const Vector3& pFree() const;
    const Vector3& v() const;

    void getLineNeighbors(std::vector<const Vector3> &) const;
    void getTriangleNeighbors(std::vector<std::pair<Vector3, Vector3> > &) const;
};

class PointModel : public core::CollisionModel, public core::VisualModel
{
public:
    typedef Vec3Types InDataTypes;
    typedef Vec3Types DataTypes;
    typedef DataTypes::VecCoord VecCoord;
    typedef DataTypes::VecDeriv VecDeriv;
    typedef DataTypes::Coord Coord;
    typedef DataTypes::Deriv Deriv;
    typedef Point Element;
    friend class Point;

    PointModel();

    std::vector< std::vector<int> > lineNeighbors;
    std::vector< std::vector< std::pair<int, int> > > triangleNeighbors;

    virtual void init();

    // -- CollisionModel interface

    virtual void resize(int size);

    virtual void computeBoundingTree(int maxDepth=0);

    virtual void computeContinuousBoundingTree(double dt, int maxDepth=0);

    void draw(int index);

    // -- VisualModel interface

    void draw();

    void initTextures() { }

    void update() { }


    core::componentmodel::behavior::MechanicalState<Vec3Types>* getMechanicalState() { return mstate; }

    //virtual const char* getTypeName() const { return "Point"; }

protected:

    core::componentmodel::behavior::MechanicalState<Vec3Types>* mstate;
};

inline Point::Point(PointModel* model, int index)
    : core::TCollisionElementIterator<PointModel>(model, index)
{}

inline Point::Point(core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<PointModel>(static_cast<PointModel*>(i.getCollisionModel()), i.getIndex())
{
}

inline const Vector3& Point::p() const { return (*model->mstate->getX())[index]; }

inline const Vector3& Point::pFree() const { return (*model->mstate->getXfree())[index]; }

inline const Vector3& Point::v() const { return (*model->mstate->getV())[index]; }

} // namespace collision

} // namespace component

} // namespace sofa

#endif
