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
#ifndef SOFA_COMPONENT_COLLISION_POINTMODEL_H
#define SOFA_COMPONENT_COLLISION_POINTMODEL_H

#include <sofa/core/CollisionModel.h>
#include <sofa/component/collision/LocalMinDistanceFilter.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
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
class PointLocalMinDistanceFilter;

class Point : public core::TCollisionElementIterator<PointModel>
{
public:
    Point(PointModel* model, int index);

    explicit Point(core::CollisionElementIterator& i);

    const Vector3& p() const;
    const Vector3& pFree() const;
    const Vector3& v() const;
    Vector3 n() const;

    /// Return true if the element stores a free position vector
    bool hasFreePosition() const;

    bool testLMD(const Vector3 &, double &, double &);

    bool activated(core::CollisionModel *cm = 0) const;
};

class PointActiver
{
public:
    PointActiver() {}
    virtual ~PointActiver() {}
    virtual bool activePoint(int /*index*/, core::CollisionModel * /*cm*/ = 0) {return true;}
};

class SOFA_MESH_COLLISION_API PointModel : public core::CollisionModel
{
public:
    SOFA_CLASS(PointModel, core::CollisionModel);

    typedef Vec3Types InDataTypes;
    typedef Vec3Types DataTypes;
    typedef DataTypes::VecCoord VecCoord;
    typedef DataTypes::VecDeriv VecDeriv;
    typedef DataTypes::Coord Coord;
    typedef DataTypes::Deriv Deriv;
    typedef Point Element;
    typedef helper::vector<unsigned int> VecIndex;

    friend class Point;
protected:
    PointModel();
public:
    virtual void init();

    // -- CollisionModel interface

    virtual void resize(int size);

    virtual void computeBoundingTree(int maxDepth=0);

    virtual void computeContinuousBoundingTree(double dt, int maxDepth=0);

    void draw(const core::visual::VisualParams*,int index);

    void draw(const core::visual::VisualParams* vparams);

    virtual bool canCollideWithElement(int index, CollisionModel* model2, int index2);

    core::behavior::MechanicalState<Vec3Types>* getMechanicalState() { return mstate; }

    //virtual const char* getTypeName() const { return "Point"; }

    PointLocalMinDistanceFilter *getFilter() const;

    //template< class TFilter >
    //TFilter *getFilter() const
    //{
    //	if (m_lmdFilter != 0)
    //		return m_lmdFilter;
    //	else
    //		return &m_emptyFilter;
    //}

    void setFilter(PointLocalMinDistanceFilter * /*lmdFilter*/);

    Data<bool> bothSide; // to activate collision on both side of the point model (when surface normals are defined on these points)

protected:

    core::behavior::MechanicalState<Vec3Types>* mstate;

    Data<bool> computeNormals;

    Data<std::string> PointActiverPath;

    VecDeriv normals;

    PointLocalMinDistanceFilter *m_lmdFilter;
    EmptyFilter m_emptyFilter;

    Data<bool> m_displayFreePosition;

    void updateNormals();

    PointActiver *myActiver;
};

inline Point::Point(PointModel* model, int index)
    : core::TCollisionElementIterator<PointModel>(model, index)
{

}

inline Point::Point(core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<PointModel>(static_cast<PointModel*>(i.getCollisionModel()), i.getIndex())
{

}

inline const Vector3& Point::p() const { return (*model->mstate->getX())[index]; }

inline const Vector3& Point::pFree() const
{
    if (hasFreePosition())
        return model->mstate->read(core::ConstVecCoordId::freePosition())->getValue()[index];
    else
        return p();
}

inline const Vector3& Point::v() const { return (*model->mstate->getV())[index]; }

inline Vector3 Point::n() const { return ((unsigned)index<model->normals.size()) ? model->normals[index] : Vector3(); }

inline bool Point::hasFreePosition() const { return model->mstate->read(core::ConstVecCoordId::freePosition())->isSet(); }

inline bool Point::activated(core::CollisionModel *cm) const
{
    return model->myActiver->activePoint(index, cm);
}

//bool Point::testLMD(const Vector3 &PQ, double &coneFactor, double &coneExtension);

} // namespace collision

} // namespace component

} // namespace sofa

#endif
