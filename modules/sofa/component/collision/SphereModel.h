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
#ifndef SOFA_COMPONENT_COLLISION_SPHEREMODEL_H
#define SOFA_COMPONENT_COLLISION_SPHEREMODEL_H

#include <sofa/core/CollisionModel.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/component/component.h>
#include <sofa/defaulttype/VecTypes.h>
#include <vector>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

class SphereModel;

class Sphere : public core::TCollisionElementIterator<SphereModel>
{
public:
    typedef SReal   Real;
    typedef Vector3 Coord;

    Sphere(SphereModel* model, int index);

    explicit Sphere(core::CollisionElementIterator& i);

    const Coord& center() const;
    const Coord& p() const;
    const Coord& pFree() const;
    const Coord& v() const;
    Real r() const;
};

class SOFA_COMPONENT_COLLISION_API SphereModel : public core::CollisionModel
{
public:
    SOFA_CLASS(SphereModel, core::CollisionModel);

    typedef Vec3Types InDataTypes;
    typedef Vec3Types DataTypes;
    typedef DataTypes::VecCoord VecCoord;
    typedef DataTypes::VecDeriv VecDeriv;
    typedef DataTypes::Coord Coord;
    typedef DataTypes::Deriv Deriv;
    typedef DataTypes::Real Real;
    typedef DataTypes::VecReal VecReal;
    typedef Sphere Element;
    friend class Sphere;

    SphereModel();

    virtual void init();

    // -- CollisionModel interface

    virtual void resize(int size);

    virtual void computeBoundingTree(int maxDepth=0);

    virtual void computeContinuousBoundingTree(double dt, int maxDepth=0);

    void draw(int index);

    void draw();

    virtual void drawColourPicking();

    core::behavior::MechanicalState<Vec3Types>* getMechanicalState() { return mstate; }

    virtual bool load(const char* filename);

    int addSphere(const Vector3& pos, Real r);
    void setSphere(int i, const Vector3& pos, Real r);

    Real getRadius(const int i) const;
    void setRadius(const int i, const Real r);
    void setRadius(const Real r);

protected:

    core::behavior::MechanicalState<Vec3Types>* mstate;

    Data< VecReal > radius;
    Data< SReal > defaultRadius;

    sofa::core::objectmodel::DataFileName filename;

    class Loader;
};

inline Sphere::Sphere(SphereModel* model, int index)
    : core::TCollisionElementIterator<SphereModel>(model, index)
{}

inline Sphere::Sphere(core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<SphereModel>(static_cast<SphereModel*>(i.getCollisionModel()), i.getIndex())
{
}

inline const Sphere::Coord& Sphere::center() const { return (*model->mstate->getX())[index]; }

inline const Sphere::Coord& Sphere::p() const { return (*model->mstate->getX())[index]; }

inline const Sphere::Coord& Sphere::pFree() const { return (*model->mstate->getXfree())[index]; }

inline const Sphere::Coord& Sphere::v() const { return (*model->mstate->getV())[index]; }

inline Sphere::Real Sphere::r() const { return (Real) model->getRadius((unsigned)index); }

} // namespace collision

} // namespace component

} // namespace sofa

#endif
