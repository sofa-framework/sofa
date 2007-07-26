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
#ifndef SOFA_COMPONENT_COLLISION_SPHEREMODEL_H
#define SOFA_COMPONENT_COLLISION_SPHEREMODEL_H

#include <sofa/core/CollisionModel.h>
#include <sofa/core/VisualModel.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

template<class DataTypes>
class TSphereModel;

template<class DataTypes>
class TSphere;

typedef TSphereModel<Vec3Types> SphereModel;
typedef TSphere<Vec3Types> Sphere;

template<class DataTypes>
class TSphere : public core::TCollisionElementIterator<TSphereModel<DataTypes> >
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    TSphere(TSphereModel<DataTypes>* model, int index);

    explicit TSphere(core::CollisionElementIterator& i);

    const Coord& center() const;

    const Deriv& v() const;

    Real r() const;
};

template<class TDataTypes>
class TSphereModel : public component::MechanicalObject<TDataTypes>, public core::CollisionModel, public core::VisualModel
{
public:
    typedef TDataTypes InDataTypes;
    typedef component::MechanicalObject<InDataTypes> Inherit;
    typedef typename InDataTypes::Real Real;
    typedef typename InDataTypes::VecReal VecReal;

    typedef TDataTypes DataTypes;
    typedef TSphere<DataTypes> Element;
    friend class TSphere<DataTypes>;

protected:
    VecReal radius;

    DataField<double> defaultRadius;

    class Loader;
public:

    TSphereModel(double radius = 1.0);

    int addSphere(const Vector3& pos, double radius);
    void setSphere(int index, const Vector3& pos, double radius);

    virtual bool load(const char* filename);
    void applyScale (const double s);

    sofa::core::componentmodel::behavior::MechanicalState<InDataTypes>* getMechanicalState() { return this; }

    Real getRadius(int i) const { return this->radius[i]; }

    const VecReal& getR() const { return this->radius; }

    // -- CollisionModel interface

    // remove ambiguity
    int getSize() const { return this->Inherit::getSize(); }

    virtual void resize(int size);

    virtual void computeBoundingTree(int maxDepth=0);

    virtual void computeContinuousBoundingTree(double dt, int maxDepth=0);

    void draw(int index);

    // -- VisualModel interface

    void draw();

    void initTextures() { }

    void update() { }
};

template<class TDataTypes>
inline TSphere<TDataTypes>::TSphere(TSphereModel<TDataTypes>* model, int index)
    : core::TCollisionElementIterator<TSphereModel<TDataTypes> >(model, index)
{}

template<class TDataTypes>
inline TSphere<TDataTypes>::TSphere(core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<TSphereModel<TDataTypes> >(static_cast<TSphereModel<TDataTypes>*>(i.getCollisionModel()), i.getIndex())
{
}

template<class TDataTypes>
inline const typename TDataTypes::Coord& TSphere<TDataTypes>::center() const
{
    return (*this->model->getX())[this->index];
}

template<class TDataTypes>
inline const typename TDataTypes::Deriv& TSphere<TDataTypes>::v() const
{
    return (*this->model->getV())[this->index];
}

template<class TDataTypes>
inline typename TDataTypes::Real TSphere<TDataTypes>::r() const
{
    return this->model->getRadius(this->index);
}

} // namespace collision

} // namespace component

} // namespace sofa

#endif
