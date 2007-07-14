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

class SphereModel;

class Sphere : public core::TCollisionElementIterator<SphereModel>
{
public:
    Sphere(SphereModel* model, int index);

    explicit Sphere(core::CollisionElementIterator& i);

    const Vector3& center() const;

    const Vector3& v() const;

    double r() const;
};

class SphereModel : public component::MechanicalObject<Vec3Types>, public core::CollisionModel, public core::VisualModel
{
protected:
    std::vector<double> radius;

    DataField<double> defaultRadius;

    class Loader;
public:
    typedef Vec3Types DataTypes;
    typedef Sphere Element;
    typedef component::MechanicalObject<Vec3Types> Inherit;

    friend class Sphere;

    SphereModel(double radius = 1.0);

    int addSphere(const Vector3& pos, double radius);
    void setSphere(int index, const Vector3& pos, double radius);

    virtual bool load(const char* filename);
    void applyScale (const double s);

    // -- CollisionModel interface

    // remove ambiguity
    int getSize() const { return Inherit::getSize(); }

    virtual void resize(int size);

    virtual void computeBoundingTree(int maxDepth=0);

    virtual void computeContinuousBoundingTree(double dt, int maxDepth=0);

    void draw(int index);

    // -- VisualModel interface

    void draw();

    void initTextures() { }

    void update() { }
};

inline Sphere::Sphere(SphereModel* model, int index)
    : core::TCollisionElementIterator<SphereModel>(model, index)
{}

inline Sphere::Sphere(core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<SphereModel>(static_cast<SphereModel*>(i.getCollisionModel()), i.getIndex())
{
}

inline const Vector3& Sphere::center() const
{
    return (*model->getX())[index];
}

inline const Vector3& Sphere::v() const
{
    return (*model->getV())[index];
}

inline double Sphere::r() const
{
    return model->radius[index];
}

} // namespace collision

} // namespace component

} // namespace sofa

#endif
