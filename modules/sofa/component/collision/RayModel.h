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
#ifndef SOFA_COMPONENT_COLLISION_RAYMODEL_H
#define SOFA_COMPONENT_COLLISION_RAYMODEL_H

#include <sofa/core/CollisionModel.h>
#include <sofa/core/VisualModel.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <set>


namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

class RayModel;

class Ray : public core::TCollisionElementIterator<RayModel>
{
public:
    Ray(RayModel* model, int index);

    explicit Ray(core::CollisionElementIterator& i);

    const Vector3& origin() const;
    const Vector3& direction() const;
    double l() const;

    Vector3& origin();
    Vector3& direction();
    double& l();
};

class BaseRayContact;

class RayModel : public component::MechanicalObject<Vec3Types>, public core::CollisionModel, public core::VisualModel
{
protected:
    sofa::helper::vector<double> length;

    DataField<double> defaultLength;

    std::set<BaseRayContact*> contacts;
public:
    typedef Vec3Types InDataTypes;
    typedef Vec3Types DataTypes;
    typedef Ray Element;
    friend class Ray;

    RayModel(double defaultLength=1);

    int addRay(const Vector3& origin, const Vector3& direction, double length);

    int getNbRay() const { return size; }

    void setNbRay(int n) { resize(2*n); }

    Ray getRay(int index) { return Ray(this, index); }

    virtual void addContact(BaseRayContact* contact) { contacts.insert(contact); }

    virtual void removeContact(BaseRayContact* contact) { contacts.erase(contact); }

    virtual void resize(int size);

    // -- CollisionModel interface

    virtual void computeBoundingTree(int maxDepth);

    void draw(int index);

    void applyTranslation(const double dx,const double dy,const double dz);

    // -- VisualModel interface

    void draw();

    void initTextures() { }

    void update() { }
};

inline Ray::Ray(RayModel* model, int index)
    : core::TCollisionElementIterator<RayModel>(model, index)
{}

inline Ray::Ray(core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<RayModel>(static_cast<RayModel*>(i.getCollisionModel()), i.getIndex())
{
}

inline const Vector3& Ray::origin() const
{
    return (*model->getX())[2*index+0];
}

inline const Vector3& Ray::direction() const
{
    return (*model->getX())[2*index+1];
}

inline double Ray::l() const
{
    return model->length[index];
}

inline Vector3& Ray::origin()
{
    return (*model->getX())[2*index+0];
}

inline Vector3& Ray::direction()
{
    return (*model->getX())[2*index+1];
}

inline double& Ray::l()
{
    return model->length[index];
}

} // namespace collision

} // namespace component

} // namespace sofa

#endif
