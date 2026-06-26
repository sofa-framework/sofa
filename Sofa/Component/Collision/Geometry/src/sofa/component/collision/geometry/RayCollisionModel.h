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
#include <sofa/component/collision/geometry/config.h>
#include <sofa/core/fwd.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/defaulttype/VecTypes.h>
#include <set>

namespace sofa::component::collision::response::contact
{
    class BaseRayContact;
} // namespace sofa::component::collision::response::contact

namespace sofa::component::collision::geometry
{

class RayCollisionModel;

class SOFA_COMPONENT_COLLISION_GEOMETRY_API Ray : public core::TCollisionElementIterator<RayCollisionModel>
{
public:
    Ray(RayCollisionModel* model, int index);

    explicit Ray(const core::CollisionElementIterator& i);

    const type::Vec3& origin() const;
    const type::Vec3& direction() const;
    SReal l() const;

    void setOrigin(const type::Vec3& newOrigin);
    void setDirection(const type::Vec3& newDirection);
    void setL(SReal newL);
};

class SOFA_COMPONENT_COLLISION_GEOMETRY_API RayCollisionModel : public core::CollisionModel
{
public:
    SOFA_CLASS(RayCollisionModel, core::CollisionModel);

    typedef sofa::defaulttype::Vec3Types InDataTypes;
    typedef sofa::defaulttype::Vec3Types DataTypes;
    typedef Ray Element;
    friend class Ray;
protected:
    RayCollisionModel(SReal defaultLength=1);
public:
    void init() override;

    // -- CollisionModel interface
    void resize(sofa::Size size) override;

    void computeBoundingTree(int maxDepth) override;

    void draw(const core::visual::VisualParams*, sofa::Index index) override;

    core::behavior::MechanicalState<defaulttype::Vec3Types>* getMechanicalState() { return mstate; }
    // ----------------------------
    int addRay(const type::Vec3& origin, const type::Vec3& direction, SReal length);
    Ray getRay(int index) { return Ray(this, index); }

    int getNbRay() const { return size; }
    void setNbRay(int n) { resize(n); }


    void applyTranslation(const double dx,const double dy,const double dz);
    virtual void addContact(response::contact::BaseRayContact* contact) { contacts.insert(contact); }
    virtual void removeContact(response::contact::BaseRayContact* contact) { contacts.erase(contact); }

    virtual const std::set<response::contact::BaseRayContact*> &getContacts() const { return contacts;}

protected:
    sofa::type::vector<SReal> length;
    sofa::type::vector<type::Vec3> direction;

    Data<SReal> d_defaultLength; ///< The default length for all rays in this collision model

    std::set<response::contact::BaseRayContact*> contacts;
    core::behavior::MechanicalState<defaulttype::Vec3Types>* mstate;

};

inline Ray::Ray(RayCollisionModel* model, int index)
    : core::TCollisionElementIterator<RayCollisionModel>(model, index)
{}

inline Ray::Ray(const core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<RayCollisionModel>(static_cast<RayCollisionModel*>(i.getCollisionModel()), i.getIndex())
{
}

inline void Ray::setDirection(const type::Vec3& newDirection)
{
    model->direction[index] = newDirection;
}

inline void Ray::setL(SReal newL)
{
    model->length[index] = newL;
}

} // namespace sofa::component::collision::geometry
