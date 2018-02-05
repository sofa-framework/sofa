/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_COLLISION_RAYMODEL_H
#define SOFA_COMPONENT_COLLISION_RAYMODEL_H
#include "config.h"

#include <sofa/core/CollisionModel.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <set>


namespace sofa
{

namespace component
{

namespace collision
{

class RayModel;

class Ray : public core::TCollisionElementIterator<RayModel>
{
public:
    Ray(RayModel* model, int index);

    explicit Ray(const core::CollisionElementIterator& i);

    const defaulttype::Vector3& origin() const;
    const defaulttype::Vector3& direction() const;
    SReal l() const;

    void setOrigin(const defaulttype::Vector3& newOrigin);
    void setDirection(const defaulttype::Vector3& newDirection);
    void setL(SReal newL);
};

class BaseRayContact;

class SOFA_USER_INTERACTION_API RayModel : public core::CollisionModel
{
public:
    SOFA_CLASS(RayModel, core::CollisionModel);

    typedef sofa::defaulttype::Vec3Types InDataTypes;
    typedef sofa::defaulttype::Vec3Types DataTypes;
    typedef Ray Element;
    friend class Ray;
protected:
    RayModel(SReal defaultLength=1);
public:
    void init() override;

    // -- CollisionModel interface
    virtual void resize(int size) override;

    virtual void computeBoundingTree(int maxDepth) override;

    void draw(const core::visual::VisualParams*,int index) override;
    void draw(const core::visual::VisualParams* vparams) override;

    core::behavior::MechanicalState<defaulttype::Vec3Types>* getMechanicalState() { return mstate; }
    // ----------------------------
    int addRay(const defaulttype::Vector3& origin, const defaulttype::Vector3& direction, SReal length);
    Ray getRay(int index) { return Ray(this, index); }

    int getNbRay() const { return size; }
    void setNbRay(int n) { resize(n); }


    void applyTranslation(const double dx,const double dy,const double dz);
    virtual void addContact(BaseRayContact* contact) { contacts.insert(contact); }
    virtual void removeContact(BaseRayContact* contact) { contacts.erase(contact); }

    virtual const std::set<BaseRayContact*> &getContacts() const { return contacts;}

protected:
    sofa::helper::vector<SReal> length;
    sofa::helper::vector<defaulttype::Vector3> direction;

    Data<SReal> defaultLength;

    std::set<BaseRayContact*> contacts;
    core::behavior::MechanicalState<defaulttype::Vec3Types>* mstate;

};

inline Ray::Ray(RayModel* model, int index)
    : core::TCollisionElementIterator<RayModel>(model, index)
{}

inline Ray::Ray(const core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<RayModel>(static_cast<RayModel*>(i.getCollisionModel()), i.getIndex())
{
}

inline const defaulttype::Vector3& Ray::origin() const
{
    return model->getMechanicalState()->read(core::ConstVecCoordId::position())->getValue()[index];
}

inline const defaulttype::Vector3& Ray::direction() const
{
    return model->direction[index];
}

inline defaulttype::Vector3::value_type Ray::l() const
{
    return model->length[index];
}

inline void Ray::setOrigin(const defaulttype::Vector3& newOrigin)
{
    helper::WriteAccessor<Data<helper::vector<defaulttype::Vector3> > > xData =
        *model->getMechanicalState()->write(core::VecCoordId::position());
    xData.wref()[index] = newOrigin;

    helper::WriteAccessor<Data<helper::vector<defaulttype::Vector3> > > xDataFree =
        *model->getMechanicalState()->write(core::VecCoordId::freePosition());
    defaulttype::Vec3Types::VecCoord& freePos = xDataFree.wref();
    freePos.resize(model->getMechanicalState()->getSize());
    freePos[index] = newOrigin;
}

inline void Ray::setDirection(const defaulttype::Vector3& newDirection)
{
    model->direction[index] = newDirection;
}

inline void Ray::setL(SReal newL)
{
    model->length[index] = newL;
}

} // namespace collision

} // namespace component

} // namespace sofa

#endif
