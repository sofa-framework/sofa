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

#include <sofa/gui/component/performer/ConstraintAttachBodyPerformer.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/gui/component/performer/MouseInteractor.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/simulation/Node.h>

namespace sofa::gui::component::performer
{

template <class DataTypes>
void ConstraintAttachBodyPerformer<DataTypes>::start()
{
    if (m_constraint)
    {
        clear();
        return;
    }
    const BodyPicked picked=this->interactor->getBodyPicked();
    if (!picked.body && !picked.mstate) return;

    if (!start_partial(picked)) return; //template specialized code is here

    double distanceFromMouse=picked.rayLength;
    this->interactor->setDistanceFromMouse(distanceFromMouse);
    sofa::component::collision::geometry::Ray ray = this->interactor->getMouseRayModel()->getRay(0);
    ray.setOrigin(ray.origin() + ray.direction()*distanceFromMouse);
    sofa::core::BaseMapping *mapping;
    this->interactor->getContext()->get(mapping); assert(mapping);
    mapping->apply(core::mechanicalparams::defaultInstance());
    mapping->applyJ(core::mechanicalparams::defaultInstance());
    m_constraint->init();
    this->interactor->setMouseAttached(true);
}



template <class DataTypes>
void ConstraintAttachBodyPerformer<DataTypes>::execute()
{
    sofa::core::BaseMapping *mapping;
    this->interactor->getContext()->get(mapping); assert(mapping);
    mapping->apply(core::mechanicalparams::defaultInstance());
    mapping->applyJ(core::mechanicalparams::defaultInstance());
    this->interactor->setMouseAttached(true);
}

template <class DataTypes>
void ConstraintAttachBodyPerformer<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (m_constraint)
    {
        core::visual::VisualParams* vp = const_cast<core::visual::VisualParams*>(vparams);
        const core::visual::DisplayFlags backup = vp->displayFlags();
        vp->displayFlags() = flags;
        m_constraint->draw(vp);
        vp->displayFlags() = backup;
    }
}

template <class DataTypes>
ConstraintAttachBodyPerformer<DataTypes>::ConstraintAttachBodyPerformer(BaseMouseInteractor *i):
    TInteractionPerformer<DataTypes>(i),
    mapper(nullptr)
{
    flags.setShowVisualModels(false);
    flags.setShowInteractionForceFields(true);
}

template <class DataTypes>
void ConstraintAttachBodyPerformer<DataTypes>::clear()
{
    if (m_constraint)
    {
        m_constraint->cleanup();
        m_constraint->getContext()->removeObject(m_constraint);
        m_constraint.reset();
    }

    if (mapper)
    {
        mapper->cleanup();
        delete mapper; mapper=nullptr;
    }

    this->interactor->setDistanceFromMouse(0);
    this->interactor->setMouseAttached(false);
}


template <class DataTypes>
ConstraintAttachBodyPerformer<DataTypes>::~ConstraintAttachBodyPerformer()
{
    clear();
}

template <class DataTypes>
bool ConstraintAttachBodyPerformer<DataTypes>::start_partial(const BodyPicked& picked)
{
    core::behavior::MechanicalState<DataTypes>* mstateCollision=nullptr;
    int index;
    if (picked.body)
    {
        mapper = MouseContactMapper::Create(picked.body);
        if (!mapper)
        {
            msg_error(this->interactor) << "Problem with Mouse Mapper creation.";
            return false;
        }
        const std::string name = "contactMouse";
        mstateCollision = mapper->createMapping(name.c_str());
        mapper->resize(1);

        const typename DataTypes::Coord pointPicked=picked.point;
        const int idx=picked.indexCollisionElement;
        typename DataTypes::Real r=0.0;

        index = mapper->addPointB(pointPicked, idx, r);
        mapper->update();

        if (mstateCollision->getContext() != picked.body->getContext())
        {
            const simulation::Node *mappedNode=(simulation::Node *) mstateCollision->getContext();
            const simulation::Node *mainNode=(simulation::Node *) picked.body->getContext();
            const core::behavior::BaseMechanicalState *mainDof=mainNode->getMechanicalState();
            const core::objectmodel::TagSet &tags=mainDof->getTags();
            for (auto tag : tags)
            {
                mstateCollision->addTag(tag);
                mappedNode->mechanicalMapping->addTag(tag);
            }
            mstateCollision->setName("AttachedPoint");
            mappedNode->mechanicalMapping->setName("MouseMapping");
        }
    }
    else
    {
        mstateCollision = dynamic_cast< core::behavior::MechanicalState<DataTypes>*  >(picked.mstate);
        index = picked.indexCollisionElement;
        if (!mstateCollision)
        {
            msg_error(this->interactor) << "incompatible MState during Mouse Interaction.";
            return false;
        }
    }

    mstate1 = dynamic_cast<MouseContainer*>(this->interactor->getMouseContainer());
    mstate2 = mstateCollision;

    type::Vec3d point1;
    type::Vec3d point2;

    using sofa::component::constraint::lagrangian::model::BilateralLagrangianConstraint;

    m_constraint = sofa::core::objectmodel::New<BilateralLagrangianConstraint<sofa::defaulttype::Vec3Types> >(mstate1, mstate2);
    BilateralLagrangianConstraint< DataTypes >* bconstraint = static_cast< BilateralLagrangianConstraint< sofa::defaulttype::Vec3Types >* >(m_constraint.get());
    bconstraint->setName("Constraint-Mouse-Contact");

    type::Vec3d normal = point1-point2;

    bconstraint->addContact(normal, point1, point2, normal.norm(), 0, index, point2, point1);

    const core::objectmodel::TagSet &tags=mstateCollision->getTags();
    for (auto tag : tags)
        bconstraint->addTag(tag);

    mstateCollision->getContext()->addObject(bconstraint);
    return true;
}

} // namespace sofa::gui::component::performer
