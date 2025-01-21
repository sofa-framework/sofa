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
#include <sofa/core/BaseMapping.h>
#include <sofa/simulation/Node.h>
#include <sofa/type/isRigidType.h>

namespace sofa::gui::component::performer
{


template <class DataTypes>
ConstraintAttachBodyPerformer<DataTypes>::ConstraintAttachBodyPerformer(BaseMouseInteractor *i):
  BaseAttachBodyPerformer<DataTypes>(i)
{}


template <class DataTypes>
bool ConstraintAttachBodyPerformer<DataTypes>::startPartial(const BodyPicked& picked)
{
    core::behavior::MechanicalState<DataTypes>* mstateCollision=nullptr;
    int index;
    if (picked.body)
    {
        this->m_mapper = MouseContactMapper::Create(picked.body);
        if (!(this->m_mapper))
        {
            msg_error(this->m_interactor) << "Problem with Mouse Mapper creation.";
            return false;
        }
        const std::string name = "contactMouse";
        mstateCollision = this->m_mapper->createMapping(name.c_str());
        this->m_mapper->resize(1);
        
        const int idx=picked.indexCollisionElement;
        typename DataTypes::CPos pointPicked = picked.point;
        typename DataTypes::Real r=0.0;
        typename DataTypes::Coord dofPicked;
        DataTypes::setCPos(dofPicked, pointPicked);
        index = this->m_mapper->addPointB(dofPicked, idx, r);
        this->m_mapper->update();

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
            msg_error(this->m_interactor) << "incompatible MState during Mouse Interaction.";
            return false;
        }
    }

    m_mstate1 = dynamic_cast<MouseContainer*>(this->m_interactor->getMouseContainer());
    m_mstate2 = mstateCollision;

    using sofa::component::constraint::lagrangian::model::BilateralLagrangianConstraint;


    this->m_interactionObject = sofa::core::objectmodel::New<BilateralLagrangianConstraint<DataTypes> >(m_mstate1, m_mstate2);
    auto* bconstraint = dynamic_cast< BilateralLagrangianConstraint< DataTypes >* >(this->m_interactionObject.get());

    if constexpr (sofa::type::isRigidType<DataTypes>)
    {
        // BilateralLagrangianConstraint::d_keepOrientDiff is protected
        auto* keepOrientDiffBaseData = bconstraint->findData("keepOrientationDifference");
        assert(keepOrientDiffBaseData);
        auto* keepOrientDiffData = dynamic_cast<Data<bool>*>(keepOrientDiffBaseData);
        assert(keepOrientDiffData);

        // setting "keepOrientDiffData" to True
        // would avoid having the beam forced to be oriented with the world frame.
        // But it is unstable in v24.12 so for now, we set to false.
        keepOrientDiffData->setValue(false);
        
        bconstraint->init();
    }

    bconstraint->setName("Constraint-Mouse-Contact");

    static const typename DataTypes::Coord point1 {};
    static const typename DataTypes::Coord point2 {};
    static const typename DataTypes::Deriv normal {};

    bconstraint->addContact(normal, point1, point2, 0.0, 0, index, point2, point1);
    
    bconstraint->bwdInit();
    
    const core::objectmodel::TagSet &tags=mstateCollision->getTags();
    for (auto tag : tags)
        bconstraint->addTag(tag);

    mstateCollision->getContext()->addObject(bconstraint);
    return true;
}

} // namespace sofa::gui::component::performer
