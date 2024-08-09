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

#include <sofa/gui/component/performer/AttachBodyPerformer.h>
#include <sofa/gui/component/performer/MouseInteractor.h>
#include <sofa/component/solidmechanics/spring/SpringForceField.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/simulation/Node.h>

namespace sofa::gui::component::performer
{


template <class DataTypes>
AttachBodyPerformer<DataTypes>::AttachBodyPerformer(BaseMouseInteractor *i):
  BaseAttachBodyPerformer<DataTypes>(i)
{}


template <class DataTypes>
bool AttachBodyPerformer<DataTypes>::startPartial(const BodyPicked& picked)
{

    core::behavior::MechanicalState<DataTypes>* mstateCollision=nullptr;
    int index;
    if (picked.body)
    {
        this->m_mapper = MouseContactMapper::Create(picked.body);
        if (!this->m_mapper)
        {
            msg_warning(this->m_interactor) << "Problem with Mouse Mapper creation " ;
            return false;
        }
        const std::string name = "contactMouse";
        mstateCollision = this->m_mapper->createMapping(name.c_str());
        this->m_mapper->resize(1);

        const unsigned int idx=picked.indexCollisionElement;
        typename DataTypes::CPos pointPicked=(typename DataTypes::CPos)picked.point;
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
            for (core::objectmodel::TagSet::const_iterator it=tags.begin(); it!=tags.end(); ++it)
            {
                mstateCollision->addTag(*it);
                mappedNode->mechanicalMapping->addTag(*it);
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
            msg_warning(this->m_interactor) << "incompatible MState during Mouse Interaction " ;
            return false;
        }
    }

    using sofa::component::solidmechanics::spring::SpringForceField;

    this->m_interactionObject = sofa::core::objectmodel::New< SpringForceField<DataTypes> >(dynamic_cast<MouseContainer*>(this->m_interactor->getMouseContainer()), mstateCollision);
    auto* springforcefield = dynamic_cast< SpringForceField< DataTypes >* >(this->m_interactionObject.get());
    springforcefield->setName("Spring-Mouse-Contact");
    springforcefield->setArrowSize((float)this->m_size);
    springforcefield->setDrawMode(2); //Arrow mode if size > 0
    springforcefield->addSpring(0,index, m_stiffness, 0.0, picked.dist);

    const core::objectmodel::TagSet &tags=mstateCollision->getTags();
    for (core::objectmodel::TagSet::const_iterator it=tags.begin(); it!=tags.end(); ++it)
        springforcefield->addTag(*it);

    mstateCollision->getContext()->addObject(springforcefield);
    return true;
}

} // namespace sofa::gui::component::performer
