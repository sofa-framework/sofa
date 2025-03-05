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

#include <sofa/gui/component/performer/BaseAttachBodyPerformer.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/gui/component/performer/MouseInteractor.h>


namespace sofa::gui::component::performer
{

template <class DataTypes>
BaseAttachBodyPerformer<DataTypes>::BaseAttachBodyPerformer(BaseMouseInteractor* i)
    : TInteractionPerformer<DataTypes>(i)
    , m_mapper(nullptr)
{
    this->m_flags.setShowVisualModels(false);
    this->m_flags.setShowInteractionForceFields(true);
};

template <class DataTypes>
BaseAttachBodyPerformer<DataTypes>::~BaseAttachBodyPerformer()
{
    clear();
};

template <class DataTypes>
void BaseAttachBodyPerformer<DataTypes>::start()
{
    if (m_interactionObject)
    {
        clear();
        return;
    }
    const BodyPicked picked=this->m_interactor->getBodyPicked();
    if (!picked.body && !picked.mstate)
        return;

    if (!startPartial(picked)) //template specialized code is here
        return;

    double distanceFromMouse=picked.rayLength;
    this->m_interactor->setDistanceFromMouse(distanceFromMouse);
    
    sofa::component::collision::geometry::Ray ray = this->m_interactor->getMouseRayModel()->getRay(0);
    ray.setOrigin(ray.origin() + ray.direction()*distanceFromMouse);
    
    sofa::core::BaseMapping *mapping;
    this->m_interactor->getContext()->get(mapping); assert(mapping);
    mapping->apply(core::mechanicalparams::defaultInstance());
    mapping->applyJ(core::mechanicalparams::defaultInstance());
    
    m_interactionObject->init();
    this->m_interactor->setMouseAttached(true);
}

template <class DataTypes>
void BaseAttachBodyPerformer<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (m_interactionObject)
    {
        core::visual::VisualParams* vp = const_cast<core::visual::VisualParams*>(vparams);
        const core::visual::DisplayFlags backup = vp->displayFlags();
        vp->displayFlags() = m_flags;
        m_interactionObject->draw(vp);
        vp->displayFlags() = backup;
    }
}

template <class DataTypes>
void BaseAttachBodyPerformer<DataTypes>::clear()
{
    if (m_interactionObject)
    {
        m_interactionObject->cleanup();
        m_interactionObject->getContext()->removeObject(m_interactionObject);
        m_interactionObject.reset();
    }

    if (m_mapper)
    {
        m_mapper->cleanup();
        delete m_mapper;
        m_mapper = nullptr;
    }

    this->m_interactor->setDistanceFromMouse(0);
    this->m_interactor->setMouseAttached(false);
}

template <class DataTypes>
void BaseAttachBodyPerformer<DataTypes>::execute()
{
    sofa::core::BaseMapping *mapping;
    this->m_interactor->getContext()->get(mapping); assert(mapping);
    mapping->apply(core::mechanicalparams::defaultInstance());
    mapping->applyJ(core::mechanicalparams::defaultInstance());
    this->m_interactor->setMouseAttached(true);
}

template <class DataTypes>
sofa::core::objectmodel::BaseObject::SPtr BaseAttachBodyPerformer<DataTypes>::getInteractionObject()
{
        return m_interactionObject;
};



}
