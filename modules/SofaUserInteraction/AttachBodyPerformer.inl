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
#ifndef SOFA_COMPONENT_COLLISION_ATTACHBODYPERFORMER_INL
#define SOFA_COMPONENT_COLLISION_ATTACHBODYPERFORMER_INL

#include <SofaUserInteraction/AttachBodyPerformer.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaUserInteraction/MouseInteractor.h>



namespace sofa
{

namespace component
{

namespace collision
{

template <class DataTypes>
void AttachBodyPerformer<DataTypes>::start()
{
    if (m_forcefield)
    {
        clear();
        return;
    }
    BodyPicked picked=this->interactor->getBodyPicked();
    if (!picked.body && !picked.mstate) return;

    if (!start_partial(picked)) return; //template specialized code is here

    double distanceFromMouse=picked.rayLength;
    this->interactor->setDistanceFromMouse(distanceFromMouse);
    Ray ray = this->interactor->getMouseRayModel()->getRay(0);
    ray.setOrigin(ray.origin() + ray.direction()*distanceFromMouse);
    sofa::core::BaseMapping *mapping;
    this->interactor->getContext()->get(mapping); assert(mapping);
    mapping->apply(core::MechanicalParams::defaultInstance());
    mapping->applyJ(core::MechanicalParams::defaultInstance());
    m_forcefield->init();
    this->interactor->setMouseAttached(true);
}



template <class DataTypes>
void AttachBodyPerformer<DataTypes>::execute()
{
    sofa::core::BaseMapping *mapping;
    this->interactor->getContext()->get(mapping); assert(mapping);
    mapping->apply(core::MechanicalParams::defaultInstance());
    mapping->applyJ(core::MechanicalParams::defaultInstance());
    this->interactor->setMouseAttached(true);
}

template <class DataTypes>
void AttachBodyPerformer<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (m_forcefield)
    {
        core::visual::VisualParams* vp = const_cast<core::visual::VisualParams*>(vparams);
        core::visual::DisplayFlags backup = vp->displayFlags();
        vp->displayFlags() = flags;
        m_forcefield->draw(vp);
        vp->displayFlags() = backup;
    }
}

template <class DataTypes>
AttachBodyPerformer<DataTypes>::AttachBodyPerformer(BaseMouseInteractor *i):
    TInteractionPerformer<DataTypes>(i),
    mapper(NULL)
{
    flags.setShowVisualModels(false);
    flags.setShowInteractionForceFields(true);
}

template <class DataTypes>
void AttachBodyPerformer<DataTypes>::clear()
{
    if (m_forcefield)
    {
        m_forcefield->cleanup();
        m_forcefield->getContext()->removeObject(m_forcefield);
        m_forcefield.reset();
    }

    if (mapper)
    {
        mapper->cleanup();
        delete mapper; mapper=NULL;
    }

    this->interactor->setDistanceFromMouse(0);
    this->interactor->setMouseAttached(false);
}


template <class DataTypes>
AttachBodyPerformer<DataTypes>::~AttachBodyPerformer()
{
    clear();
}

template <class DataTypes>
bool AttachBodyPerformer<DataTypes>::start_partial(const BodyPicked& picked)
{

    core::behavior::MechanicalState<DataTypes>* mstateCollision=NULL;
    int index;
    if (picked.body)
    {
        mapper = MouseContactMapper::Create(picked.body);
        if (!mapper)
        {
            this->interactor->serr << "Problem with Mouse Mapper creation : " << this->interactor->sendl;
            return false;
        }
        std::string name = "contactMouse";
        mstateCollision = mapper->createMapping(name.c_str());
        mapper->resize(1);

        const typename DataTypes::Coord pointPicked=(typename DataTypes::Coord)picked.point;
        const int idx=picked.indexCollisionElement;
        typename DataTypes::Real r=0.0;

        index = mapper->addPointB(pointPicked, idx, r
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                , picked.baryCoords
#endif
                                 );
        mapper->update();

        if (mstateCollision->getContext() != picked.body->getContext())
        {

            simulation::Node *mappedNode=(simulation::Node *) mstateCollision->getContext();
            simulation::Node *mainNode=(simulation::Node *) picked.body->getContext();
            core::behavior::BaseMechanicalState *mainDof=mainNode->getMechanicalState();
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
            this->interactor->serr << "incompatible MState during Mouse Interaction " << this->interactor->sendl;
            return false;
        }
    }

    using sofa::component::interactionforcefield::StiffSpringForceField;

    m_forcefield = sofa::core::objectmodel::New< StiffSpringForceField<DataTypes> >(dynamic_cast<MouseContainer*>(this->interactor->getMouseContainer()), mstateCollision);
    StiffSpringForceField< DataTypes >* stiffspringforcefield = static_cast< StiffSpringForceField< DataTypes >* >(m_forcefield.get());
    stiffspringforcefield->setName("Spring-Mouse-Contact");
    stiffspringforcefield->setArrowSize((float)this->size);
    stiffspringforcefield->setDrawMode(2); //Arrow mode if size > 0


    stiffspringforcefield->addSpring(0,index, stiffness, 0.0, picked.dist);
    const core::objectmodel::TagSet &tags=mstateCollision->getTags();
    for (core::objectmodel::TagSet::const_iterator it=tags.begin(); it!=tags.end(); ++it)
        stiffspringforcefield->addTag(*it);

    mstateCollision->getContext()->addObject(stiffspringforcefield);
    return true;
}

}
}
}

#endif
