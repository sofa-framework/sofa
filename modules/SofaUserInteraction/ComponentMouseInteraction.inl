/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_COLLISION_COMPONENTMOUSEINTERACTION_INL
#define SOFA_COMPONENT_COLLISION_COMPONENTMOUSEINTERACTION_INL

#include <SofaUserInteraction/ComponentMouseInteraction.h>
#include <sofa/core/visual/VisualParams.h>

#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaUserInteraction/RayModel.h>
#include <SofaUserInteraction/MouseInteractor.inl>
#include <SofaBaseMechanics/IdentityMapping.inl>

#include <sofa/simulation/InitVisitor.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/simulation/Node.h>

namespace sofa
{

namespace component
{

namespace collision
{

using sofa::component::collision::BodyPicked;

template <class DataTypes>
void TComponentMouseInteraction<DataTypes>::createInteractionComponents( sofa::simulation::Node* parent,  sofa::simulation::Node* current)
{
    if( parent )
    {
        current->setName( current->getName() + "_" + DataTypes::Name() );

        mouseInSofa = sofa::core::objectmodel::New< MouseContainer >();
        mouseInSofa->resize(1);
        mouseInSofa->setName("MappedMousePosition");
        current->addObject(mouseInSofa);

        mouseInteractor = sofa::core::objectmodel::New< Interactor >();
        mouseInteractor->setName("MouseInteractor");
        current->addObject(mouseInteractor);

        MousePosition *mecha = dynamic_cast< MousePosition* >(parent->getMechanicalState());

        this->mouseMapping = sofa::core::objectmodel::New< IdentityMechanicalMapping >();
        this->mouseMapping->setModels(mecha, static_cast< MouseContainer* >(mouseInSofa.get()));

        current->addObject(mouseMapping);

        mouseMapping->setNonMechanical();
        mouseInSofa->init();
        mouseInteractor->init();
        mouseMapping->init();
    }
}

template <class DataTypes>
bool TComponentMouseInteraction<DataTypes>::isCompatible( core::objectmodel::BaseContext *context) const
{
    return (dynamic_cast<MouseContainer*>(context->getMechanicalState()) != NULL);
}
}
}
}

#endif
