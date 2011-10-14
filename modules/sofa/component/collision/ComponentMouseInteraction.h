/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
 *                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
 *                                                                             *
 * This program is free software; you can redistribute it and/or modify it     *
 * under the terms of the GNU General Public License as published by the Free  *
 * Software Foundation; either version 2 of the License, or (at your option)   *
 * any later version.                                                          *
 *                                                                             *
 * This program is distributed in the hope that it will be useful, but WITHOUT *
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
 * more details.                                                               *
 *                                                                             *
 * You should have received a copy of the GNU General Public License along     *
 * with this program; if not, write to the Free Software Foundation, Inc., 51  *
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
 *******************************************************************************
 *                            SOFA :: Applications                             *
 *                                                                             *
 * Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
 * H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
 * M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
 *                                                                             *
 * Contact information: contact@sofa-framework.org                             *
 ******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_COMPONENTMOUSEINTERACTION_H
#define SOFA_COMPONENT_COLLISION_COMPONENTMOUSEINTERACTION_H

#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/common/DeleteVisitor.h>
#include <sofa/component/collision/MouseInteractor.h>
#include <sofa/component/mapping/IdentityMapping.h>
#include <sofa/component/component.h>

namespace sofa
{

namespace simulation
{
class Node;
}
namespace component
{

namespace collision
{


class SOFA_USER_INTERACTION_API ComponentMouseInteraction
{
public:
    ComponentMouseInteraction();

    virtual ~ComponentMouseInteraction();

    virtual void createInteractionComponents(sofa::simulation::Node* parent,
            sofa::simulation::Node* current) = 0;

    void attach(simulation::Node* parentNode);

    void detach();

    void reset();

    virtual bool isCompatible( core::objectmodel::BaseContext *)const=0;

    typedef helper::Factory<std::string, ComponentMouseInteraction, core::objectmodel::BaseContext*> ComponentMouseInteractionFactory;

    template <class RealObject>
    static void create( RealObject*& obj, core::objectmodel::BaseContext* /* context */)
    {
        obj = new RealObject;
    }

    //Components
    simulation::Node* nodeRayPick;
    sofa::core::behavior::BaseMechanicalState::SPtr mouseInSofa;
    sofa::core::BaseMapping::SPtr mouseMapping;
    sofa::component::collision::BaseMouseInteractor::SPtr mouseInteractor;
};



template <class DataTypes>
class TComponentMouseInteraction : public ComponentMouseInteraction
{
    typedef sofa::component::container::MechanicalObject< defaulttype::Vec3Types > MousePosition;
    typedef sofa::component::container::MechanicalObject< DataTypes > MouseContainer;
    typedef sofa::component::collision::MouseInteractor< DataTypes > Interactor;
    typedef sofa::component::mapping::IdentityMapping< defaulttype::Vec3Types, DataTypes > IdentityMechanicalMapping;

public:


    void createInteractionComponents(sofa::simulation::Node* parent, sofa::simulation::Node* current);

    bool  isCompatible( core::objectmodel::BaseContext *context) const;

};

#if defined(WIN32) && !defined(SOFA_COMPONENT_COLLISION_COMPONENTMOUSEINTERACTION_CPP)
#ifndef SOFA_DOUBLE
extern template class SOFA_USER_INTERACTION_API TComponentMouseInteraction<defaulttype::Vec3fTypes>;
extern template class SOFA_USER_INTERACTION_API TComponentMouseInteraction<defaulttype::Rigid3fTypes>;
#endif
#ifndef SOFA_FLOAT
extern template class SOFA_USER_INTERACTION_API TComponentMouseInteraction<defaulttype::Vec3dTypes>;
extern template class SOFA_USER_INTERACTION_API TComponentMouseInteraction<defaulttype::Rigid3dTypes>;
#endif
extern template class SOFA_USER_INTERACTION_API helper::Factory<std::string, ComponentMouseInteraction, core::objectmodel::BaseContext*>;
#endif
}
}
}

#endif
