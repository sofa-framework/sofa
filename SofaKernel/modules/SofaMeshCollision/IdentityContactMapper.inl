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
#ifndef SOFA_COMPONENT_COLLISION_IDENTITYCONTACTMAPPER_INL
#define SOFA_COMPONENT_COLLISION_IDENTITYCONTACTMAPPER_INL

#include <SofaMeshCollision/IdentityContactMapper.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/DeleteVisitor.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace collision
{

template < class TCollisionModel, class DataTypes >
void IdentityContactMapper<TCollisionModel,DataTypes>::cleanup()
{
    if (mapping!=NULL)
    {
        simulation::Node* parent = dynamic_cast<simulation::Node*>(model->getContext());
        if (parent!=NULL)
        {
            simulation::Node::SPtr child = dynamic_cast<simulation::Node*>(mapping->getContext());
            child->detachFromGraph();
            child->execute<simulation::DeleteVisitor>(sofa::core::ExecParams::defaultInstance());
            child.reset(); //delete child;
            mapping = NULL;
        }
    }
}
template < class TCollisionModel, class DataTypes >
typename IdentityContactMapper<TCollisionModel,DataTypes>::MMechanicalState* IdentityContactMapper<TCollisionModel,DataTypes>::createMapping(const char* name)
{
    if (model==NULL) return NULL;
    simulation::Node* parent = dynamic_cast<simulation::Node*>(model->getContext());
    if (parent==NULL)
    {
        msg_error("IdentityContactMapper") << "IdentityContactMapper only works for scenegraph scenes.";
        return NULL;
    }
    simulation::Node::SPtr child = parent->createChild(name);
    typename MMechanicalState::SPtr mstate = sofa::core::objectmodel::New<MMechanicalObject>(); child->addObject(mstate);
    mapping = sofa::core::objectmodel::New<MMapping>(model->getMechanicalState(), mstate); child->addObject(mapping);
    return mstate.get();
}


} // namespace collision

} // namespace component

} // namespace sofa

#endif
