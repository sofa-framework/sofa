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
#ifndef SOFA_COMPONENT_COLLISION_SUBSETCONTACTMAPPER_INL
#define SOFA_COMPONENT_COLLISION_SUBSETCONTACTMAPPER_INL

#include <SofaMeshCollision/SubsetContactMapper.h>
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
void SubsetContactMapper<TCollisionModel,DataTypes>::cleanup()
{
    if (child!=NULL)
    {
        child->detachFromGraph();
        child->execute<simulation::DeleteVisitor>(sofa::core::ExecParams::defaultInstance());
        child.reset();
    }
}

template < class TCollisionModel, class DataTypes >
typename SubsetContactMapper<TCollisionModel,DataTypes>::MMechanicalState* SubsetContactMapper<TCollisionModel,DataTypes>::createMapping(const char* name)
{
    if (model==NULL) return NULL;
    InMechanicalState* instate = model->getMechanicalState();
    if (instate!=NULL)
    {
        simulation::Node* parent = dynamic_cast<simulation::Node*>(instate->getContext());
        if (parent==NULL)
        {
            msg_error("SubsetContactMapper") << "SubsetContactMapper only works for scenegraph scenes.";
            return NULL;
        }
        child = parent->createChild(name);
        outmodel = sofa::core::objectmodel::New<MMechanicalObject>(); child->addObject(outmodel);
        mapping = sofa::core::objectmodel::New<MMapping>();
        child->addObject(mapping);
        mapping->setModels(instate, outmodel.get());
    }
    else
    {
        simulation::Node* parent = dynamic_cast<simulation::Node*>(model->getContext());
        if (parent==NULL)
        {
            msg_error("SubsetContactMapper") << "SubsetContactMapper only works for scenegraph scenes.";
            return NULL;
        }
        child = parent->createChild(name);
        outmodel = sofa::core::objectmodel::New<MMechanicalObject>(); child->addObject(outmodel);
        mapping = NULL;
    }
    return outmodel.get();
}
} // namespace collision

} // namespace component

} // namespace sofa

#endif /* SOFA_COMPONENT_COLLISION_SUBSETCONTACTMAPPER_INL */
