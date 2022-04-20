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
#include <sofa/component/collision/response/mapper/RigidContactMapper.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/DeleteVisitor.h>

namespace sofa::component::collision::response::mapper
{

template < class TCollisionModel, class DataTypes >
RigidContactMapper<TCollisionModel,DataTypes>::RigidContactMapper()
    : model(nullptr)
    , child(nullptr)
    , mapping(nullptr)
    , outmodel(nullptr)
    , nbp(0)
{
}

template < class TCollisionModel, class DataTypes >
void RigidContactMapper<TCollisionModel,DataTypes>::cleanup()
{
    if (child!=nullptr)
    {
        child->detachFromGraph();
        child->execute<simulation::DeleteVisitor>(sofa::core::execparams::defaultInstance());
        child.reset();
    }
}

template < class TCollisionModel, class DataTypes >
typename RigidContactMapper<TCollisionModel,DataTypes>::MMechanicalState* RigidContactMapper<TCollisionModel,DataTypes>::createMapping(const char* name)
{
    if (model==nullptr) return nullptr;
    InMechanicalState* instate = model->getMechanicalState();
    if (instate!=nullptr)
    {
        simulation::Node* parent = dynamic_cast<simulation::Node*>(instate->getContext());
        if (parent==nullptr)
        {
            msg_error("RigidContactMapper") << "RigidContactMapper only works for scenegraph scenes.";
            return nullptr;
        }
        child = parent->createChild(name);
        outmodel = sofa::core::objectmodel::New<MMechanicalObject>();
        child->addObject(outmodel);
        mapping = sofa::core::objectmodel::New<MMapping>();
        mapping->setModels(instate,outmodel.get());
        mapping->init();
        child->addObject(mapping);
    }
    else
    {
        simulation::Node* parent = dynamic_cast<simulation::Node*>(model->getContext());
        if (parent==nullptr)
        {
            msg_error("RigidContactMapper") << "RigidContactMapper only works for scenegraph scenes.";
            return nullptr;
        }
        child = parent->createChild(name);
        outmodel = sofa::core::objectmodel::New<MMechanicalObject>(); child->addObject(outmodel);
        mapping = nullptr;
    }
    return outmodel.get();
}

} //namespace sofa::component::collision::response::mapper
