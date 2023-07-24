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
#include <sofa/component/collision/response/mapper/BarycentricContactMapper.h>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperMeshTopology.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/DeleteVisitor.h>

namespace sofa::component::collision::response::mapper
{

template < class TCollisionModel, class DataTypes >
void BarycentricContactMapper<TCollisionModel,DataTypes>::cleanup()
{
    if (mapping != nullptr && model != nullptr)
    {
        auto* modelContext = model->getContext();
        if (dynamic_cast<simulation::Node*>(modelContext) != nullptr)
        {
            if (simulation::Node::SPtr child = dynamic_cast<simulation::Node*>(mapping->getContext()))
            {
                child->detachFromGraph();
                child->execute<simulation::DeleteVisitor>(sofa::core::execparams::defaultInstance());
                child.reset();
            }
            mapping.reset();
        }
    }
}

template < class TCollisionModel, class DataTypes >
typename BarycentricContactMapper<TCollisionModel,DataTypes>::MMechanicalState* BarycentricContactMapper<TCollisionModel,DataTypes>::createMapping(const char* name)
{
    if (model == nullptr)
        return nullptr;
    simulation::Node* parent = dynamic_cast<simulation::Node*>(model->getContext());
    if (parent==nullptr)
    {
        msg_error("BarycentricContactMapper") << "BarycentricContactMapper only works for scenegraph scenes.";
        return nullptr;
    }
    const simulation::Node::SPtr child = parent->createChild(name);
    typename MMechanicalObject::SPtr mstate = sofa::core::objectmodel::New<MMechanicalObject>();
    mstate->setName(GenerateStringID::generate().c_str());
    child->addObject(mstate);
    //mapper = mapping->getMapper();
    mapper = sofa::core::objectmodel::New<mapping::linear::BarycentricMapperMeshTopology<InDataTypes, typename BarycentricContactMapper::DataTypes> >(model->getCollisionTopology(), nullptr);
    mapper->setName(GenerateStringID::generate().c_str());
    mapping =  sofa::core::objectmodel::New<MMapping>(model->getMechanicalState(), mstate.get(), mapper);
    mapping->setName(GenerateStringID::generate().c_str());
    child->addObject(mapping);
    return mstate.get();
}

} //namespace sofa::component::collision::response::mapper
