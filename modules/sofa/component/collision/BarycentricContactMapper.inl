/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_BARYCENTRICCONTACTMAPPER_INL
#define SOFA_COMPONENT_COLLISION_BARYCENTRICCONTACTMAPPER_INL

#include <sofa/component/collision/BarycentricContactMapper.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/simulation/tree/Simulation.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace collision
{

template < class TCollisionModel, class DataTypes >
void BarycentricContactMapper<TCollisionModel,DataTypes>::cleanup()
{
    if (mapping!=NULL)
    {
        simulation::tree::GNode* parent = dynamic_cast<simulation::tree::GNode*>(model->getContext());
        if (parent!=NULL)
        {
            simulation::tree::GNode* child = dynamic_cast<simulation::tree::GNode*>(mapping->getContext());
            child->removeObject(mapping->getTo());
            child->removeObject(mapping);
            parent->removeChild(child);
            delete mapping->getTo();
            delete mapping;
            delete child;
            mapping = NULL;
        }
    }
}

template < class TCollisionModel, class DataTypes >
typename BarycentricContactMapper<TCollisionModel,DataTypes>::MMechanicalState* BarycentricContactMapper<TCollisionModel,DataTypes>::createMapping(const char* name)
{
    if (model==NULL) return NULL;
    simulation::tree::GNode* parent = dynamic_cast<simulation::tree::GNode*>(model->getContext());
    if (parent==NULL)
    {
        std::cerr << "ERROR: BarycentricContactMapper only works for scenegraph scenes.\n";
        return NULL;
    }
    simulation::tree::GNode* child = new simulation::tree::GNode(name); parent->addChild(child); child->updateSimulationContext();
    MMechanicalState* mstate = new MMechanicalObject; child->addObject(mstate);
    //mapping = new MMapping(model->getMechanicalState(), mstate, model->getTopology());
    //mapper = mapping->getMapper();
    mapper = new mapping::BarycentricMapperMeshTopology<InDataTypes, DataTypes>(model->getTopology());
    mapping = new MMapping(model->getMechanicalState(), mstate, mapper);
    child->addObject(mapping);
    return mstate;
}

template < class TCollisionModel, class DataTypes >
void IdentityContactMapper<TCollisionModel,DataTypes>::cleanup()
{
    if (mapping!=NULL)
    {
        simulation::tree::GNode* parent = dynamic_cast<simulation::tree::GNode*>(model->getContext());
        if (parent!=NULL)
        {
            simulation::tree::GNode* child = dynamic_cast<simulation::tree::GNode*>(mapping->getContext());
            child->removeObject(mapping->getTo());
            child->removeObject(mapping);
            parent->removeChild(child);
            delete mapping->getTo();
            delete mapping;
            delete child;
            mapping = NULL;
        }
    }
}
template < class TCollisionModel, class DataTypes >
typename IdentityContactMapper<TCollisionModel,DataTypes>::MMechanicalState* IdentityContactMapper<TCollisionModel,DataTypes>::createMapping(const char* name)
{
    if (model==NULL) return NULL;
    simulation::tree::GNode* parent = dynamic_cast<simulation::tree::GNode*>(model->getContext());
    if (parent==NULL)
    {
        std::cerr << "ERROR: IdentityContactMapper only works for scenegraph scenes.\n";
        return NULL;
    }
    simulation::tree::GNode* child = new simulation::tree::GNode(name); parent->addChild(child); child->updateSimulationContext();
    MMechanicalState* mstate = new MMechanicalObject; child->addObject(mstate);
    mapping = new MMapping(model->getMechanicalState(), mstate); child->addObject(mapping);
    return mstate;
}

template < class TCollisionModel, class DataTypes >
void RigidContactMapper<TCollisionModel,DataTypes>::cleanup()
{
    if (child!=NULL)
    {
        simulation::tree::getSimulation()->unload(child);
        child = NULL;
    }
}

template < class TCollisionModel, class DataTypes >
typename RigidContactMapper<TCollisionModel,DataTypes>::MMechanicalState* RigidContactMapper<TCollisionModel,DataTypes>::createMapping(const char* name)
{
    if (model==NULL) return NULL;
    InMechanicalState* instate = model->getMechanicalState();
    if (instate!=NULL)
    {
        simulation::tree::GNode* parent = dynamic_cast<simulation::tree::GNode*>(instate->getContext());
        if (parent==NULL)
        {
            std::cerr << "ERROR: RigidContactMapper only works for scenegraph scenes.\n";
            return NULL;
        }
        child = new simulation::tree::GNode(name); parent->addChild(child); child->updateSimulationContext();
        outmodel = new MMechanicalObject; child->addObject(outmodel);
        mapping = new MMapping(instate, outmodel); child->addObject(mapping);
    }
    else
    {
        simulation::tree::GNode* parent = dynamic_cast<simulation::tree::GNode*>(model->getContext());
        if (parent==NULL)
        {
            std::cerr << "ERROR: RigidContactMapper only works for scenegraph scenes.\n";
            return NULL;
        }
        child = new simulation::tree::GNode(name); parent->addChild(child); child->updateSimulationContext();
        outmodel = new MMechanicalObject; child->addObject(outmodel);
        mapping = NULL;
    }
    return outmodel;
}

template < class TCollisionModel, class DataTypes >
void SubsetContactMapper<TCollisionModel,DataTypes>::cleanup()
{
    if (child!=NULL)
    {
        simulation::tree::getSimulation()->unload(child);
        child = NULL;
    }
}

template < class TCollisionModel, class DataTypes >
typename SubsetContactMapper<TCollisionModel,DataTypes>::MMechanicalState* SubsetContactMapper<TCollisionModel,DataTypes>::createMapping(const char* name)
{
    if (model==NULL) return NULL;
    InMechanicalState* instate = model->getMechanicalState();
    if (instate!=NULL)
    {
        simulation::tree::GNode* parent = dynamic_cast<simulation::tree::GNode*>(instate->getContext());
        if (parent==NULL)
        {
            std::cerr << "ERROR: SubsetContactMapper only works for scenegraph scenes.\n";
            return NULL;
        }
        child = new simulation::tree::GNode(name); parent->addChild(child); child->updateSimulationContext();
        outmodel = new MMechanicalObject; child->addObject(outmodel);
        mapping = new MMapping(instate, outmodel); child->addObject(mapping);
    }
    else
    {
        simulation::tree::GNode* parent = dynamic_cast<simulation::tree::GNode*>(model->getContext());
        if (parent==NULL)
        {
            std::cerr << "ERROR: SubsetContactMapper only works for scenegraph scenes.\n";
            return NULL;
        }
        child = new simulation::tree::GNode(name); parent->addChild(child); child->updateSimulationContext();
        outmodel = new MMechanicalObject; child->addObject(outmodel);
        mapping = NULL;
    }
    return outmodel;
}

template <class DataTypes>
typename ContactMapper<RigidDistanceGridCollisionModel,DataTypes>::MMechanicalState* ContactMapper<RigidDistanceGridCollisionModel,DataTypes>::createMapping(const char* name)
{
    MMechanicalState* outmodel = Inherit::createMapping(name);
    if (this->child!=NULL && this->mapping==NULL)
    {
        // add velocity visualization
        sofa::component::visualmodel::DrawV* visu = new sofa::component::visualmodel::DrawV;
        this->child->addObject(visu);
        visu->useAlpha.setValue(true);
        visu->vscale.setValue(this->model->getContext()->getDt());
        sofa::component::mapping::IdentityMapping< core::Mapping< core::componentmodel::behavior::State<DataTypes>, core::componentmodel::behavior::MappedModel< ExtVectorTypes< Vec<3,GLfloat>, Vec<3,GLfloat> > > > >* map = new sofa::component::mapping::IdentityMapping< core::Mapping< core::componentmodel::behavior::State<DataTypes> , core::componentmodel::behavior::MappedModel< ExtVectorTypes< Vec<3,GLfloat>, Vec<3,GLfloat> > > > > ( outmodel, visu );
        this->child->addObject(map);
        visu->init();
        map->init();
    }
    return outmodel;
}

} // namespace collision

} // namespace component

} // namespace sofa

#endif
