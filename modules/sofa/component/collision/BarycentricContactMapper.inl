/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/common/DeleteVisitor.h>
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
        simulation::Node* parent = dynamic_cast<simulation::Node*>(model->getContext());
        if (parent!=NULL)
        {
            simulation::Node* child = dynamic_cast<simulation::Node*>(mapping->getContext());
            child->detachFromGraph();
            child->execute<simulation::DeleteVisitor>();
            delete child;
            mapping = NULL;
        }
    }
}

template < class TCollisionModel, class DataTypes >
typename BarycentricContactMapper<TCollisionModel,DataTypes>::MMechanicalState* BarycentricContactMapper<TCollisionModel,DataTypes>::createMapping(const char* name)
{
    if (model==NULL) return NULL;
    simulation::Node* parent = dynamic_cast<simulation::Node*>(model->getContext());
    if (parent==NULL)
    {
        std::cerr << "ERROR: BarycentricContactMapper only works for scenegraph scenes.\n";
        return NULL;
    }
    simulation::Node* child = simulation::getSimulation()->newNode(name);
    parent->addChild(child); child->updateContext();
    MMechanicalState* mstate = new MMechanicalObject; child->addObject(mstate);
    mstate->useMask.setValue(true);
    //mapping = new MMapping(model->getMechanicalState(), mstate, model->getMeshTopology());
    //mapper = mapping->getMapper();
    mapper = new mapping::BarycentricMapperMeshTopology<InDataTypes, typename BarycentricContactMapper::DataTypes>(model->getMeshTopology(), NULL, &model->getMechanicalState()->forceMask, &mstate->forceMask);
    mapping = new MMapping(model->getMechanicalState(), mstate, mapper);
    child->addObject(mapping);
    return mstate;
}


template < class TCollisionModel, class DataTypes >
void RigidContactMapper<TCollisionModel,DataTypes>::cleanup()
{
    if (child!=NULL)
    {
        child->detachFromGraph();
        child->execute<simulation::DeleteVisitor>();
        delete child;
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
        simulation::Node* parent = dynamic_cast<simulation::Node*>(instate->getContext());
        if (parent==NULL)
        {
            std::cerr << "ERROR: RigidContactMapper only works for scenegraph scenes.\n";
            return NULL;
        }
        child = simulation::getSimulation()->newNode(name);
        parent->addChild(child); child->updateSimulationContext();
        outmodel = new MMechanicalObject; child->addObject(outmodel);
        outmodel->useMask.setValue(true);
        mapping = new MMapping(instate, outmodel); child->addObject(mapping);
    }
    else
    {
        simulation::Node* parent = dynamic_cast<simulation::Node*>(model->getContext());
        if (parent==NULL)
        {
            std::cerr << "ERROR: RigidContactMapper only works for scenegraph scenes.\n";
            return NULL;
        }
        child = simulation::getSimulation()->newNode(name);
        parent->addChild(child); child->updateSimulationContext();
        outmodel = new MMechanicalObject; child->addObject(outmodel);
        outmodel->useMask.setValue(true);
        mapping = NULL;
    }
    return outmodel;
}

template < class TCollisionModel, class DataTypes >
void SubsetContactMapper<TCollisionModel,DataTypes>::cleanup()
{
    if (child!=NULL)
    {
        child->detachFromGraph();
        child->execute<simulation::DeleteVisitor>();
        delete child;
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
        simulation::Node* parent = dynamic_cast<simulation::Node*>(instate->getContext());
        if (parent==NULL)
        {
            std::cerr << "ERROR: SubsetContactMapper only works for scenegraph scenes.\n";
            return NULL;
        }
        child = simulation::getSimulation()->newNode(name);
        parent->addChild(child); child->updateSimulationContext();
        outmodel = new MMechanicalObject; child->addObject(outmodel);
        outmodel->useMask.setValue(true);
        mapping = new MMapping(instate, outmodel); child->addObject(mapping);
    }
    else
    {
        simulation::Node* parent = dynamic_cast<simulation::Node*>(model->getContext());
        if (parent==NULL)
        {
            std::cerr << "ERROR: SubsetContactMapper only works for scenegraph scenes.\n";
            return NULL;
        }
        child = simulation::getSimulation()->newNode(name);
        parent->addChild(child); child->updateSimulationContext();
        outmodel = new MMechanicalObject; child->addObject(outmodel);
        outmodel->useMask.setValue(true);
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
