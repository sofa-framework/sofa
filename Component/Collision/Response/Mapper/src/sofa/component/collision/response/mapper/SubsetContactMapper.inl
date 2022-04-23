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
#include <sofa/component/collision/response/mapper/SubsetContactMapper.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/DeleteVisitor.h>

namespace sofa::component::collision::response::mapper
{

template < class TCollisionModel, class DataTypes >
void SubsetContactMapper<TCollisionModel,DataTypes>::cleanup()
{
    if (child!=nullptr)
    {
        child->detachFromGraph();
        child->execute<simulation::DeleteVisitor>(sofa::core::execparams::defaultInstance());
        child.reset();
    }
}

template < class TCollisionModel, class DataTypes >
typename SubsetContactMapper<TCollisionModel,DataTypes>::MMechanicalState* SubsetContactMapper<TCollisionModel,DataTypes>::createMapping(const char* name)
{
    if (model==nullptr) return nullptr;
    InMechanicalState* instate = model->getMechanicalState();
    if (instate!=nullptr)
    {
        simulation::Node* parent = dynamic_cast<simulation::Node*>(instate->getContext());
        if (parent==nullptr)
        {
            msg_error("SubsetContactMapper") << "SubsetContactMapper only works for scenegraph scenes.";
            return nullptr;
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
        if (parent==nullptr)
        {
            msg_error("SubsetContactMapper") << "SubsetContactMapper only works for scenegraph scenes.";
            return nullptr;
        }
        child = parent->createChild(name);
        outmodel = sofa::core::objectmodel::New<MMechanicalObject>(); child->addObject(outmodel);
        mapping = nullptr;
    }
    return outmodel.get();
}


template < class TCollisionModel, class DataTypes >
SubsetContactMapper<TCollisionModel,DataTypes>::SubsetContactMapper()
    : model(nullptr), child(nullptr), mapping(nullptr), outmodel(nullptr), nbp(0), needInit(false)
{
}


template < class TCollisionModel, class DataTypes >
void SubsetContactMapper<TCollisionModel,DataTypes>::setCollisionModel(MCollisionModel* model)
{
    this->model = model;
}

template < class TCollisionModel, class DataTypes >
void SubsetContactMapper<TCollisionModel,DataTypes>::resize(Size size)
{
    if (mapping!=nullptr)
        mapping->clear(size);
    if (outmodel!=nullptr)
        outmodel->resize(size);
    nbp = 0;
}

template < class TCollisionModel, class DataTypes >
typename SubsetContactMapper<TCollisionModel, DataTypes>::Index SubsetContactMapper<TCollisionModel,DataTypes>::addPoint(const Coord& P, Index index, Real&)
{
    Index i = nbp++;
    if (outmodel->getSize() <= i)
        outmodel->resize(i+1);
    if (mapping)
    {
        i = mapping->addPoint(index);
        needInit = true;
    }
    else
    {
        helper::WriteAccessor<Data<VecCoord> > d_x = *outmodel->write(core::VecCoordId::position());
        VecCoord& x = d_x.wref();
        x[i] = P;
    }
    return i;
}

template < class TCollisionModel, class DataTypes >
void SubsetContactMapper<TCollisionModel,DataTypes>::update()
{
    if (mapping!=nullptr)
    {
        if (needInit)
        {
            mapping->init();
            needInit = false;
        }
        core::BaseMapping* map = mapping.get();
        map->apply(core::mechanicalparams::defaultInstance(), core::VecCoordId::position(), core::ConstVecCoordId::position());
        map->applyJ(core::mechanicalparams::defaultInstance(), core::VecDerivId::velocity(), core::ConstVecDerivId::velocity());
    }
}

template < class TCollisionModel, class DataTypes >
void SubsetContactMapper<TCollisionModel,DataTypes>::updateXfree()
{
    if (mapping!=nullptr)
    {
        if (needInit)
        {
            mapping->init();
            needInit = false;
        }

        core::BaseMapping* map = mapping.get();
        map->apply(core::mechanicalparams::defaultInstance(), core::VecCoordId::freePosition(), core::ConstVecCoordId::freePosition());
    }
}

} //namespace sofa::component::collision::response::mapper
