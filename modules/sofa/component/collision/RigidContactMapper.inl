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
#ifndef SOFA_COMPONENT_COLLISION_RIGIDCONTACTMAPPER_INL
#define SOFA_COMPONENT_COLLISION_RIGIDCONTACTMAPPER_INL

#include <sofa/component/collision/RigidContactMapper.h>
#include <sofa/core/visual/VisualParams.h>
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
void RigidContactMapper<TCollisionModel,DataTypes>::cleanup()
{
    if (child!=NULL)
    {
        child->detachFromGraph();
        child->execute<simulation::DeleteVisitor>(sofa::core::ExecParams::defaultInstance());
        child.reset();
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
        child = parent->createChild(name);
        outmodel = sofa::core::objectmodel::New<MMechanicalObject>(); child->addObject(outmodel);
        outmodel->useMask.setValue(true);
        mapping = sofa::core::objectmodel::New<MMapping>(instate, outmodel.get()); child->addObject(mapping);
    }
    else
    {
        simulation::Node* parent = dynamic_cast<simulation::Node*>(model->getContext());
        if (parent==NULL)
        {
            std::cerr << "ERROR: RigidContactMapper only works for scenegraph scenes.\n";
            return NULL;
        }
        child = parent->createChild(name);
        outmodel = sofa::core::objectmodel::New<MMechanicalObject>(); child->addObject(outmodel);
        outmodel->useMask.setValue(true);
        mapping = NULL;
    }
    return outmodel.get();
}

template <class DataTypes>
typename ContactMapper<RigidDistanceGridCollisionModel,DataTypes>::MMechanicalState* ContactMapper<RigidDistanceGridCollisionModel,DataTypes>::createMapping(const char* name)
{
    using sofa::component::mapping::IdentityMapping;

    MMechanicalState* outmodel = Inherit::createMapping(name);
    if (this->child!=NULL && this->mapping==NULL)
    {
        // add velocity visualization
        /*        sofa::component::visualmodel::DrawV* visu = new sofa::component::visualmodel::DrawV;
                this->child->addObject(visu);
                visu->useAlpha.setValue(true);
                visu->vscale.setValue(this->model->getContext()->getDt());
                IdentityMapping< DataTypes, ExtVectorTypes< Vec<3,GLfloat>, Vec<3,GLfloat> > > * map = new IdentityMapping< DataTypes, ExtVectorTypes< Vec<3,GLfloat>, Vec<3,GLfloat> > >( outmodel, visu );
                this->child->addObject(map);
                visu->init();
                map->init(); */
    }
    return outmodel;
}


} // namespace collision

} // namespace component

} // namespace sofa

#endif /* SOFA_COMPONENT_COLLISION_RIGIDCONTACTMAPPER_INL */
