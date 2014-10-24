#ifndef BULLET_RIGID_CONTACT_MAPPER_INL
#define BULLET_RIGID_CONTACT_MAPPER_INL

#include "BulletRigidContactMapper.h"
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
void BulletRigidContactMapper<TCollisionModel,DataTypes>::cleanup()
{
    if (child!=NULL)
    {
        child->detachFromGraph();
        child->execute<simulation::DeleteVisitor>(sofa::core::ExecParams::defaultInstance());
        child.reset();
    }
}

template < class TCollisionModel, class DataTypes >
typename BulletRigidContactMapper<TCollisionModel,DataTypes>::MMechanicalState* BulletRigidContactMapper<TCollisionModel,DataTypes>::createMapping(const char* name)
{
    if (model==NULL) return NULL;
    InMechanicalState* instate = model->getMechanicalState();
    if (instate!=NULL)
    {
        simulation::Node* parent = dynamic_cast<simulation::Node*>(instate->getContext());
        if (parent==NULL)
        {
            std::cerr << "ERROR: BulletRigidContactMapper only works for scenegraph scenes.\n";
            return NULL;
        }
        child = parent->createChild(name);
        outmodel = sofa::core::objectmodel::New<MMechanicalObject>();
        child->addObject(outmodel);
        outmodel->useMask.setValue(true);
        mapping = sofa::core::objectmodel::New<MMapping>();
        mapping->setModels(instate,outmodel.get());
        mapping->init();
        child->addObject(mapping);
    }
    else
    {
        simulation::Node* parent = dynamic_cast<simulation::Node*>(model->getContext());
        if (parent==NULL)
        {
            std::cerr << "ERROR: BulletRigidContactMapper only works for scenegraph scenes.\n";
            return NULL;
        }
        child = parent->createChild(name);
        outmodel = sofa::core::objectmodel::New<MMechanicalObject>(); child->addObject(outmodel);
        outmodel->useMask.setValue(true);
        mapping = NULL;
    }
    return outmodel.get();
}

} // namespace collision

} // namespace component

} // namespace sofa

#endif
