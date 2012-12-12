/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_COLLISION_IDENTITYCONTACTMAPPER_INL
#define SOFA_COMPONENT_COLLISION_IDENTITYCONTACTMAPPER_INL

#include <sofa/component/collision/IdentityContactMapper.h>
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
        std::cerr << "ERROR: IdentityContactMapper only works for scenegraph scenes.\n";
        return NULL;
    }
    simulation::Node::SPtr child = parent->createChild(name);
    typename MMechanicalState::SPtr mstate = sofa::core::objectmodel::New<MMechanicalObject>(); child->addObject(mstate);
    mstate->useMask.setValue(true);
    mapping = sofa::core::objectmodel::New<MMapping>(model->getMechanicalState(), mstate); child->addObject(mapping);
    return mstate.get();
}

//template <class DataTypes>
//int ContactMapper<CapsuleModel, DataTypes>::addPoint(const typename DataTypes::Coord& C, int index, typename DataTypes::Real& r){
//    const Coord & cap_center = this->model->center(index);
//    Vector3 cap_p1 = this->model->point1(index);
//    Vector3 cap_p2 = this->model->point2(index);
//    double cap_rad = this->model->radius(index);

//    Vector3 AB = cap_p2 - cap_p1;
//    Vector3 AC = C - cap_p1;
//    Vector3 P_on_capsule;
//    double alpha = (AB * AC)/AB.norm2();
//    std::cout<<"alpha "<<alpha<<std::endl;
//    if(alpha < 0.000001){
//        Vector3 PC = C - cap_p1;
//        PC.normalize();

//        P_on_capsule = cap_p1 + cap_rad * PC;
//    }
//    else if(alpha > 0.999999){
//        Vector3 PC = C - cap_p2;
//        PC.normalize();

//        P_on_capsule = cap_p2 + cap_rad * PC;
//    }
//    else{
//        Vector3 P = cap_p1 + alpha * AB;
//        Vector3 PC = C - P;
//        PC.normalize();

//        P_on_capsule = P + cap_rad * PC;
//    }

//    Vector3 cap_center_P_on_capsule = P_on_capsule - cap_center;
//    r = cap_center_P_on_capsule.norm();
//    std::cout<<"r "<<r<<std::endl;

//    return index;
//}


} // namespace collision

} // namespace component

} // namespace sofa

#endif
