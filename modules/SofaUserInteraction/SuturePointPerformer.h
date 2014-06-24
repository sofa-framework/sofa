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
#ifndef SOFA_COMPONENT_COLLISION_SUTUREPOINTPERFORMER_H
#define SOFA_COMPONENT_COLLISION_SUTUREPOINTPERFORMER_H

#include <SofaUserInteraction/InteractionPerformer.h>
#include <SofaDeformable/StiffSpringForceField.h>
#include <SofaDeformable/SpringForceField.h>
#include <SofaBoundaryCondition/FixedConstraint.h>

#include <SofaUserInteraction/MouseInteractor.h>

namespace sofa
{

namespace component
{

namespace collision
{

class SuturePointPerformerConfiguration
{
public:
    void setStiffness (double f) {stiffness=f;}
    void setDamping (double f) {damping=f;}

protected:
    double stiffness;
    double damping;
};


template <class DataTypes>
class SOFA_USER_INTERACTION_API SuturePointPerformer: public TInteractionPerformer<DataTypes>, public SuturePointPerformerConfiguration
{
public:
    typedef typename DataTypes::Real Real;
    typedef sofa::component::interactionforcefield::LinearSpring<Real> Spring;
    typedef sofa::component::interactionforcefield::StiffSpringForceField<DataTypes> SpringObjectType;
    typedef sofa::component::projectiveconstraintset::FixedConstraint<DataTypes> FixObjectType;

    SuturePointPerformer(BaseMouseInteractor *i);
    ~SuturePointPerformer();

    void start();
    void execute() {};
    void draw() {};

protected:
    bool first;
    unsigned int fixedIndex;

    sofa::helper::vector <Spring> addedSprings;

    BodyPicked firstPicked;
    SpringObjectType *SpringObject;
    FixObjectType *FixObject;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_COLLISION_SUTUREPOINTPERFORMER_CPP)
#ifndef SOFA_DOUBLE
extern template class SOFA_USER_INTERACTION_API  SuturePointPerformer<defaulttype::Vec3fTypes>;
#endif
#ifndef SOFA_FLOAT
extern template class SOFA_USER_INTERACTION_API  SuturePointPerformer<defaulttype::Vec3dTypes>;
#endif
#endif

}
}
}

#endif
