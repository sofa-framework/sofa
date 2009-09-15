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
#ifndef SOFA_COMPONENT_COLLISION_FIXPARTICLEPERFORMER_H
#define SOFA_COMPONENT_COLLISION_FIXPARTICLEPERFORMER_H

#include <sofa/component/collision/InteractionPerformer.h>



#include <sofa/component/collision/BarycentricContactMapper.h>
#include <sofa/component/forcefield/StiffSpringForceField.h>

namespace sofa
{

namespace component
{

namespace collision
{

class FixParticlePerformerConfiguration
{
public:
    void setStiffness(double s) {stiffness=s;}
protected:
    SReal stiffness;
};

template <class DataTypes>
class FixParticlePerformer: public TInteractionPerformer<DataTypes>, public FixParticlePerformerConfiguration
{
    typedef sofa::component::collision::BaseContactMapper< DataTypes >        MouseContactMapper;
    typedef sofa::component::forcefield::StiffSpringForceField< DataTypes >   MouseForceField;
    typedef sofa::component::container::MechanicalObject< DataTypes >         MouseContainer;
public:
    FixParticlePerformer(BaseMouseInteractor *i);
    ~FixParticlePerformer();

    void start();
    void execute();
    void draw();

protected:
    MouseContactMapper   *mapper;
    std::map< core::CollisionModel*, MouseContactMapper* > mapperFixations;
    std::vector< simulation::Node * > fixations;
    std::vector< core::componentmodel::behavior::BaseConstraint * > fixationConstraint;
};


#if defined(WIN32) && !defined(SOFA_COMPONENT_COLLISION_FIXPARTICLEPERFORMER_CPP)
extern template class SOFA_COMPONENT_COLLISION_API FixParticlePerformer<defaulttype::Vec3Types>;
#endif


}
}
}

#endif
