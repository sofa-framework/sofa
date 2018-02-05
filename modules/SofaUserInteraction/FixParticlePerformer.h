/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_COLLISION_FIXPARTICLEPERFORMER_H
#define SOFA_COMPONENT_COLLISION_FIXPARTICLEPERFORMER_H
#include "config.h"

#include <SofaUserInteraction/InteractionPerformer.h>

#include <SofaDeformable/StiffSpringForceField.h>
#include <SofaUserInteraction/MouseInteractor.h>

namespace sofa
{

namespace component
{

namespace collision
{

class FixParticlePerformerConfiguration
{
public:
    void setStiffness(SReal s) {stiffness=s;}
protected:
    SReal stiffness;
};

template <class DataTypes>
class FixParticlePerformer: public TInteractionPerformer<DataTypes>, public FixParticlePerformerConfiguration
{
public:
    typedef sofa::component::interactionforcefield::StiffSpringForceField< DataTypes >   MouseForceField;
    typedef sofa::component::container::MechanicalObject< DataTypes >         MouseContainer;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;

    FixParticlePerformer(BaseMouseInteractor *i);

    void start();
    void execute();
    void draw(const core::visual::VisualParams* vparams);

protected:
    MouseContainer* getFixationPoints(const BodyPicked &b, helper::vector<unsigned int> &points, typename DataTypes::Coord &fixPoint);

    std::vector< simulation::Node * > fixations;
};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_COLLISION_FIXPARTICLEPERFORMER_CPP)
#ifndef SOFA_DOUBLE
extern template class SOFA_USER_INTERACTION_API FixParticlePerformer<defaulttype::Vec3fTypes>;
#endif
#ifndef SOFA_FLOAT
extern template class SOFA_USER_INTERACTION_API FixParticlePerformer<defaulttype::Vec3dTypes>;
#endif
#endif


}
}
}

#endif
