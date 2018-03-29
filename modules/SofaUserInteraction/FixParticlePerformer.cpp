/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#ifndef SOFA_COMPONENT_COLLISION_FIXPARTICLEPERFORMER_CPP
#define SOFA_COMPONENT_COLLISION_FIXPARTICLEPERFORMER_CPP

#include <SofaUserInteraction/FixParticlePerformer.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/helper/Factory.inl>

namespace sofa
{

namespace component
{

namespace collision
{
#ifndef SOFA_DOUBLE
template class SOFA_USER_INTERACTION_API FixParticlePerformer<defaulttype::Vec3fTypes>;
helper::Creator<InteractionPerformer::InteractionPerformerFactory, FixParticlePerformer<defaulttype::Vec3fTypes> >  FixParticlePerformerVec3fClass("FixParticle",true);
#endif
#ifndef SOFA_FLOAT
template class SOFA_USER_INTERACTION_API FixParticlePerformer<defaulttype::Vec3dTypes>;
helper::Creator<InteractionPerformer::InteractionPerformerFactory, FixParticlePerformer<defaulttype::Vec3dTypes> >  FixParticlePerformerVec3dClass("FixParticle",true);
#endif
}
}
}

#endif
