/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include "OpenCLTypes.h"

#include "OpenCLMechanicalObject.h"
#include "OpenCLIdentityMapping.h"
#include "OpenCLFixedConstraint.h"
#include "OpenCLSpringForceField.h"

#include <SofaUserInteraction/MouseInteractor.inl>
#include <SofaUserInteraction/ComponentMouseInteraction.inl>
#include <SofaUserInteraction/AttachBodyPerformer.inl>
#include <SofaUserInteraction/FixParticlePerformer.inl>
#include <sofa/helper/Factory.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace collision
{
	
using namespace sofa::gpu::opencl;

template class MouseInteractor<OpenCLVec3fTypes>;
template class TComponentMouseInteraction< OpenCLVec3fTypes >;
template class AttachBodyPerformer< OpenCLVec3fTypes >;
template class FixParticlePerformer< OpenCLVec3fTypes >;

#ifdef SOFAOPENCL_DOUBLE
template class MouseInteractor<OpenCLVec3dTypes>;
template class TComponentMouseInteraction< OpenCLVec3dTypes >;
template class AttachBodyPerformer< OpenCLVec3dTypes >;
template class FixParticlePerformer< OpenCLVec3dTypes >;
#endif

helper::Creator<ComponentMouseInteraction::ComponentMouseInteractionFactory, TComponentMouseInteraction<OpenCLVec3fTypes> > ComponentMouseInteractionOpenCLVec3fClass ("MouseSpringOpenCLVec3f",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer <OpenCLVec3fTypes> >  AttachBodyPerformerOpenCLVec3fClass("AttachBody",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, FixParticlePerformer<OpenCLVec3fTypes> >  FixParticlePerformerOpenCLVec3fClass("FixParticle",true);

#ifdef SOFAOPENCL_DOUBLE
helper::Creator<ComponentMouseInteraction::ComponentMouseInteractionFactory, TComponentMouseInteraction<OpenCLVec3dTypes> > ComponentMouseInteractionOpenCLVec3dClass ("MouseSpringOpenCLVec3d",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, AttachBodyPerformer <OpenCLVec3dTypes> >  AttachBodyPerformerOpenCLVec3dClass("AttachBody",true);
helper::Creator<InteractionPerformer::InteractionPerformerFactory, FixParticlePerformer<OpenCLVec3dTypes> >  FixParticlePerformerOpenCLVec3dClass("FixParticle",true);
#endif

} //namespace collision

} //namespace component


namespace gpu
{

namespace opencl
{

SOFA_DECL_CLASS(OpenCLMouseInteractor)

int MouseInteractorOpenCLClass = core::RegisterObject("Supports Mouse Interaction using OPENCL")
        .add< component::collision::MouseInteractor<OpenCLVec3fTypes> >()
#ifdef SOFAOPENCL_DOUBLE
        .add< component::collision::MouseInteractor<OpenCLVec3dTypes> >()
#endif
        ;
		
}

}

}
