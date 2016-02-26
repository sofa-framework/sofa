/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define FRAME_FRAMEVOLUMEPRESERVATIONFORCEFIELD_CPP

#include "FrameVolumePreservationForceField.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(FrameVolumePreservationForceField)

// Register in the Factory
int FrameVolumePreservationForceFieldClass = core::RegisterObject("Compute volume preservation forces on deformation gradients")
#ifndef SOFA_FLOAT
        .add< FrameVolumePreservationForceField<DeformationGradient331dTypes> >()
        .add< FrameVolumePreservationForceField<DeformationGradient332dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< FrameVolumePreservationForceField<DeformationGradient331fTypes> >()
        .add< FrameVolumePreservationForceField<DeformationGradient332fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_FRAME_API FrameVolumePreservationForceField<DeformationGradient331dTypes>;
template class SOFA_FRAME_API FrameVolumePreservationForceField<DeformationGradient332dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_FRAME_API FrameVolumePreservationForceField<DeformationGradient331fTypes>;
template class SOFA_FRAME_API FrameVolumePreservationForceField<DeformationGradient332fTypes>;
#endif



}
}

} // namespace sofa


