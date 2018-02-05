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
#include "BeamLinearMapping_mt.inl"
#include <sofa/core/ObjectFactory.h>
//#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/Mapping.inl>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(BeamLinearMapping_mt)

//using namespace defaulttype;
// Register in the Factory
int BeamLinearMapping_mtClass = core::RegisterObject("Set the positions and velocities of points attached to a beam using linear interpolation between DOFs")

#ifndef SOFA_FLOAT
        .add< BeamLinearMapping_mt< Rigid3dTypes, Vec3dTypes > >()
        .add< BeamLinearMapping_mt< Rigid3dTypes, ExtVec3fTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< BeamLinearMapping_mt< Rigid3fTypes, Vec3fTypes > >()
        .add< BeamLinearMapping_mt< Rigid3fTypes, ExtVec3fTypes > >()
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< BeamLinearMapping_mt< Rigid3dTypes, Vec3fTypes > >()
        .add< BeamLinearMapping_mt< Rigid3fTypes, Vec3dTypes > >()
#endif
#endif
        ;

#ifndef SOFA_FLOAT
template class BeamLinearMapping_mt< Rigid3dTypes, Vec3dTypes >;
template class BeamLinearMapping_mt< Rigid3dTypes, ExtVec3fTypes >;
#endif
#ifndef SOFA_DOUBLE
template class BeamLinearMapping_mt< Rigid3fTypes, Vec3fTypes >;
template class BeamLinearMapping_mt< Rigid3fTypes, ExtVec3fTypes >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class BeamLinearMapping_mt< Rigid3dTypes, Vec3fTypes >;
template class BeamLinearMapping_mt< Rigid3fTypes, Vec3dTypes >;
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

