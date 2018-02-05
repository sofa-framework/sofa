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
#define SOFA_COMPONENT_MAPPING_FRAMEBLENDINGMAPPING_CPP

#include "FrameBlendingMapping.inl"
#include <sofa/core/ObjectFactory.h>



namespace sofa
{
namespace component
{

namespace mapping
{
SOFA_DECL_CLASS(FrameBlendingMapping_Quadratic3d_2);

using namespace defaulttype;
using namespace core;

//////////////////////////////////////////////////////////////////////////////////
////  Instanciations
//////////////////////////////////////////////////////////////////////////////////

// Register in the Factory
int FrameBlendingMappingClass_Quadratic3d_2 = core::RegisterObject("skin a model from a set of frames.")

#ifndef SOFA_FLOAT
//                                            .add< FrameBlendingMapping< Affine3dTypes, Vec3dTypes > >()
//                                            .add< FrameBlendingMapping< Affine3dTypes, Affine3dTypes > >()
//                                            .add< FrameBlendingMapping< Affine3dTypes, Rigid3dTypes > >()
//                                            .add< FrameBlendingMapping< Affine3dTypes, DeformationGradient331dTypes > >()
//                                            .add< FrameBlendingMapping< Affine3dTypes, DeformationGradient332dTypes > >()
//                                            .add< FrameBlendingMapping< Quadratic3dTypes, Vec3dTypes > >()
//                                            .add< FrameBlendingMapping< Quadratic3dTypes, Affine3dTypes > >()
        .add< FrameBlendingMapping< Quadratic3dTypes, DeformationGradient331dTypes > >()
        .add< FrameBlendingMapping< Quadratic3dTypes, DeformationGradient332dTypes > >()
//                                            .add< FrameBlendingMapping< Rigid3dTypes, Vec3dTypes > >()
//                                            .add< FrameBlendingMapping< Rigid3dTypes, Affine3dTypes > >()
//                                            .add< FrameBlendingMapping< Rigid3dTypes, Rigid3dTypes > >()
//                                            .add< FrameBlendingMapping< Rigid3dTypes, DeformationGradient331dTypes > >()
//                                            .add< FrameBlendingMapping< Rigid3dTypes, DeformationGradient332dTypes > >()
#endif
#ifndef SOFA_DOUBLE
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
//                                            .add< FrameBlendingMapping< Affine3dTypes, ExtVec3fTypes > >()
        .add< FrameBlendingMapping< Quadratic3dTypes, ExtVec3fTypes > >()
//                                            .add< FrameBlendingMapping< Rigid3dTypes, ExtVec3fTypes > >()
#endif
#endif
        ;

#ifndef SOFA_FLOAT
//            template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, Vec3dTypes >;
//            template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, Affine3dTypes >;
//            template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, Rigid3dTypes >;
//            template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, DeformationGradient331dTypes >;
//            template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, DeformationGradient332dTypes >;
//            template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, Vec3dTypes >;
//            template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, Affine3dTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, DeformationGradient331dTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, DeformationGradient332dTypes >;
//            template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, Vec3dTypes >;
//            template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, Affine3dTypes >;
//            template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, Rigid3dTypes >;
//            template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, DeformationGradient331dTypes >;
//            template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, DeformationGradient332dTypes >;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
#endif //SOFA_DOUBLE
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
//            template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, ExtVec3fTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, ExtVec3fTypes >;
//            template class SOFA_FRAME_API FrameBlendingMapping< Rigid3dTypes, ExtVec3fTypes >;
#endif //SOFA_DOUBLE
#endif //SOFA_FLOAT


} // namespace mapping

} // namespace component

} // namespace sofa

