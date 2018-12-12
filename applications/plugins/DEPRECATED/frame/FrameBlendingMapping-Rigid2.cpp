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
#define SOFA_COMPONENT_MAPPING_FRAMEBLENDINGMAPPING_CPP

#include "FrameBlendingMapping.inl"
#include <sofa/core/ObjectFactory.h>



namespace sofa
{
namespace component
{

namespace mapping
{
using namespace defaulttype;
using namespace core;

//////////////////////////////////////////////////////////////////////////////////
////  Instanciations
//////////////////////////////////////////////////////////////////////////////////

// Register in the Factory
int FrameBlendingMappingClass_Rigid2 = core::RegisterObject("skin a model from a set of frames.")

//                                            .add< FrameBlendingMapping< Affine3dTypes, Vec3Types > >()
//                                            .add< FrameBlendingMapping< Affine3dTypes, Affine3dTypes > >()
//                                            .add< FrameBlendingMapping< Affine3dTypes, Rigid3Types > >()
//                                            .add< FrameBlendingMapping< Affine3dTypes, DeformationGradient331dTypes > >()
//                                            .add< FrameBlendingMapping< Affine3dTypes, DeformationGradient332dTypes > >()
//                                            .add< FrameBlendingMapping< Quadratic3dTypes, Vec3Types > >()
//                                            .add< FrameBlendingMapping< Quadratic3dTypes, Affine3dTypes > >()
//                                            .add< FrameBlendingMapping< Quadratic3dTypes, DeformationGradient331dTypes > >()
//                                            .add< FrameBlendingMapping< Quadratic3dTypes, DeformationGradient332dTypes > >()
//                                            .add< FrameBlendingMapping< Rigid3Types, Vec3Types > >()
//                                            .add< FrameBlendingMapping< Rigid3Types, Affine3dTypes > >()
//                                            .add< FrameBlendingMapping< Rigid3Types, Rigid3Types > >()
        .add< FrameBlendingMapping< Rigid3Types, DeformationGradient331dTypes > >()
        .add< FrameBlendingMapping< Rigid3Types, DeformationGradient332dTypes > >()


        ;

//            template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, Vec3Types >;
//            template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, Affine3dTypes >;
//            template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, Rigid3Types >;
//            template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, DeformationGradient331dTypes >;
//            template class SOFA_FRAME_API FrameBlendingMapping< Affine3dTypes, DeformationGradient332dTypes >;
//            template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, Vec3Types >;
//            template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, Affine3dTypes >;
//            template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, DeformationGradient331dTypes >;
//            template class SOFA_FRAME_API FrameBlendingMapping< Quadratic3dTypes, DeformationGradient332dTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Rigid3Types, Vec3Types >;
template class SOFA_FRAME_API FrameBlendingMapping< Rigid3Types, Affine3dTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Rigid3Types, Rigid3Types >;
template class SOFA_FRAME_API FrameBlendingMapping< Rigid3Types, DeformationGradient331dTypes >;
template class SOFA_FRAME_API FrameBlendingMapping< Rigid3Types, DeformationGradient332dTypes >;
 //SOFA_FLOAT
 //SOFA_FLOAT


} // namespace mapping

} // namespace component

} // namespace sofa

