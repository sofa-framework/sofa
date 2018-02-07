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
#define SOFA_IMAGE_VoronoiToMeshENGINE_CPP

#include "VoronoiToMeshEngine.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(VoronoiToMeshEngine)

int VoronoiToMeshEngineClass = core::RegisterObject("Generate flat faces between adjacent regions of an image")
        .add<VoronoiToMeshEngine<ImageUI> >(true)
        .add<VoronoiToMeshEngine<ImageUC> >()
#ifdef BUILD_ALL_IMAGE_TYPES
        .add<VoronoiToMeshEngine<ImageUS> >()
        .add<VoronoiToMeshEngine<ImageUL> >()
#endif
        ;

template class SOFA_IMAGE_API VoronoiToMeshEngine<ImageUI>;
template class SOFA_IMAGE_API VoronoiToMeshEngine<ImageUC>;
#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_API VoronoiToMeshEngine<ImageUS>;
template class SOFA_IMAGE_API VoronoiToMeshEngine<ImageUL>;
#endif

} //
} // namespace component

} // namespace sofa

