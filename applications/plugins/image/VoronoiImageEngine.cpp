/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#define SOFA_IMAGE_VoronoiImageEngine_CPP

#include "VoronoiImageEngine.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(VoronoiImageEngine)

int VoronoiImageEngineClass = core::RegisterObject("Computes a voronoi image from a set of points inside an image mask")
        .add<VoronoiImageEngine<ImageB> >(true)
        .add<VoronoiImageEngine<ImageUC> >()
        .add<VoronoiImageEngine<ImageD> >()
#ifdef BUILD_ALL_IMAGE_TYPES
.add<VoronoiImageEngine<ImageC> >()
.add<VoronoiImageEngine<ImageI> >()
.add<VoronoiImageEngine<ImageUI> >()
.add<VoronoiImageEngine<ImageS> >()
.add<VoronoiImageEngine<ImageUS> >()
.add<VoronoiImageEngine<ImageL> >()
.add<VoronoiImageEngine<ImageUL> >()
.add<VoronoiImageEngine<ImageF> >()
#endif
        ;

template class SOFA_IMAGE_API VoronoiImageEngine<ImageB>;
template class SOFA_IMAGE_API VoronoiImageEngine<ImageUC>;
template class SOFA_IMAGE_API VoronoiImageEngine<ImageD>;
#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_API VoronoiImageEngine<ImageC>;
template class SOFA_IMAGE_API VoronoiImageEngine<ImageI>;
template class SOFA_IMAGE_API VoronoiImageEngine<ImageUI>;
template class SOFA_IMAGE_API VoronoiImageEngine<ImageS>;
template class SOFA_IMAGE_API VoronoiImageEngine<ImageUS>;
template class SOFA_IMAGE_API VoronoiImageEngine<ImageL>;
template class SOFA_IMAGE_API VoronoiImageEngine<ImageUL>;
template class SOFA_IMAGE_API VoronoiImageEngine<ImageF>;
#endif



} //
} // namespace component

} // namespace sofa

