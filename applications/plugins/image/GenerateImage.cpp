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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define SOFA_IMAGE_GenerateImage_CPP

#include "GenerateImage.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(GenerateImage)

int GenerateImageClass = core::RegisterObject("Create an image with custom dimensions")
        .add<GenerateImage<ImageUC> >(true)
        .add<GenerateImage<ImageD> >()
#ifdef BUILD_ALL_IMAGE_TYPES
        .add<GenerateImage<ImageC> >()
        .add<GenerateImage<ImageI> >()
        .add<GenerateImage<ImageUI> >()
        .add<GenerateImage<ImageS> >()
        .add<GenerateImage<ImageUS> >()
        .add<GenerateImage<ImageL> >()
        .add<GenerateImage<ImageUL> >()
        .add<GenerateImage<ImageF> >()
        .add<GenerateImage<ImageB> >()
#endif
        ;

template class SOFA_IMAGE_API GenerateImage<ImageUC>;
template class SOFA_IMAGE_API GenerateImage<ImageD>;
#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_API GenerateImage<ImageC>;
template class SOFA_IMAGE_API GenerateImage<ImageI>;
template class SOFA_IMAGE_API GenerateImage<ImageUI>;
template class SOFA_IMAGE_API GenerateImage<ImageS>;
template class SOFA_IMAGE_API GenerateImage<ImageUS>;
template class SOFA_IMAGE_API GenerateImage<ImageL>;
template class SOFA_IMAGE_API GenerateImage<ImageUL>;
template class SOFA_IMAGE_API GenerateImage<ImageF>;
template class SOFA_IMAGE_API GenerateImage<ImageB>;
#endif

} //
} // namespace component

} // namespace sofa

