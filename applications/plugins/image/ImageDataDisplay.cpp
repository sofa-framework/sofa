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
#define SOFA_IMAGE_ImageDataDisplay_CPP

#include "ImageDataDisplay.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(ImageDataDisplay)

int ImageDataDisplayClass = core::RegisterObject("Store custom data in an image. A template input image with non zero voxels (where data will be stored) has to be provided")
        .add<ImageDataDisplay<ImageUC, ImageD> >(true)
        .add<ImageDataDisplay<ImageD, ImageD> >()
        .add<ImageDataDisplay<ImageB, ImageD> >()
        ;

template class SOFA_IMAGE_API ImageDataDisplay<ImageUC, ImageD>;
template class SOFA_IMAGE_API ImageDataDisplay<ImageD, ImageD>;
template class SOFA_IMAGE_API ImageDataDisplay<ImageB, ImageD>;



} //
} // namespace component

} // namespace sofa

