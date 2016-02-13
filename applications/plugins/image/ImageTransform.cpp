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
#define SOFA_IMAGE_IMAGETRANSFORM_CPP

#include "ImageTransform.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(ImageTransform)

int ImageTransformClass = core::RegisterObject("Read data from ImageContainer")
        .add<ImageTransform<ImageUC> >(true)
        .add<ImageTransform<ImageD> >()
#ifdef BUILD_ALL_IMAGE_TYPES
        .add<ImageTransform<ImageC> >()
        .add<ImageTransform<ImageI> >()
        .add<ImageTransform<ImageUI> >()
        .add<ImageTransform<ImageS> >()
        .add<ImageTransform<ImageUS> >()
        .add<ImageTransform<ImageL> >()
        .add<ImageTransform<ImageUL> >()
        .add<ImageTransform<ImageF> >()
        .add<ImageTransform<ImageB> >()
#endif
        ;

template class SOFA_IMAGE_API ImageTransform<ImageUC>;
template class SOFA_IMAGE_API ImageTransform<ImageD>;
#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_API ImageTransform<ImageC>;
template class SOFA_IMAGE_API ImageTransform<ImageI>;
template class SOFA_IMAGE_API ImageTransform<ImageUI>;
template class SOFA_IMAGE_API ImageTransform<ImageS>;
template class SOFA_IMAGE_API ImageTransform<ImageUS>;
template class SOFA_IMAGE_API ImageTransform<ImageL>;
template class SOFA_IMAGE_API ImageTransform<ImageUL>;
template class SOFA_IMAGE_API ImageTransform<ImageF>;
template class SOFA_IMAGE_API ImageTransform<ImageB>;
#endif




} //
} // namespace component
} // namespace sofa

