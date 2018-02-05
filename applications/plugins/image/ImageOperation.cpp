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
#define SOFA_IMAGE_IMAGEOPERATION_CPP

#include "ImageOperation.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(ImageOperation)

int ImageOperationClass = core::RegisterObject("This class computes an image as an operation between two images")
        .add<ImageOperation<ImageUC> >(true)
        .add<ImageOperation<ImageD> >()
#ifdef BUILD_ALL_IMAGE_TYPES
        .add<ImageOperation<ImageC> >()
        .add<ImageOperation<ImageI> >()
        .add<ImageOperation<ImageUI> >()
        .add<ImageOperation<ImageS> >()
        .add<ImageOperation<ImageUS> >()
        .add<ImageOperation<ImageL> >()
        .add<ImageOperation<ImageUL> >()
        .add<ImageOperation<ImageF> >()
        .add<ImageOperation<ImageB> >()
#endif
        ;

template class SOFA_IMAGE_API ImageOperation<ImageUC>;
template class SOFA_IMAGE_API ImageOperation<ImageD>;
#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_API ImageOperation<ImageC>;
template class SOFA_IMAGE_API ImageOperation<ImageI>;
template class SOFA_IMAGE_API ImageOperation<ImageUI>;
template class SOFA_IMAGE_API ImageOperation<ImageS>;
template class SOFA_IMAGE_API ImageOperation<ImageUS>;
template class SOFA_IMAGE_API ImageOperation<ImageL>;
template class SOFA_IMAGE_API ImageOperation<ImageUL>;
template class SOFA_IMAGE_API ImageOperation<ImageF>;
template class SOFA_IMAGE_API ImageOperation<ImageB>;
#endif
} //
} // namespace component
} // namespace sofa

