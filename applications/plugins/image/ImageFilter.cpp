/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#define SOFA_IMAGE_IMAGEFILTER_CPP

#include "ImageFilter.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(ImageFilter)

int ImageFilterClass = core::RegisterObject("Compute a filtered image")
        .add<ImageFilter<ImageC> >()
        .add<ImageFilter<ImageUC> >(true)
        .add<ImageFilter<ImageI> >()
        .add<ImageFilter<ImageUI> >()
        .add<ImageFilter<ImageS> >()
        .add<ImageFilter<ImageUS> >()
        .add<ImageFilter<ImageL> >()
        .add<ImageFilter<ImageUL> >()
        .add<ImageFilter<ImageF> >()
        .add<ImageFilter<ImageD> >()
        .add<ImageFilter<ImageB> >()
        ;

template class SOFA_IMAGE_API ImageFilter<ImageC>;
template class SOFA_IMAGE_API ImageFilter<ImageUC>;
template class SOFA_IMAGE_API ImageFilter<ImageI>;
template class SOFA_IMAGE_API ImageFilter<ImageUI>;
template class SOFA_IMAGE_API ImageFilter<ImageS>;
template class SOFA_IMAGE_API ImageFilter<ImageUS>;
template class SOFA_IMAGE_API ImageFilter<ImageL>;
template class SOFA_IMAGE_API ImageFilter<ImageUL>;
template class SOFA_IMAGE_API ImageFilter<ImageF>;
template class SOFA_IMAGE_API ImageFilter<ImageD>;
template class SOFA_IMAGE_API ImageFilter<ImageB>;


} //
} // namespace component

} // namespace sofa

