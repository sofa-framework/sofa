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

int ImageFilterClass = core::RegisterObject("Filter an image")
        .add<ImageFilter<ImageUC,ImageUC    > >(true)
        .add<ImageFilter<ImageD ,ImageD     > >()

        .add<ImageFilter<ImageUC,ImageD    > >()

        .add<ImageFilter<ImageD,ImageUC    > >()
        .add<ImageFilter<ImageD,ImageB    > >()

#ifdef BUILD_ALL_IMAGE_TYPES
        .add<ImageFilter<ImageC ,ImageC     > >()
        .add<ImageFilter<ImageI ,ImageI     > >()
        .add<ImageFilter<ImageUI,ImageUI    > >()
        .add<ImageFilter<ImageS ,ImageS     > >()
        .add<ImageFilter<ImageUS,ImageUS    > >()
        .add<ImageFilter<ImageL ,ImageL     > >()
        .add<ImageFilter<ImageUL,ImageUL    > >()
        .add<ImageFilter<ImageF ,ImageF     > >()
        .add<ImageFilter<ImageB ,ImageB     > >()

        .add<ImageFilter<ImageC ,ImageD     > >()
        .add<ImageFilter<ImageI ,ImageD     > >()
        .add<ImageFilter<ImageUI,ImageD    > >()
        .add<ImageFilter<ImageS ,ImageD     > >()
        .add<ImageFilter<ImageUS,ImageD    > >()
        .add<ImageFilter<ImageL ,ImageD     > >()
        .add<ImageFilter<ImageUL,ImageD    > >()
        .add<ImageFilter<ImageF ,ImageD     > >()
        .add<ImageFilter<ImageB ,ImageD     > >()

        .add<ImageFilter<ImageUS,ImageUC    > >()
        .add<ImageFilter<ImageUC,ImageUS    > >()
        .add<ImageFilter<ImageUC,ImageB    > >()
        .add<ImageFilter<ImageUS,ImageB    > >()
        .add<ImageFilter<ImageS ,ImageB     > >()
        .add<ImageFilter<ImageS ,ImageUC     > >()
#endif
        ;

template class SOFA_IMAGE_API ImageFilter<ImageUC  ,ImageUC    >;
template class SOFA_IMAGE_API ImageFilter<ImageD   ,ImageD     >;

template class SOFA_IMAGE_API ImageFilter<ImageUC  ,ImageD    >;

template class SOFA_IMAGE_API ImageFilter<ImageD  ,ImageUC    >;
template class SOFA_IMAGE_API ImageFilter<ImageD  ,ImageB    >;

#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_API ImageFilter<ImageC   ,ImageC     >;
template class SOFA_IMAGE_API ImageFilter<ImageI   ,ImageI     >;
template class SOFA_IMAGE_API ImageFilter<ImageUI  ,ImageUI    >;
template class SOFA_IMAGE_API ImageFilter<ImageS   ,ImageS     >;
template class SOFA_IMAGE_API ImageFilter<ImageUS  ,ImageUS    >;
template class SOFA_IMAGE_API ImageFilter<ImageL   ,ImageL     >;
template class SOFA_IMAGE_API ImageFilter<ImageUL  ,ImageUL    >;
template class SOFA_IMAGE_API ImageFilter<ImageF   ,ImageF     >;
template class SOFA_IMAGE_API ImageFilter<ImageB   ,ImageB     >;

template class SOFA_IMAGE_API ImageFilter<ImageC   ,ImageD     >;
template class SOFA_IMAGE_API ImageFilter<ImageI   ,ImageD     >;
template class SOFA_IMAGE_API ImageFilter<ImageUI  ,ImageD    >;
template class SOFA_IMAGE_API ImageFilter<ImageS   ,ImageD     >;
template class SOFA_IMAGE_API ImageFilter<ImageUS  ,ImageD    >;
template class SOFA_IMAGE_API ImageFilter<ImageL   ,ImageD     >;
template class SOFA_IMAGE_API ImageFilter<ImageUL  ,ImageD    >;
template class SOFA_IMAGE_API ImageFilter<ImageF   ,ImageD     >;
template class SOFA_IMAGE_API ImageFilter<ImageB   ,ImageD     >;

template class SOFA_IMAGE_API ImageFilter<ImageUS   ,ImageUC     >;
template class SOFA_IMAGE_API ImageFilter<ImageUC   ,ImageUS     >;
template class SOFA_IMAGE_API ImageFilter<ImageUC   ,ImageB     >;
template class SOFA_IMAGE_API ImageFilter<ImageUS   ,ImageB     >;
template class SOFA_IMAGE_API ImageFilter<ImageS   ,ImageB     >;
template class SOFA_IMAGE_API ImageFilter<ImageS   ,ImageUC     >;
#endif


} //
} // namespace component

} // namespace sofa

