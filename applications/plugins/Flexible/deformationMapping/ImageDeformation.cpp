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
#define SOFA_Flexible_ImageDeformation_CPP

#include <Flexible/config.h>
#include "ImageDeformation.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(ImageDeformation)

int ImageDeformationClass = core::RegisterObject("Deform an image based on an existing DeformationMapping")
        .add<ImageDeformation<ImageUC > >(true)
        .add<ImageDeformation<ImageD > >()
        .add<ImageDeformation<ImageB > >()

#ifdef BUILD_ALL_IMAGE_TYPES
        .add<ImageDeformation<ImageC > >()
        .add<ImageDeformation<ImageI > >()
        .add<ImageDeformation<ImageUI> >()
        .add<ImageDeformation<ImageS > >()
        .add<ImageDeformation<ImageUS> >()
        .add<ImageDeformation<ImageL > >()
        .add<ImageDeformation<ImageUL> >()
        .add<ImageDeformation<ImageF > >()
#endif
        ;

template class SOFA_Flexible_API ImageDeformation<ImageUC >;
template class SOFA_Flexible_API ImageDeformation<ImageD  >;
template class SOFA_Flexible_API ImageDeformation<ImageB  >;

#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_Flexible_API ImageDeformation<ImageC   >;
template class SOFA_Flexible_API ImageDeformation<ImageI   >;
template class SOFA_Flexible_API ImageDeformation<ImageUI  >;
template class SOFA_Flexible_API ImageDeformation<ImageS   >;
template class SOFA_Flexible_API ImageDeformation<ImageUS  >;
template class SOFA_Flexible_API ImageDeformation<ImageL   >;
template class SOFA_Flexible_API ImageDeformation<ImageUL  >;
template class SOFA_Flexible_API ImageDeformation<ImageF   >;
#endif


}
}
}
