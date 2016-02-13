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
#define SOFA_IMAGE_ImageValuesFromPositions_CPP

#include "ImageValuesFromPositions.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(ImageValuesFromPositions)

int ImageValuesFromPositionsClass = core::RegisterObject("Get image intensities at sample locations")
        .add<ImageValuesFromPositions<ImageD> >(true)
        .add<ImageValuesFromPositions<ImageUC> >()
#ifdef BUILD_ALL_IMAGE_TYPES
        .add<ImageValuesFromPositions<ImageC> >()
        .add<ImageValuesFromPositions<ImageI> >()
        .add<ImageValuesFromPositions<ImageUI> >()
        .add<ImageValuesFromPositions<ImageS> >()
        .add<ImageValuesFromPositions<ImageUS> >()
        .add<ImageValuesFromPositions<ImageL> >()
        .add<ImageValuesFromPositions<ImageUL> >()
        .add<ImageValuesFromPositions<ImageF> >()
        .add<ImageValuesFromPositions<ImageB> >()
#endif
        ;

template class SOFA_IMAGE_API ImageValuesFromPositions<ImageD>;
template class SOFA_IMAGE_API ImageValuesFromPositions<ImageUC>;
#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_API ImageValuesFromPositions<ImageC>;
template class SOFA_IMAGE_API ImageValuesFromPositions<ImageI>;
template class SOFA_IMAGE_API ImageValuesFromPositions<ImageUI>;
template class SOFA_IMAGE_API ImageValuesFromPositions<ImageS>;
template class SOFA_IMAGE_API ImageValuesFromPositions<ImageUS>;
template class SOFA_IMAGE_API ImageValuesFromPositions<ImageL>;
template class SOFA_IMAGE_API ImageValuesFromPositions<ImageUL>;
template class SOFA_IMAGE_API ImageValuesFromPositions<ImageF>;
template class SOFA_IMAGE_API ImageValuesFromPositions<ImageB>;
#endif


} //
} // namespace component

} // namespace sofa

