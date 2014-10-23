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
#define SOFA_IMAGE_CONCATENATEIMAGES_CPP

#include "ConcatenateImages.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(ConcatenateImages)

int ConcatenateImagesClass = core::RegisterObject("Concatenate images")
        .add<ConcatenateImages<ImageUC> >(true)
        .add<ConcatenateImages<ImageD> >()
#ifdef BUILD_ALL_IMAGE_TYPES
        .add<ConcatenateImages<ImageC> >()
        .add<ConcatenateImages<ImageI> >()
        .add<ConcatenateImages<ImageUI> >()
        .add<ConcatenateImages<ImageS> >()
        .add<ConcatenateImages<ImageUS> >()
        .add<ConcatenateImages<ImageL> >()
        .add<ConcatenateImages<ImageUL> >()
        .add<ConcatenateImages<ImageF> >()
        .add<ConcatenateImages<ImageB> >()
#endif
        ;

template class SOFA_IMAGE_API ConcatenateImages<ImageUC>;
template class SOFA_IMAGE_API ConcatenateImages<ImageD>;
#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_API ConcatenateImages<ImageC>;
template class SOFA_IMAGE_API ConcatenateImages<ImageI>;
template class SOFA_IMAGE_API ConcatenateImages<ImageUI>;
template class SOFA_IMAGE_API ConcatenateImages<ImageS>;
template class SOFA_IMAGE_API ConcatenateImages<ImageUS>;
template class SOFA_IMAGE_API ConcatenateImages<ImageL>;
template class SOFA_IMAGE_API ConcatenateImages<ImageUL>;
template class SOFA_IMAGE_API ConcatenateImages<ImageF>;
template class SOFA_IMAGE_API ConcatenateImages<ImageB>;
#endif


} //
} // namespace component

} // namespace sofa

