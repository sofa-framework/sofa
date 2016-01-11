/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_IMAGE_IMAGEVIEWER_CPP

#include "ImageViewer.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace misc
{
using namespace sofa::defaulttype;

SOFA_DECL_CLASS (ImageViewer)
// Register in the Factory

int ImageViewerClass = core::RegisterObject ( "Image viewer" )
        .add<ImageViewer<ImageUC> >(true)
        .add<ImageViewer<ImageD> >()
#ifdef BUILD_ALL_IMAGE_TYPES
        .add<ImageViewer<ImageC> >()
        .add<ImageViewer<ImageI> >()
        .add<ImageViewer<ImageUI> >()
        .add<ImageViewer<ImageS> >()
        .add<ImageViewer<ImageUS> >()
        .add<ImageViewer<ImageL> >()
        .add<ImageViewer<ImageUL> >()
        .add<ImageViewer<ImageF> >()
        .add<ImageViewer<ImageB> >()
#endif
        ;

template class SOFA_IMAGE_API ImageViewer<ImageUC>;
template class SOFA_IMAGE_API ImageViewer<ImageD>;
#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_API ImageViewer<ImageC>;
template class SOFA_IMAGE_API ImageViewer<ImageI>;
template class SOFA_IMAGE_API ImageViewer<ImageUI>;
template class SOFA_IMAGE_API ImageViewer<ImageS>;
template class SOFA_IMAGE_API ImageViewer<ImageUS>;
template class SOFA_IMAGE_API ImageViewer<ImageL>;
template class SOFA_IMAGE_API ImageViewer<ImageUL>;
template class SOFA_IMAGE_API ImageViewer<ImageF>;
template class SOFA_IMAGE_API ImageViewer<ImageB>;
#endif

} // namespace misc

} // namespace component

} // namespace sofa

