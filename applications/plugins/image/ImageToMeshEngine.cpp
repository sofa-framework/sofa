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
#define SOFA_IMAGE_IMAGETOMESHENGINE_CPP

#include "ImageToMeshEngine.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(ImageToMeshEngine)

int ImageToMeshEngineClass = core::RegisterObject("Compute a mesh from a depth map image ")
        .add<ImageToMeshEngine<ImageUC> >(true)
        .add<ImageToMeshEngine<ImageD> >()
#ifdef BUILD_ALL_IMAGE_TYPES
        .add<ImageToMeshEngine<ImageC> >()
        .add<ImageToMeshEngine<ImageI> >()
        .add<ImageToMeshEngine<ImageUI> >()
        .add<ImageToMeshEngine<ImageS> >()
        .add<ImageToMeshEngine<ImageUS> >()
        .add<ImageToMeshEngine<ImageL> >()
        .add<ImageToMeshEngine<ImageUL> >()
        .add<ImageToMeshEngine<ImageF> >()
        .add<ImageToMeshEngine<ImageB> >()
#endif
        ;

template class SOFA_IMAGE_API ImageToMeshEngine<ImageUC>;
template class SOFA_IMAGE_API ImageToMeshEngine<ImageD>;
#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_API ImageToMeshEngine<ImageC>;
template class SOFA_IMAGE_API ImageToMeshEngine<ImageI>;
template class SOFA_IMAGE_API ImageToMeshEngine<ImageUI>;
template class SOFA_IMAGE_API ImageToMeshEngine<ImageS>;
template class SOFA_IMAGE_API ImageToMeshEngine<ImageUS>;
template class SOFA_IMAGE_API ImageToMeshEngine<ImageL>;
template class SOFA_IMAGE_API ImageToMeshEngine<ImageUL>;
template class SOFA_IMAGE_API ImageToMeshEngine<ImageF>;
template class SOFA_IMAGE_API ImageToMeshEngine<ImageB>;
#endif

} //
} // namespace component

} // namespace sofa

