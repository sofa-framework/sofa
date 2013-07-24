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
        .add<ImageValuesFromPositions<BranchingImageD> >()
        .add<ImageValuesFromPositions<BranchingImageUC> >()
#ifdef BUILD_ALL_IMAGE_TYPES
        .add<ImageValuesFromPositions<BranchingImageC> >()
        .add<ImageValuesFromPositions<BranchingImageI> >()
        .add<ImageValuesFromPositions<BranchingImageUI> >()
        .add<ImageValuesFromPositions<BranchingImageS> >()
        .add<ImageValuesFromPositions<BranchingImageUS> >()
        .add<ImageValuesFromPositions<BranchingImageL> >()
        .add<ImageValuesFromPositions<BranchingImageUL> >()
        .add<ImageValuesFromPositions<BranchingImageF> >()
        .add<ImageValuesFromPositions<BranchingImageB> >()
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

template class SOFA_IMAGE_API ImageValuesFromPositions<BranchingImageD>;
template class SOFA_IMAGE_API ImageValuesFromPositions<BranchingImageUC>;
#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_API ImageValuesFromPositions<BranchingImageC>;
template class SOFA_IMAGE_API ImageValuesFromPositions<BranchingImageI>;
template class SOFA_IMAGE_API ImageValuesFromPositions<BranchingImageUI>;
template class SOFA_IMAGE_API ImageValuesFromPositions<BranchingImageS>;
template class SOFA_IMAGE_API ImageValuesFromPositions<BranchingImageUS>;
template class SOFA_IMAGE_API ImageValuesFromPositions<BranchingImageL>;
template class SOFA_IMAGE_API ImageValuesFromPositions<BranchingImageUL>;
template class SOFA_IMAGE_API ImageValuesFromPositions<BranchingImageF>;
template class SOFA_IMAGE_API ImageValuesFromPositions<BranchingImageB>;
#endif

} //
} // namespace component

} // namespace sofa

