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
#define SOFA_IMAGE_MERGEBRANCHINGIMAGES_CPP

#include "MergeBranchingImages.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(MergeBranchingImages)

int MergeBranchingImagesClass = core::RegisterObject("Merge branching images")
        .add<MergeBranchingImages<BranchingImageUC> >(true)
        .add<MergeBranchingImages<BranchingImageD> >()
#ifdef BUILD_ALL_IMAGE_TYPES
        .add<MergeBranchingImages<BranchingImageC> >()
        .add<MergeBranchingImages<BranchingImageI> >()
        .add<MergeBranchingImages<BranchingImageUI> >()
        .add<MergeBranchingImages<BranchingImageS> >()
        .add<MergeBranchingImages<BranchingImageUS> >()
        .add<MergeBranchingImages<BranchingImageL> >()
        .add<MergeBranchingImages<BranchingImageUL> >()
        .add<MergeBranchingImages<BranchingImageF> >()
        .add<MergeBranchingImages<BranchingImageB> >()
#endif
        ;

template class SOFA_IMAGE_API MergeBranchingImages<BranchingImageUC>;
template class SOFA_IMAGE_API MergeBranchingImages<BranchingImageD>;
#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_API MergeBranchingImages<BranchingImageC>;
template class SOFA_IMAGE_API MergeBranchingImages<BranchingImageI>;
template class SOFA_IMAGE_API MergeBranchingImages<BranchingImageUI>;
template class SOFA_IMAGE_API MergeBranchingImages<BranchingImageS>;
template class SOFA_IMAGE_API MergeBranchingImages<BranchingImageUS>;
template class SOFA_IMAGE_API MergeBranchingImages<BranchingImageL>;
template class SOFA_IMAGE_API MergeBranchingImages<BranchingImageUL>;
template class SOFA_IMAGE_API MergeBranchingImages<BranchingImageF>;
template class SOFA_IMAGE_API MergeBranchingImages<BranchingImageB>;
#endif


} //
} // namespace component

} // namespace sofa

