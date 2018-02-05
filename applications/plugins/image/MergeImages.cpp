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
#define SOFA_IMAGE_MERGEIMAGES_CPP

#include "MergeImages.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(MergeImages)

int MergeImagesClass = core::RegisterObject("Merge images")
        .add<MergeImages<ImageUC> >(true)
        .add<MergeImages<ImageD> >()
#ifdef BUILD_ALL_IMAGE_TYPES
        .add<MergeImages<ImageC> >()
        .add<MergeImages<ImageI> >()
        .add<MergeImages<ImageUI> >()
        .add<MergeImages<ImageS> >()
        .add<MergeImages<ImageUS> >()
        .add<MergeImages<ImageL> >()
        .add<MergeImages<ImageUL> >()
        .add<MergeImages<ImageF> >()
        .add<MergeImages<ImageB> >()
#endif
        ;

template class SOFA_IMAGE_API MergeImages<ImageUC>;
template class SOFA_IMAGE_API MergeImages<ImageD>;
#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_API MergeImages<ImageC>;
template class SOFA_IMAGE_API MergeImages<ImageI>;
template class SOFA_IMAGE_API MergeImages<ImageUI>;
template class SOFA_IMAGE_API MergeImages<ImageS>;
template class SOFA_IMAGE_API MergeImages<ImageUS>;
template class SOFA_IMAGE_API MergeImages<ImageL>;
template class SOFA_IMAGE_API MergeImages<ImageUL>;
template class SOFA_IMAGE_API MergeImages<ImageF>;
template class SOFA_IMAGE_API MergeImages<ImageB>;
#endif


} //
} // namespace component

} // namespace sofa

