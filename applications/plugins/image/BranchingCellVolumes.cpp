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
#define SOFA_IMAGE_BranchingCellVolumes_CPP

#include "BranchingCellVolumes.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(BranchingCellVolumes)

int BranchingCellVolumesClass = core::RegisterObject("Returns volumes of branching voxels, given a fine image of superimposed offsets")
        .add<BranchingCellVolumes<ImageUI,BranchingImageUC> >(true)
        .add<BranchingCellVolumes<ImageUI,BranchingImageB> >()
        .add<BranchingCellVolumes<ImageUI,BranchingImageD> >()
        .add<BranchingCellVolumes<ImageUI,BranchingImageUS> >()
        ;

template class SOFA_IMAGE_API BranchingCellVolumes<ImageUI,BranchingImageUC>;
template class SOFA_IMAGE_API BranchingCellVolumes<ImageUI,BranchingImageB>;
template class SOFA_IMAGE_API BranchingCellVolumes<ImageUI,BranchingImageD>;
template class SOFA_IMAGE_API BranchingCellVolumes<ImageUI,BranchingImageUS>;

} //
} // namespace component
} // namespace sofa

