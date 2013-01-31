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
#define IMAGE_BRANCHINGIMAGECONTAINER_CPP

#include "BranchingImageContainer.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace container
{

using namespace defaulttype;


SOFA_DECL_CLASS (BranchingImageContainer);
// Register in the Factory

int BranchingImageContainerClass = core::RegisterObject ( "Branching Image Container" )
        .add<BranchingImageContainer<BranchingImageUC> >(true)
        .add<BranchingImageContainer<BranchingImageD> >()
#ifdef BUILD_ALL_IMAGE_TYPES
        .add<BranchingImageContainer<BranchingImageC> >()
        .add<BranchingImageContainer<BranchingImageI> >()
        .add<BranchingImageContainer<BranchingImageUI> >()
        .add<BranchingImageContainer<BranchingImageS> >()
        .add<BranchingImageContainer<BranchingImageUS> >()
        .add<BranchingImageContainer<BranchingImageL> >()
        .add<BranchingImageContainer<BranchingImageUL> >()
        .add<BranchingImageContainer<BranchingImageF> >()
        .add<BranchingImageContainer<BranchingImageB> >()
#endif
        ;

template class SOFA_IMAGE_API BranchingImageContainer<BranchingImageUC>;
template class SOFA_IMAGE_API BranchingImageContainer<BranchingImageD>;
#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_API BranchingImageContainer<BranchingImageC>;
template class SOFA_IMAGE_API BranchingImageContainer<BranchingImageI>;
template class SOFA_IMAGE_API BranchingImageContainer<BranchingImageUI>;
template class SOFA_IMAGE_API BranchingImageContainer<BranchingImageS>;
template class SOFA_IMAGE_API BranchingImageContainer<BranchingImageUS>;
template class SOFA_IMAGE_API BranchingImageContainer<BranchingImageL>;
template class SOFA_IMAGE_API BranchingImageContainer<BranchingImageUL>;
template class SOFA_IMAGE_API BranchingImageContainer<BranchingImageF>;
template class SOFA_IMAGE_API BranchingImageContainer<BranchingImageB>;
#endif

} // namespace container

} // namespace component

} // namespace sofa
