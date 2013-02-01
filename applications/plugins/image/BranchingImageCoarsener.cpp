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
#define SOFA_IMAGE_BRANCHINGIMAGECOARSENER_CPP

#include "BranchingImageCoarsener.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(BranchingImageCoarsener)

int BranchingImageCoarsenerClass = core::RegisterObject("BranchingImageCoarsener")
        .add<BranchingImageCoarsener<unsigned char> >(true)
        .add<BranchingImageCoarsener<double> >()
#ifdef BUILD_ALL_IMAGE_TYPES
        .add<BranchingImageCoarsener<char> >()
        .add<BranchingImageCoarsener<int> >()
        .add<BranchingImageCoarsener<unsigned int> >()
        .add<BranchingImageCoarsener<short> >()
        .add<BranchingImageCoarsener<unsigned short> >()
        .add<BranchingImageCoarsener<long> >()
        .add<BranchingImageCoarsener<unsigned long> >()
        .add<BranchingImageCoarsener<float> >()
        .add<BranchingImageCoarsener<bool> >()
#endif
        ;

template class SOFA_IMAGE_API BranchingImageCoarsener<unsigned char>;
template class SOFA_IMAGE_API BranchingImageCoarsener<double>;
#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_API BranchingImageCoarsener<char>;
template class SOFA_IMAGE_API BranchingImageCoarsener<int>;
template class SOFA_IMAGE_API BranchingImageCoarsener<unsigned int>;
template class SOFA_IMAGE_API BranchingImageCoarsener<short>;
template class SOFA_IMAGE_API BranchingImageCoarsener<unsigned short>;
template class SOFA_IMAGE_API BranchingImageCoarsener<long>;
template class SOFA_IMAGE_API BranchingImageCoarsener<unsigned long>;
template class SOFA_IMAGE_API BranchingImageCoarsener<float>;
template class SOFA_IMAGE_API BranchingImageCoarsener<bool>;
#endif


} //
} // namespace component

} // namespace sofa

