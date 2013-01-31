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
#define SOFA_IMAGE_BRANCHINGIMAGECONVERTER_CPP

#include "BranchingImageConverter.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(ImageToBranchingImageEngine)

int ImageToBranchingImageEngineClass = core::RegisterObject("ImageToBranchingImageEngine")
        .add<ImageToBranchingImageEngine<unsigned char> >(true)
        .add<ImageToBranchingImageEngine<double> >()
#ifdef BUILD_ALL_IMAGE_TYPES
        .add<ImageToBranchingImageEngine<char> >()
        .add<ImageToBranchingImageEngine<int> >()
        .add<ImageToBranchingImageEngine<unsigned int> >()
        .add<ImageToBranchingImageEngine<short> >()
        .add<ImageToBranchingImageEngine<unsigned short> >()
        .add<ImageToBranchingImageEngine<long> >()
        .add<ImageToBranchingImageEngine<unsigned long> >()
        .add<ImageToBranchingImageEngine<float> >()
        .add<ImageToBranchingImageEngine<bool> >()
#endif
        ;

template class SOFA_IMAGE_API ImageToBranchingImageEngine<unsigned char>;
template class SOFA_IMAGE_API ImageToBranchingImageEngine<double>;
#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_API ImageToBranchingImageEngine<char>;
template class SOFA_IMAGE_API ImageToBranchingImageEngine<int>;
template class SOFA_IMAGE_API ImageToBranchingImageEngine<unsigned int>;
template class SOFA_IMAGE_API ImageToBranchingImageEngine<short>;
template class SOFA_IMAGE_API ImageToBranchingImageEngine<unsigned short>;
template class SOFA_IMAGE_API ImageToBranchingImageEngine<long>;
template class SOFA_IMAGE_API ImageToBranchingImageEngine<unsigned long>;
template class SOFA_IMAGE_API ImageToBranchingImageEngine<float>;
template class SOFA_IMAGE_API ImageToBranchingImageEngine<bool>;
#endif



SOFA_DECL_CLASS(BranchingImageToImageEngine)

int BranchingImageToImageEngineClass = core::RegisterObject("BranchingImageToImageEngine")
        .add<BranchingImageToImageEngine<unsigned char> >(true)
        .add<BranchingImageToImageEngine<double> >()
#ifdef BUILD_ALL_IMAGE_TYPES
        .add<BranchingImageToImageEngine<char> >()
        .add<BranchingImageToImageEngine<int> >()
        .add<BranchingImageToImageEngine<unsigned int> >()
        .add<BranchingImageToImageEngine<short> >()
        .add<BranchingImageToImageEngine<unsigned short> >()
        .add<BranchingImageToImageEngine<long> >()
        .add<BranchingImageToImageEngine<unsigned long> >()
        .add<BranchingImageToImageEngine<float> >()
        .add<BranchingImageToImageEngine<bool> >()
#endif
        ;

template class SOFA_IMAGE_API BranchingImageToImageEngine<unsigned char>;
template class SOFA_IMAGE_API BranchingImageToImageEngine<double>;
#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_API BranchingImageToImageEngine<char>;
template class SOFA_IMAGE_API BranchingImageToImageEngine<int>;
template class SOFA_IMAGE_API BranchingImageToImageEngine<unsigned int>;
template class SOFA_IMAGE_API BranchingImageToImageEngine<short>;
template class SOFA_IMAGE_API BranchingImageToImageEngine<unsigned short>;
template class SOFA_IMAGE_API BranchingImageToImageEngine<long>;
template class SOFA_IMAGE_API BranchingImageToImageEngine<unsigned long>;
template class SOFA_IMAGE_API BranchingImageToImageEngine<float>;
template class SOFA_IMAGE_API BranchingImageToImageEngine<bool>;
#endif


} //
} // namespace component

} // namespace sofa

