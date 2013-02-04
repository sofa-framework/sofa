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

SOFA_DECL_CLASS(ImageToBranchingImageConverter)

int ImageToBranchingImageConverterClass = core::RegisterObject("ImageToBranchingImageConverter")
        .add<ImageToBranchingImageConverter<unsigned char> >(true)
        .add<ImageToBranchingImageConverter<double> >()
#ifdef BUILD_ALL_IMAGE_TYPES
        .add<ImageToBranchingImageConverter<char> >()
        .add<ImageToBranchingImageConverter<int> >()
        .add<ImageToBranchingImageConverter<unsigned int> >()
        .add<ImageToBranchingImageConverter<short> >()
        .add<ImageToBranchingImageConverter<unsigned short> >()
        .add<ImageToBranchingImageConverter<long> >()
        .add<ImageToBranchingImageConverter<unsigned long> >()
        .add<ImageToBranchingImageConverter<float> >()
        .add<ImageToBranchingImageConverter<bool> >()
#endif
        ;

template class SOFA_IMAGE_API ImageToBranchingImageConverter<unsigned char>;
template class SOFA_IMAGE_API ImageToBranchingImageConverter<double>;
#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_API ImageToBranchingImageConverter<char>;
template class SOFA_IMAGE_API ImageToBranchingImageConverter<int>;
template class SOFA_IMAGE_API ImageToBranchingImageConverter<unsigned int>;
template class SOFA_IMAGE_API ImageToBranchingImageConverter<short>;
template class SOFA_IMAGE_API ImageToBranchingImageConverter<unsigned short>;
template class SOFA_IMAGE_API ImageToBranchingImageConverter<long>;
template class SOFA_IMAGE_API ImageToBranchingImageConverter<unsigned long>;
template class SOFA_IMAGE_API ImageToBranchingImageConverter<float>;
template class SOFA_IMAGE_API ImageToBranchingImageConverter<bool>;
#endif



SOFA_DECL_CLASS(BranchingImageToImageConverter)

int BranchingImageToImageConverterClass = core::RegisterObject("BranchingImageToImageConverter")
        .add<BranchingImageToImageConverter<unsigned char> >(true)
        .add<BranchingImageToImageConverter<double> >()
#ifdef BUILD_ALL_IMAGE_TYPES
        .add<BranchingImageToImageConverter<char> >()
        .add<BranchingImageToImageConverter<int> >()
        .add<BranchingImageToImageConverter<unsigned int> >()
        .add<BranchingImageToImageConverter<short> >()
        .add<BranchingImageToImageConverter<unsigned short> >()
        .add<BranchingImageToImageConverter<long> >()
        .add<BranchingImageToImageConverter<unsigned long> >()
        .add<BranchingImageToImageConverter<float> >()
        .add<BranchingImageToImageConverter<bool> >()
#endif
        ;

template class SOFA_IMAGE_API BranchingImageToImageConverter<unsigned char>;
template class SOFA_IMAGE_API BranchingImageToImageConverter<double>;
#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_API BranchingImageToImageConverter<char>;
template class SOFA_IMAGE_API BranchingImageToImageConverter<int>;
template class SOFA_IMAGE_API BranchingImageToImageConverter<unsigned int>;
template class SOFA_IMAGE_API BranchingImageToImageConverter<short>;
template class SOFA_IMAGE_API BranchingImageToImageConverter<unsigned short>;
template class SOFA_IMAGE_API BranchingImageToImageConverter<long>;
template class SOFA_IMAGE_API BranchingImageToImageConverter<unsigned long>;
template class SOFA_IMAGE_API BranchingImageToImageConverter<float>;
template class SOFA_IMAGE_API BranchingImageToImageConverter<bool>;
#endif


} //
} // namespace component

} // namespace sofa

