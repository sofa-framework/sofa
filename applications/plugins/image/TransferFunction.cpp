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
#define SOFA_IMAGE_TRANSFERFUNCTION_CPP

#include "TransferFunction.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(TransferFunction)

int TransferFunctionClass = core::RegisterObject("Transforms pixel intensities")
        .add<TransferFunction<ImageUC,ImageUC    > >(true)
        .add<TransferFunction<ImageD ,ImageD     > >()
        .add<TransferFunction<ImageUC,ImageD    > >()
        .add<TransferFunction<ImageD,ImageUC    > >()
        .add<TransferFunction<ImageUC,ImageUI    > >()
        .add<TransferFunction<ImageUC,ImageUS    > >()
        .add<TransferFunction<ImageUS,ImageUC    > >()
        .add<TransferFunction<ImageUC,ImageB    > >()

#ifdef BUILD_ALL_IMAGE_TYPES
        .add<TransferFunction<ImageC ,ImageC     > >()
        .add<TransferFunction<ImageI ,ImageI     > >()
        .add<TransferFunction<ImageUI,ImageUI    > >()
        .add<TransferFunction<ImageS ,ImageS     > >()
        .add<TransferFunction<ImageUS,ImageUS    > >()
        .add<TransferFunction<ImageL ,ImageL     > >()
        .add<TransferFunction<ImageUL,ImageUL    > >()
        .add<TransferFunction<ImageF ,ImageF     > >()
        .add<TransferFunction<ImageB ,ImageB     > >()

        .add<TransferFunction<ImageC ,ImageD     > >()
        .add<TransferFunction<ImageI ,ImageD     > >()
        .add<TransferFunction<ImageUI,ImageD    > >()
        .add<TransferFunction<ImageS ,ImageD     > >()
        .add<TransferFunction<ImageUS,ImageD    > >()
        .add<TransferFunction<ImageL ,ImageD     > >()
        .add<TransferFunction<ImageUL,ImageD    > >()
        .add<TransferFunction<ImageF ,ImageD     > >()
        .add<TransferFunction<ImageB ,ImageD     > >()
#endif

.add<TransferFunction<BranchingImageUC,BranchingImageUC    > >(true)
.add<TransferFunction<BranchingImageD ,BranchingImageD     > >()
.add<TransferFunction<BranchingImageUC,BranchingImageD    > >()
.add<TransferFunction<BranchingImageD,BranchingImageUC    > >()
.add<TransferFunction<BranchingImageUC,BranchingImageUI    > >()
.add<TransferFunction<BranchingImageUC,BranchingImageUS    > >()
.add<TransferFunction<BranchingImageUS,BranchingImageUC    > >()
.add<TransferFunction<BranchingImageUC,BranchingImageB    > >()

#ifdef BUILD_ALL_IMAGE_TYPES
.add<TransferFunction<BranchingImageC ,BranchingImageC     > >()
.add<TransferFunction<BranchingImageI ,BranchingImageI     > >()
.add<TransferFunction<BranchingImageUI,BranchingImageUI    > >()
.add<TransferFunction<BranchingImageS ,BranchingImageS     > >()
.add<TransferFunction<BranchingImageUS,BranchingImageUS    > >()
.add<TransferFunction<BranchingImageL ,BranchingImageL     > >()
.add<TransferFunction<BranchingImageUL,BranchingImageUL    > >()
.add<TransferFunction<BranchingImageF ,BranchingImageF     > >()
.add<TransferFunction<BranchingImageB ,BranchingImageB     > >()

.add<TransferFunction<BranchingImageC ,BranchingImageD     > >()
.add<TransferFunction<BranchingImageI ,BranchingImageD     > >()
.add<TransferFunction<BranchingImageUI,BranchingImageD    > >()
.add<TransferFunction<BranchingImageS ,BranchingImageD     > >()
.add<TransferFunction<BranchingImageUS,BranchingImageD    > >()
.add<TransferFunction<BranchingImageL ,BranchingImageD     > >()
.add<TransferFunction<BranchingImageUL,BranchingImageD    > >()
.add<TransferFunction<BranchingImageF ,BranchingImageD     > >()
.add<TransferFunction<BranchingImageB ,BranchingImageD     > >()
#endif
        ;

template class SOFA_IMAGE_API TransferFunction<ImageUC  ,ImageUC    >;
template class SOFA_IMAGE_API TransferFunction<ImageD   ,ImageD     >;

template class SOFA_IMAGE_API TransferFunction<ImageUC  ,ImageD    >;

template class SOFA_IMAGE_API TransferFunction<ImageD  ,ImageUC    >;
template class SOFA_IMAGE_API TransferFunction<ImageUC   ,ImageUI     >;
template class SOFA_IMAGE_API TransferFunction<ImageUC   ,ImageUS     >;
template class SOFA_IMAGE_API TransferFunction<ImageUS   ,ImageUC     >;
template class SOFA_IMAGE_API TransferFunction<ImageUC   ,ImageB     >;

#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_API TransferFunction<ImageC   ,ImageC     >;
template class SOFA_IMAGE_API TransferFunction<ImageI   ,ImageI     >;
template class SOFA_IMAGE_API TransferFunction<ImageUI  ,ImageUI    >;
template class SOFA_IMAGE_API TransferFunction<ImageS   ,ImageS     >;
template class SOFA_IMAGE_API TransferFunction<ImageUS  ,ImageUS    >;
template class SOFA_IMAGE_API TransferFunction<ImageL   ,ImageL     >;
template class SOFA_IMAGE_API TransferFunction<ImageUL  ,ImageUL    >;
template class SOFA_IMAGE_API TransferFunction<ImageF   ,ImageF     >;
template class SOFA_IMAGE_API TransferFunction<ImageB   ,ImageB     >;

template class SOFA_IMAGE_API TransferFunction<ImageC   ,ImageD     >;
template class SOFA_IMAGE_API TransferFunction<ImageI   ,ImageD     >;
template class SOFA_IMAGE_API TransferFunction<ImageUI  ,ImageD    >;
template class SOFA_IMAGE_API TransferFunction<ImageS   ,ImageD     >;
template class SOFA_IMAGE_API TransferFunction<ImageUS  ,ImageD    >;
template class SOFA_IMAGE_API TransferFunction<ImageL   ,ImageD     >;
template class SOFA_IMAGE_API TransferFunction<ImageUL  ,ImageD    >;
template class SOFA_IMAGE_API TransferFunction<ImageF   ,ImageD     >;
template class SOFA_IMAGE_API TransferFunction<ImageB   ,ImageD     >;

#endif


template class SOFA_IMAGE_API TransferFunction<BranchingImageUC  ,BranchingImageUC    >;
template class SOFA_IMAGE_API TransferFunction<BranchingImageD   ,BranchingImageD     >;

template class SOFA_IMAGE_API TransferFunction<BranchingImageUC  ,BranchingImageD    >;

template class SOFA_IMAGE_API TransferFunction<BranchingImageD  ,BranchingImageUC    >;
template class SOFA_IMAGE_API TransferFunction<BranchingImageUC   ,BranchingImageUI     >;
template class SOFA_IMAGE_API TransferFunction<BranchingImageUC   ,BranchingImageUS     >;
template class SOFA_IMAGE_API TransferFunction<BranchingImageUS   ,BranchingImageUC     >;
template class SOFA_IMAGE_API TransferFunction<BranchingImageUC   ,BranchingImageB     >;

#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_API TransferFunction<BranchingImageC   ,BranchingImageC     >;
template class SOFA_IMAGE_API TransferFunction<BranchingImageI   ,BranchingImageI     >;
template class SOFA_IMAGE_API TransferFunction<BranchingImageUI  ,BranchingImageUI    >;
template class SOFA_IMAGE_API TransferFunction<BranchingImageS   ,BranchingImageS     >;
template class SOFA_IMAGE_API TransferFunction<BranchingImageUS  ,BranchingImageUS    >;
template class SOFA_IMAGE_API TransferFunction<BranchingImageL   ,BranchingImageL     >;
template class SOFA_IMAGE_API TransferFunction<BranchingImageUL  ,BranchingImageUL    >;
template class SOFA_IMAGE_API TransferFunction<BranchingImageF   ,BranchingImageF     >;
template class SOFA_IMAGE_API TransferFunction<BranchingImageB   ,BranchingImageB     >;

template class SOFA_IMAGE_API TransferFunction<BranchingImageC   ,BranchingImageD     >;
template class SOFA_IMAGE_API TransferFunction<BranchingImageI   ,BranchingImageD     >;
template class SOFA_IMAGE_API TransferFunction<BranchingImageUI  ,BranchingImageD    >;
template class SOFA_IMAGE_API TransferFunction<BranchingImageS   ,BranchingImageD     >;
template class SOFA_IMAGE_API TransferFunction<BranchingImageUS  ,BranchingImageD    >;
template class SOFA_IMAGE_API TransferFunction<BranchingImageL   ,BranchingImageD     >;
template class SOFA_IMAGE_API TransferFunction<BranchingImageUL  ,BranchingImageD    >;
template class SOFA_IMAGE_API TransferFunction<BranchingImageF   ,BranchingImageD     >;
template class SOFA_IMAGE_API TransferFunction<BranchingImageB   ,BranchingImageD     >;

#endif



} //
} // namespace component
} // namespace sofa

