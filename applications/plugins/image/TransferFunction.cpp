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
        .add<TransferFunction<ImageUC,ImageF    > >()

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
        ;

template class SOFA_IMAGE_API TransferFunction<ImageUC  ,ImageUC    >;
template class SOFA_IMAGE_API TransferFunction<ImageD   ,ImageD     >;

template class SOFA_IMAGE_API TransferFunction<ImageUC  ,ImageD    >;

template class SOFA_IMAGE_API TransferFunction<ImageD  ,ImageUC    >;
template class SOFA_IMAGE_API TransferFunction<ImageUC   ,ImageUI     >;
template class SOFA_IMAGE_API TransferFunction<ImageUC   ,ImageUS     >;
template class SOFA_IMAGE_API TransferFunction<ImageUS   ,ImageUC     >;
template class SOFA_IMAGE_API TransferFunction<ImageUC   ,ImageB     >;
template class SOFA_IMAGE_API TransferFunction<ImageUC   ,ImageF     >;

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





} //
} // namespace component
} // namespace sofa

