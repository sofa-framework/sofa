/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_IMAGE_COLLISIONTOCARVINGENGINE_CPP

#include "CollisionToCarvingEngine.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(CollisionToCarvingEngine)

int CollisionToCarvingEngineClass = core::RegisterObject("Filter an image")
        .add<CollisionToCarvingEngine<ImageUC,ImageUC    > >(true)
        .add<CollisionToCarvingEngine<ImageD ,ImageD     > >()

        .add<CollisionToCarvingEngine<ImageUC,ImageD    > >()

        .add<CollisionToCarvingEngine<ImageD,ImageUC    > >()
        .add<CollisionToCarvingEngine<ImageD,ImageB    > >()

#ifdef BUILD_ALL_IMAGE_TYPES
        .add<CollisionToCarvingEngine<ImageC ,ImageC     > >()
        .add<CollisionToCarvingEngine<ImageI ,ImageI     > >()
        .add<CollisionToCarvingEngine<ImageUI,ImageUI    > >()
        .add<CollisionToCarvingEngine<ImageS ,ImageS     > >()
        .add<CollisionToCarvingEngine<ImageUS,ImageUS    > >()
        .add<CollisionToCarvingEngine<ImageL ,ImageL     > >()
        .add<CollisionToCarvingEngine<ImageUL,ImageUL    > >()
        .add<CollisionToCarvingEngine<ImageF ,ImageF     > >()
        .add<CollisionToCarvingEngine<ImageB ,ImageB     > >()

        .add<CollisionToCarvingEngine<ImageC ,ImageD     > >()
        .add<CollisionToCarvingEngine<ImageI ,ImageD     > >()
        .add<CollisionToCarvingEngine<ImageUI,ImageD    > >()
        .add<CollisionToCarvingEngine<ImageS ,ImageD     > >()
        .add<CollisionToCarvingEngine<ImageUS,ImageD    > >()
        .add<CollisionToCarvingEngine<ImageL ,ImageD     > >()
        .add<CollisionToCarvingEngine<ImageUL,ImageD    > >()
        .add<CollisionToCarvingEngine<ImageF ,ImageD     > >()
        .add<CollisionToCarvingEngine<ImageB ,ImageD     > >()

        .add<CollisionToCarvingEngine<ImageUS,ImageUC    > >()
        .add<CollisionToCarvingEngine<ImageUC,ImageUS    > >()
        .add<CollisionToCarvingEngine<ImageUC,ImageB    > >()
        .add<CollisionToCarvingEngine<ImageUS,ImageB    > >()
#endif
        ;


template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageUC  ,ImageUC    >;
template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageD   ,ImageD     >;

template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageUC  ,ImageD    >;

template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageD  ,ImageUC    >;
template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageD  ,ImageB    >;

#ifdef BUILD_ALL_IMAGE_TYPES
template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageC   ,ImageC     >;
template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageI   ,ImageI     >;
template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageUI  ,ImageUI    >;
template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageS   ,ImageS     >;
template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageUS  ,ImageUS    >;
template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageL   ,ImageL     >;
template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageUL  ,ImageUL    >;
template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageF   ,ImageF     >;
template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageB   ,ImageB     >;

template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageC   ,ImageD     >;
template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageI   ,ImageD     >;
template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageUI  ,ImageD    >;
template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageS   ,ImageD     >;
template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageUS  ,ImageD    >;
template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageL   ,ImageD     >;
template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageUL  ,ImageD    >;
template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageF   ,ImageD     >;
template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageB   ,ImageD     >;

template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageUS   ,ImageUC     >;
template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageUC   ,ImageUS     >;
template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageUC   ,ImageB     >;
template class SOFA_IMAGE_API CollisionToCarvingEngine<ImageUS   ,ImageB     >;
#endif

} //
} // namespace component

} // namespace sofa

