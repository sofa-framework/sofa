/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#define SOFA_REGISTRATION_IntensityProfileCreator_CPP

#include "IntensityProfileCreator.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(IntensityProfileCreator)

int IntensityProfileCreatorClass = core::RegisterObject("Create reference intensity profiles from custom values")
        .add<IntensityProfileCreator<ImageUC> >(true)
        .add<IntensityProfileCreator<ImageUS> >()
        .add<IntensityProfileCreator<ImageS> >()
        .add<IntensityProfileCreator<ImageD> >()
        .add<IntensityProfileCreator<ImageB> >()
        ;

template class SOFA_REGISTRATION_API IntensityProfileCreator<ImageUC>;
template class SOFA_REGISTRATION_API IntensityProfileCreator<ImageUS>;
template class SOFA_REGISTRATION_API IntensityProfileCreator<ImageS>;
template class SOFA_REGISTRATION_API IntensityProfileCreator<ImageD>;
template class SOFA_REGISTRATION_API IntensityProfileCreator<ImageB>;



} //
} // namespace component

} // namespace sofa

