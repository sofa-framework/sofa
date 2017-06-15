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
#define SelectConnectedLabelsROI_CPP_

#include "SelectConnectedLabelsROI.h"
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

SOFA_DECL_CLASS(SelectConnectedLabelsROI)

int SelectConnectedLabelsROIClass = core::RegisterObject("Select a subset of points or cells labeled from different sources, that are connected given a list of connection pairs")
        .add< SelectConnectedLabelsROI<unsigned int> >(true)
        .add< SelectConnectedLabelsROI<unsigned char> >()
        .add< SelectConnectedLabelsROI<unsigned short> >()
        .add< SelectConnectedLabelsROI<int> >()
        ;

template class SOFA_GENERAL_ENGINE_API SelectConnectedLabelsROI<unsigned int>;
template class SOFA_GENERAL_ENGINE_API SelectConnectedLabelsROI<unsigned char>;
template class SOFA_GENERAL_ENGINE_API SelectConnectedLabelsROI<unsigned short>;
template class SOFA_GENERAL_ENGINE_API SelectConnectedLabelsROI<int>;

} // namespace engine

} // namespace component

} // namespace sofa
