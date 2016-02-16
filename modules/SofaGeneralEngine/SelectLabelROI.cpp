/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SelectLabelROI_CPP_

#include "SelectLabelROI.h"
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

SOFA_DECL_CLASS(SelectLabelROI)

int SelectLabelROIClass = core::RegisterObject("Select a subset of labeled points or cells stored in (vector<svector<label>>) given certain labels")
        .add< SelectLabelROI<unsigned int> >(true)
        .add< SelectLabelROI<unsigned char> >()
        .add< SelectLabelROI<unsigned short> >()
        .add< SelectLabelROI<int> >()
        ;

template class SOFA_ENGINE_API SelectLabelROI<unsigned int>;
template class SOFA_ENGINE_API SelectLabelROI<unsigned char>;
template class SOFA_ENGINE_API SelectLabelROI<unsigned short>;
template class SOFA_ENGINE_API SelectLabelROI<int>;

} // namespace engine

} // namespace component

} // namespace sofa
