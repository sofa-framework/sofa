/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#define SelectLabelROI_CPP_

#include <sofa/component/engine/select/SelectLabelROI.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::engine::select
{

void registerSelectLabelROI(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Select a subset of labeled points or cells stored in (vector<svector<label>>) given certain labels.")
        .add< SelectLabelROI<unsigned int> >(true)
        .add< SelectLabelROI<unsigned char> >()
        .add< SelectLabelROI<unsigned short> >()
        .add< SelectLabelROI<int> >());
}

template class SOFA_COMPONENT_ENGINE_SELECT_API SelectLabelROI<unsigned int>;
template class SOFA_COMPONENT_ENGINE_SELECT_API SelectLabelROI<unsigned char>;
template class SOFA_COMPONENT_ENGINE_SELECT_API SelectLabelROI<unsigned short>;
template class SOFA_COMPONENT_ENGINE_SELECT_API SelectLabelROI<int>;

} //namespace sofa::component::engine::select
