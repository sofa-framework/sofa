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

#include <sofa/gui/component/AttachBodyButtonSetting.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::gui::component
{


void registerAttachBodyButtonSetting(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Attach Body Button configuration.")
        .add< AttachBodyButtonSetting >());
}

AttachBodyButtonSetting::AttachBodyButtonSetting():
    d_stiffness(initData(&d_stiffness, 1000.0_sreal, "stiffness", "Stiffness of the spring to attach a particule"))
    , d_arrowSize(initData(&d_arrowSize, 0.0_sreal, "arrowSize", "Size of the drawn spring: if >0 an arrow will be drawn"))
    , d_showFactorSize(initData(&d_showFactorSize, 1.0_sreal, "showFactorSize", "Show factor size of the JointSpringForcefield  when interacting with rigids"))
{
}

} // namespace sofa::gui::component
