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
#pragma once

#include <sofa/gui/component/config.h>

#include <sofa/core/objectmodel/ConfigurationSetting.h>
#include <sofa/component/setting/MouseButtonSetting.h>

namespace sofa::gui::component
{

/*****
* This component modifies the mouse picking behavior in the GUI 
* and set it at the beginning to Lagrangian-based contraints
* It doesn't have any parameter because its only presence is sufficient.
*****/
class SOFA_GUI_COMPONENT_API ConstraintAttachButtonSetting: public sofa::component::setting::MouseButtonSetting
{
public:
    SOFA_CLASS(ConstraintAttachButtonSetting,MouseButtonSetting);
    std::string getOperationType() override {return "ConstraintAttach";}

};

} // namespace sofa::gui::component
