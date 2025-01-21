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

#include <sofa/component/setting/config.h>

#include <sofa/core/objectmodel/ConfigurationSetting.h>
#include <sofa/helper/OptionsGroup.h>

#include <sofa/core/objectmodel/lifecycle/RenamedData.h>

namespace sofa::component::setting
{

class SOFA_COMPONENT_SETTING_API MouseButtonSetting: public core::objectmodel::ConfigurationSetting
{
public:
    SOFA_CLASS(MouseButtonSetting,core::objectmodel::ConfigurationSetting);
protected:
    MouseButtonSetting();
public:
    virtual std::string getOperationType()=0;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SETTING()
    sofa::core::objectmodel::lifecycle::RenamedData<sofa::helper::OptionsGroup> button;

    core::objectmodel::Data<sofa::helper::OptionsGroup> d_button; ///< Mouse button used

};

} // namespace sofa::component::setting
