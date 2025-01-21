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
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/type/RGBAColor.h>

#include <sofa/core/objectmodel/lifecycle/RenamedData.h>

namespace sofa::component::setting
{

///Class for the configuration of background settings.
class SOFA_COMPONENT_SETTING_API BackgroundSetting: public core::objectmodel::ConfigurationSetting
{
public:
    SOFA_CLASS(BackgroundSetting,core::objectmodel::ConfigurationSetting);  ///< Sofa macro to define typedef.

protected:
    BackgroundSetting();                                         ///< Default constructor

public:
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SETTING()
    sofa::core::objectmodel::lifecycle::RenamedData<sofa::type::RGBAColor> color;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SETTING()
    sofa::core::objectmodel::DataFileName image;                 ///< Image to be used as background of the viewer.


    Data<sofa::type::RGBAColor> d_color; ///< Color of the background
    sofa::core::objectmodel::DataFileName d_image;                 ///< Image to be used as background of the viewer.

};

} // namespace sofa::component::setting
