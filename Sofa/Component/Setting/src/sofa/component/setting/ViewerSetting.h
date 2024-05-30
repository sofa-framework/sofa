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
#include <sofa/type/Vec.h>
#include <sofa/helper/OptionsGroup.h>

namespace sofa::component::setting
{

///Class for the configuration of viewer settings.
class SOFA_COMPONENT_SETTING_API ViewerSetting: public sofa::core::objectmodel::ConfigurationSetting
{
public:
    SOFA_CLASS(ViewerSetting,ConfigurationSetting); ///< Sofa macro to define typedef.
protected:

    /**
     * @brief Default constructor.
     *
     * By default :
     *  - @ref resolution is set to 800x600.
     *  - @ref fullscreen is set to false.
     *  - @ref cameraMode is set to projective.
     *  - @ref objectPickingMethod is set to ray casting.
     */
    ViewerSetting();
public:

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SETTING()
    Data<sofa::type::Vec<2,int> > resolution;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SETTING()
    Data<bool> fullscreen;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SETTING()
    Data<sofa::helper::OptionsGroup> cameraMode;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SETTING()
    Data<sofa::helper::OptionsGroup> objectPickingMethod;

    Data<sofa::type::Vec<2,int> > d_resolution;                           ///< Screen resolution (width, height).
    Data<bool> d_fullscreen;                                  ///< True if viewer should be fullscreen.
    Data<sofa::helper::OptionsGroup> d_cameraMode;                          ///< Camera mode.
                                                            /**<    \arg Perspective.
                                                             *      \arg Orthographic.
                                                             */
    Data<sofa::helper::OptionsGroup> d_objectPickingMethod;                 ///< Picking Method.
                                                            /**<    \arg Ray casting.
                                                             *      \arg Selection Buffer.
                                                             */
};

} // namespace sofa::component::setting
