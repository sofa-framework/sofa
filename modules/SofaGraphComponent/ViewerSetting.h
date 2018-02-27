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
#ifndef SOFA_COMPONENT_CONFIGURATIONSETTING_VIEWER_H
#define SOFA_COMPONENT_CONFIGURATIONSETTING_VIEWER_H
#include "config.h"

#include <sofa/core/objectmodel/ConfigurationSetting.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/OptionsGroup.h>

namespace sofa
{

namespace component
{

namespace configurationsetting
{

///Class for the configuration of viewer settings.
class SOFA_GRAPH_COMPONENT_API ViewerSetting: public sofa::core::objectmodel::ConfigurationSetting
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

    Data<sofa::defaulttype::Vec<2,int> > resolution;                           ///< Screen resolution (width, height).
    Data<bool> fullscreen;                                  ///< True if viewer should be fullscreen.
    Data<sofa::helper::OptionsGroup> cameraMode;                          ///< Camera mode.
                                                            /**<    \arg Perspective.
                                                             *      \arg Orthographic.
                                                             */
    Data<sofa::helper::OptionsGroup> objectPickingMethod;                 ///< Picking Method.
                                                            /**<    \arg Ray casting.
                                                             *      \arg Selection Buffer.
                                                             */
};

}

}

}
#endif
