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

class SOFA_GUI_COMPONENT_API FixPickedParticleButtonSetting: public sofa::component::setting::MouseButtonSetting
{
public:
    SOFA_CLASS(FixPickedParticleButtonSetting,MouseButtonSetting);
protected:
    FixPickedParticleButtonSetting();
public:
    std::string getOperationType() override {return "Fix";}
    Data<SReal> stiffness; ///< Stiffness of the spring to fix a particule

};

} // namespace sofa::gui::component
