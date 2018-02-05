/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

#include <SofaGraphComponent/AttachBodyButtonSetting.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace configurationsetting
{

SOFA_DECL_CLASS(AttachBodyButtonSetting)
int AttachBodyButtonSettingClass = core::RegisterObject("Attach Body Button configuration")
        .add< AttachBodyButtonSetting >()
        .addAlias("AttachBodyButton")
        ;

AttachBodyButtonSetting::AttachBodyButtonSetting():
    stiffness(initData(&stiffness, (SReal)1000.0, "stiffness", "Stiffness of the spring to attach a particule"))
    , arrowSize(initData(&arrowSize, (SReal)0.0, "arrowSize", "Size of the drawn spring: if >0 an arrow will be drawn"))
    , showFactorSize(initData(&showFactorSize, (SReal)1.0, "showFactorSize", "Show factor size of the JointSpringForcefield  when interacting with rigids"))
{
}

}

}

}
