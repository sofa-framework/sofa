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

#include <PluginExample/config.h>

#include <sofa/core/BehaviorModel.h>
#include <SofaBaseVisual/VisualModelImpl.h>


namespace sofa::component::visualmodel
{

/**
 * This VisualModel does not do anything but have a dependency on a module (SofaBaseVisual).
 */
class SOFA_PLUGINEXAMPLE_API MyVisualModel : public sofa::component::visualmodel::VisualModelImpl
{

public:
    SOFA_CLASS(MyVisualModel, sofa::component::visualmodel::VisualModelImpl);

protected:
    MyVisualModel();
    virtual ~MyVisualModel() override;

public:
    void init() override;
    void reinit() override;

    void updateBuffers() {}

protected:
};


} // sofa::component::visualmodel

