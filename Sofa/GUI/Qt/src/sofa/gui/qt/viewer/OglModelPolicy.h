/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once
#include <sofa/gui/qt/config.h>

#include <sofa/gui/qt/viewer/SofaViewer.h>
#include <sofa/gui/qt/viewer/VisualModelPolicy.h>

#include <sofa/helper/visual/DrawTool.h>

namespace sofa::gui::qt::viewer
{
	
class SOFA_GUI_QT_API OglModelPolicy : public VisualModelPolicy
{
protected:
    sofa::core::ObjectFactory::ClassEntry::SPtr classVisualModel;
    std::unique_ptr<sofa::helper::visual::DrawTool> drawTool;
public:
    void load() override;
    void unload() override;
};


template < typename VisualModelPolicyType >
class SOFA_GUI_QT_API CustomPolicySofaViewer : public VisualModelPolicyType, public sofa::gui::qt::viewer::SofaViewer
{
public:
    using VisualModelPolicyType::load;
    using VisualModelPolicyType::unload;
    CustomPolicySofaViewer() { load(); }
    ~CustomPolicySofaViewer() override { unload(); }
protected:
};

typedef CustomPolicySofaViewer< OglModelPolicy > OglModelSofaViewer;

} // namespace sofa::gui::qt::viewer
