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
#ifndef SOFA_GUI_QT_VIEWER_OGLMODELPOLICY_H
#define SOFA_GUI_QT_VIEWER_OGLMODELPOLICY_H

#include <sofa/gui/qt/SofaGuiQt.h>

#include <sofa/gui/qt/viewer/SofaViewer.h>
#include <sofa/gui/qt/viewer/VisualModelPolicy.h>

namespace sofa
{
namespace gui
{
namespace qt
{
namespace viewer
{
	
class SOFA_SOFAGUIQT_API OglModelPolicy : public VisualModelPolicy
{
protected:
    sofa::core::ObjectFactory::ClassEntry::SPtr classVisualModel;
    std::unique_ptr<sofa::core::visual::DrawTool> drawTool;
public:
    void load() override;
    void unload() override;
};


template < typename VisualModelPolicyType >
class SOFA_SOFAGUIQT_API CustomPolicySofaViewer : public VisualModelPolicyType, public sofa::gui::qt::viewer::SofaViewer
{
public:
    using VisualModelPolicyType::load;
    using VisualModelPolicyType::unload;
    CustomPolicySofaViewer() { load(); }
    ~CustomPolicySofaViewer() override { unload(); }
protected:
};

typedef SOFA_SOFAGUIQT_API CustomPolicySofaViewer< OglModelPolicy > OglModelSofaViewer;

} // namespace viewer
} // namespace qt
} // namespace gui
} // namespace sofa



#endif // SOFA_GUI_QT_VIEWER_OGLMODELPOLICY_H
