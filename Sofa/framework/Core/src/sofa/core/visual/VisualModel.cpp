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
#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa::core::visual
{

VisualModel::VisualModel():
    d_enable(initData(&d_enable, true,  "enable", "Display the object or not"))
{}

void VisualModel::drawVisual(const VisualParams* vparams)
{
    // don't draw if specified not to do so in the user interface
    if (!vparams->displayFlags().getShowVisualModels())
        return;

    // don't draw if this component is specifically configured to be disabled
    if (!d_enable.getValue())
        return;

    // don't draw if the component is not in valid state
    if( d_componentState.getValue() == sofa::core::objectmodel::ComponentState::Invalid )
        return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    if (vparams->displayFlags().getShowWireFrame())
    {
        vparams->drawTool()->setPolygonMode(0, true);
    }

    doDrawVisual(vparams);
}

bool VisualModel::insertInNode( objectmodel::BaseNode* node )
{
    node->addVisualModel(this);
    Inherit1::insertInNode(node);
    return true;
}

bool VisualModel::removeInNode( objectmodel::BaseNode* node )
{
    node->removeVisualModel(this);
    Inherit1::removeInNode(node);
    return true;
}
} // namespace sofa::core::visual

