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
#include <SofaBaseVisual/VisualStyle.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/UpdateContextVisitor.h>

namespace sofa
{
namespace component
{
namespace visualmodel
{

using namespace sofa::core::visual;
using namespace sofa::core::objectmodel;
using namespace sofa::simulation;

int VisualStyleClass = core::RegisterObject("Edit the visual style.\n Allowed values for displayFlags data are a combination of the following:\n\
showAll, hideAll,\n\
    showVisual, hideVisual,\n\
        showVisualModels, hideVisualModels,\n\
    showBehavior, hideBehavior,\n\
        showBehaviorModels, hideBehaviorModels,\n\
        showForceFields, hideForceFields,\n\
        showInteractionForceFields, hideInteractionForceFields\n\
    showMapping, hideMapping\n\
        showMappings, hideMappings\n\
        showMechanicalMappings, hideMechanicalMappings\n\
    showCollision, hideCollision\n\
        showCollisionModels, hideCollisionModels\n\
        showBoundingCollisionModels, hideBoundingCollisionModels\n\
    showOptions hideOptions\n\
        showRendering hideRendering\n\
        showNormals hideNormals\n\
        showWireframe hideWireframe").add<VisualStyle>();

VisualStyle::VisualStyle()
    :displayFlags(initData(&displayFlags,"displayFlags","Display Flags"))
{
    displayFlags.setWidget("widget_displayFlags");
//    displayFlags.setGroup("Display Flags");
}

void VisualStyle::fwdDraw(VisualParams* vparams)
{
    backupFlags = vparams->displayFlags();
    vparams->displayFlags() = sofa::core::visual::merge_displayFlags(backupFlags, displayFlags.getValue(vparams));
}

void VisualStyle::bwdDraw(VisualParams* vparams)
{
    vparams->displayFlags() = backupFlags;
}

helper::WriteAccessor<sofa::core::visual::DisplayFlags> addVisualStyle( simulation::Node::SPtr node )
{
    VisualStyle::SPtr visualStyle = New<sofa::component::visualmodel::VisualStyle>();
    node->addObject(visualStyle);
//    return visualStyle->displayFlags.setValue(displayFlags);
    return helper::write(visualStyle->displayFlags);

}


}
}
}

