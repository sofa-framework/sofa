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
#include <sofa/component/visual/VisualStyle.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/Node.h>
namespace sofa::component::visual
{

using namespace sofa::core::visual;
using namespace sofa::core::objectmodel;
using namespace sofa::simulation;

void registerVisualStyle(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Edit the visual style.\n Allowed values for displayFlags data are a combination of the following:\n\
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
        showWireframe hideWireframe").add<VisualStyle>());
}

VisualStyle::VisualStyle()
    : d_displayFlags(initData(&d_displayFlags, "displayFlags", "Display Flags"))
{
    d_displayFlags.setWidget("widget_displayFlags");

    displayFlags.setOriginalData(&d_displayFlags);
}

void VisualStyle::updateVisualFlags(VisualParams* vparams)
{
    backupFlags = vparams->displayFlags();
    vparams->displayFlags() = sofa::core::visual::merge_displayFlags(backupFlags, d_displayFlags.getValue());
}

void VisualStyle::applyBackupFlags(VisualParams* vparams)
{
    vparams->displayFlags() = backupFlags;
}


bool VisualStyle::insertInNode( sofa::core::objectmodel::BaseNode* node )
{
    node->addVisualStyle(this);
    return true;
}

bool VisualStyle::removeInNode( sofa::core::objectmodel::BaseNode* node )
{
    node->removeVisualStyle(this);
    return true;
}

helper::WriteAccessor<sofa::core::visual::DisplayFlags> addVisualStyle( simulation::Node::SPtr node )
{
    const VisualStyle::SPtr visualStyle = New<sofa::component::visual::VisualStyle>();
    node->addObject(visualStyle);
    return helper::getWriteAccessor(visualStyle->d_displayFlags);
}

} // namespace sofa::component::visual
