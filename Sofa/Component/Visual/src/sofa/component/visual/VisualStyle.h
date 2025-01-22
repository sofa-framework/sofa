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
#include <sofa/component/visual/config.h>
#include <sofa/simulation/Node.h>

#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/visual/BaseVisualStyle.h>
#include <sofa/core/visual/Data[DisplayFlags].h>
#include <sofa/simulation/fwd.h>

#include <sofa/core/objectmodel/lifecycle/RenamedData.h>

namespace sofa::component::visual
{
/** \brief VisualStyle component controls the DisplayFlags state
* embedded in the VisualParams for the current subgraph.
* It merges the DisplayFlags conveyed by the VisualParams with
* its own DisplayFlags.
*
* example:
* <VisualStyle displayFlags="hideVisual showCollision showWireframe" />
*
* allowed values for displayFlags data are a combination of the following:
* showAll, hideAll,
*   showVisual, hideVisual,
*     showVisualModels, hideVisualModels,
*   showBehavior, hideBehavior,
*     showBehaviorModels, hideBehaviorModels,
*     showForceFields, hideForceFields,
*     showInteractionForceFields, hideInteractionForceFields
*   showMapping, hideMapping
*     showMappings, hideMappings
*     showMechanicalMappings, hideMechanicalMappings
*   showCollision, hideCollision
*      showCollisionModels, hideCollisionModels
*      showBoundingCollisionModels, hideBoundingCollisionModels
* showOptions hideOptions
*   showNormals hideNormals
*   showWireframe hideWireframe
*/
class SOFA_COMPONENT_VISUAL_API VisualStyle : public sofa::core::visual::BaseVisualStyle
{
public:
    SOFA_CLASS(VisualStyle,sofa::core::visual::BaseVisualStyle);

    typedef sofa::core::visual::VisualParams VisualParams;
    typedef sofa::core::visual::DisplayFlags DisplayFlags;
protected:
    VisualStyle();
public:
    void updateVisualFlags(VisualParams* ) override;
    void applyBackupFlags(VisualParams* ) override;

    bool insertInNode(sofa::core::objectmodel::BaseNode* node) override;
    bool removeInNode(sofa::core::objectmodel::BaseNode* node) override;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_VISUAL()
    sofa::core::objectmodel::lifecycle::RenamedData<DisplayFlags> displayFlags;

    Data<DisplayFlags> d_displayFlags; ///< Display Flags

protected:
    DisplayFlags backupFlags;
};

SOFA_COMPONENT_VISUAL_API helper::WriteAccessor<sofa::core::visual::DisplayFlags> addVisualStyle( simulation::NodeSPtr node );


} // namespace sofa::component::visual
