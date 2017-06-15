/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_VISUALMODEL_VISUALSTYLE_H
#define SOFA_COMPONENT_VISUALMODEL_VISUALSTYLE_H
#include "config.h"

#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/visual/DisplayFlags.h>
#include <sofa/simulation/Node.h>

namespace sofa
{
namespace component
{
namespace visualmodel
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
class SOFA_BASE_VISUAL_API VisualStyle : public sofa::core::visual::VisualModel
{
public:
    SOFA_CLASS(VisualStyle,sofa::core::visual::VisualModel);

    typedef sofa::core::visual::VisualParams VisualParams;
    typedef sofa::core::visual::DisplayFlags DisplayFlags;
protected:
    VisualStyle();
public:
    void fwdDraw(VisualParams* );
    void bwdDraw(VisualParams* );

    Data<DisplayFlags> displayFlags;

protected:
    DisplayFlags backupFlags;
};

SOFA_BASE_VISUAL_API helper::WriteAccessor<sofa::core::visual::DisplayFlags> addVisualStyle( simulation::Node::SPtr node );


} // visual

} // component

} // sofa

#endif // SOFA_COMPONENT_VISUALMODEL_VISUALSTYLE_H
