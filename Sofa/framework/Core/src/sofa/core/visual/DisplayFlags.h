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

#include <sofa/core/config.h>
#include <sofa/type/vector.h>

#include <sofa/core/visual/FlagTreeItem.h>

namespace sofa::core::visual
{

/** \brief Class which describes the display of components in a hierarchical fashion
* DisplayFlags are conveyed by the VisualParams, and therefore are accessible in a
* read only fashion inside a Component draw method.
* A component can tell if it should draw something on the display by looking at the current state
* of the displayFlags through the VisualParams parameter.
* DisplayFlags are embeddable inside a Data and can therefore read/write their state
* from a stream.
*
* root
* |--all
* |  |--visual
* |  |  |--visualmodels
* |  |--behavior
* |  |  |--behaviormodels
* |  |  |--forcefields
* |  |  |--interactionforcefields
* |  |--collision
* |  |  |--collisionmodels
* |  |  |--boundingcollisionmodels
* |  |--mapping
* |  |  |--visualmappings
* |  |  |--mechanicalmappings
* |--options
* |  |--advancedrendering
* |  |--wireframe
* |  |--normals
*/
class SOFA_CORE_API DisplayFlags
{
public:
    DisplayFlags();
    DisplayFlags(const DisplayFlags& );
    DisplayFlags& operator=(const DisplayFlags& );
    tristate getShowAll() const { return m_showAll.state(); }
    tristate getShowVisual() const { return m_showVisual.state(); }
    tristate getShowVisualModels() const { return m_showVisualModels.state(); }
    tristate getShowBehavior() const { return m_showBehavior.state(); }
    tristate getShowBehaviorModels() const { return m_showBehaviorModels.state(); }
    tristate getShowForceFields() const { return m_showForceFields.state(); }
    tristate getShowInteractionForceFields() const { return m_showInteractionForceFields.state(); }
    tristate getShowCollision() const { return m_showCollision.state(); }
    tristate getShowCollisionModels() const { return m_showCollisionModels.state(); }
    tristate getShowBoundingCollisionModels() const { return m_showBoundingCollisionModels.state(); }
    tristate getShowDetectionOutputs() const { return m_showDetectionOutputs.state(); }
    tristate getShowMapping() const { return m_showMapping.state(); }
    tristate getShowMappings() const { return m_showVisualMappings.state(); }
    tristate getShowMechanicalMappings() const { return m_showMechanicalMappings.state(); }
    tristate getShowOptions() const { return m_showOptions.state(); }
    tristate getShowAdvancedRendering() const { return m_showAdvancedRendering.state(); }
    tristate getShowWireFrame() const { return m_showWireframe.state(); }
    tristate getShowNormals() const { return m_showNormals.state(); }

    DisplayFlags& setShowAll(tristate v=true) { m_showAll.setValue(v); return (*this); }
    DisplayFlags& setShowVisual(tristate v=true ) { m_showVisual.setValue(v); return (*this); }
    DisplayFlags& setShowVisualModels(tristate v=true)  { m_showVisualModels.setValue(v); return (*this); }
    DisplayFlags& setShowBehavior(tristate v=true) { m_showBehavior.setValue(v); return (*this); }
    DisplayFlags& setShowBehaviorModels(tristate v=true)  { m_showBehaviorModels.setValue(v); return (*this); }
    DisplayFlags& setShowForceFields(tristate v=true)  { m_showForceFields.setValue(v); return (*this); }
    DisplayFlags& setShowInteractionForceFields(tristate v=true) { m_showInteractionForceFields.setValue(v); return (*this); }
    DisplayFlags& setShowCollision(tristate v=true ) { m_showCollisionModels.setValue(v); return (*this); }
    DisplayFlags& setShowCollisionModels(tristate v=true) { m_showCollisionModels.setValue(v); return (*this); }
    DisplayFlags& setShowBoundingCollisionModels(tristate v=true) { m_showBoundingCollisionModels.setValue(v); return (*this); }
    DisplayFlags& setShowDetectionOutputs(tristate v=true) { m_showDetectionOutputs.setValue(v); return (*this); }
    DisplayFlags& setShowMapping(tristate v=true) { m_showMapping.setValue(v); return (*this); }
    DisplayFlags& setShowMappings(tristate v=true) { m_showVisualMappings.setValue(v); return (*this); }
    DisplayFlags& setShowMechanicalMappings(tristate v=true) { m_showMechanicalMappings.setValue(v); return (*this); }
    DisplayFlags& setShowOptions(tristate v=true) { m_showOptions.setValue(v); return (*this); }
    DisplayFlags& setShowAdvancedRendering(tristate v=true) { m_showAdvancedRendering.setValue(v); return (*this); }
    DisplayFlags& setShowWireFrame(tristate v=true) { m_showWireframe.setValue(v); return (*this); }
    DisplayFlags& setShowNormals(tristate v=true) { m_showNormals.setValue(v); return (*this); }
    SOFA_CORE_API friend std::ostream& operator<< ( std::ostream& os, const DisplayFlags& flags );
    SOFA_CORE_API friend std::istream& operator>> ( std::istream& in, DisplayFlags& flags );

    std::istream& read(std::istream& in,
                   const std::function<void(std::string)>& unknownFlagFunction,
                   const std::function<void(std::string, std::string)>& incorrectLetterCaseFunction);

    bool isNeutral() const;

    friend SOFA_CORE_API DisplayFlags merge_displayFlags(const DisplayFlags& previous, const DisplayFlags& current);
    friend SOFA_CORE_API DisplayFlags difference_displayFlags(const DisplayFlags& parent, const DisplayFlags& child);

    sofa::type::vector<std::string> getAllFlagsLabels() const;

protected:
    FlagTreeItem m_root;

    FlagTreeItem m_showAll;

    FlagTreeItem m_showVisual;
    FlagTreeItem m_showVisualModels;

    FlagTreeItem m_showBehavior;
    FlagTreeItem m_showBehaviorModels;
    FlagTreeItem m_showForceFields;
    FlagTreeItem m_showInteractionForceFields;

    FlagTreeItem m_showCollision;
    FlagTreeItem m_showCollisionModels;
    FlagTreeItem m_showBoundingCollisionModels;
    FlagTreeItem m_showDetectionOutputs;

    FlagTreeItem m_showMapping;
    FlagTreeItem m_showVisualMappings;
    FlagTreeItem m_showMechanicalMappings;

    FlagTreeItem m_showOptions;

    FlagTreeItem m_showAdvancedRendering;
    FlagTreeItem m_showWireframe;
    FlagTreeItem m_showNormals;
};

SOFA_CORE_API DisplayFlags merge_displayFlags(const DisplayFlags& previous, const DisplayFlags& current);
SOFA_CORE_API DisplayFlags difference_displayFlags(const DisplayFlags& parent, const DisplayFlags& child);

}
