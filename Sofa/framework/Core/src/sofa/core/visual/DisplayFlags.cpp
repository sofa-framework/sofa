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
#include <sofa/core/visual/DisplayFlags.h>
#include <sofa/helper/DiffLib.h>
#include <sofa/helper/logging/Messaging.h>

namespace sofa::core::visual
{

DisplayFlags::DisplayFlags():
    m_root(FlagTreeItem("showRoot","hideRoot",nullptr)),
    m_showAll(FlagTreeItem("showAll","hideAll",&m_root)),
    m_showVisual(FlagTreeItem("showVisual","hideVisual",&m_showAll)),
    m_showVisualModels(FlagTreeItem("showVisualModels","hideVisualModels",&m_showVisual)),
    m_showBehavior(FlagTreeItem("showBehavior","hideBehavior",&m_showAll)),
    m_showBehaviorModels(FlagTreeItem("showBehaviorModels","hideBehaviorModels",&m_showBehavior)),
    m_showForceFields(FlagTreeItem("showForceFields","hideForceFields",&m_showBehavior)),
    m_showInteractionForceFields(FlagTreeItem("showInteractionForceFields","hideInteractionForceFields",&m_showBehavior)),
    m_showCollision(FlagTreeItem("showCollision","hideCollision",&m_showAll)),
    m_showCollisionModels(FlagTreeItem("showCollisionModels","hideCollisionModels",&m_showCollision)),
    m_showBoundingCollisionModels(FlagTreeItem("showBoundingCollisionModels","hideBoundingCollisionModels",&m_showCollision)),
    m_showDetectionOutputs(FlagTreeItem("showDetectionOutputs","hideDetectionOutputs",&m_showCollision)),
    m_showMapping(FlagTreeItem("showMapping","hideMapping",&m_showAll)),
    m_showVisualMappings(FlagTreeItem("showMappings","hideMappings",&m_showMapping)),
    m_showMechanicalMappings(FlagTreeItem("showMechanicalMappings","hideMechanicalMappings",&m_showMapping)),
    m_showOptions(FlagTreeItem("showOptions","hideOptions",&m_root)),
    m_showAdvancedRendering(FlagTreeItem("showAdvancedRendering","hideAdvancedRendering",&m_showOptions)),
    m_showWireframe(FlagTreeItem("showWireframe","hideWireframe",&m_showOptions)),
    m_showNormals(FlagTreeItem("showNormals","hideNormals",&m_showOptions))
{
    m_showVisualModels.setValue(tristate::neutral_value);
    m_showBehaviorModels.setValue(tristate::neutral_value);
    m_showForceFields.setValue(tristate::neutral_value);
    m_showInteractionForceFields.setValue(tristate::neutral_value);
    m_showCollisionModels.setValue(tristate::neutral_value);
    m_showBoundingCollisionModels.setValue(tristate::neutral_value);
    m_showDetectionOutputs.setValue(tristate::neutral_value);
    m_showVisualMappings.setValue(tristate::neutral_value);
    m_showMechanicalMappings.setValue(tristate::neutral_value);
    m_showAdvancedRendering.setValue(tristate::neutral_value);
    m_showWireframe.setValue(tristate::neutral_value);
    m_showNormals.setValue(tristate::neutral_value);

    m_showAdvancedRendering.addAliasShow("showRendering");
    m_showAdvancedRendering.addAliasHide("hideRendering");
}

DisplayFlags::DisplayFlags(const DisplayFlags & other):
    m_root(FlagTreeItem("showRoot","hideRoot",nullptr)),
    m_showAll(FlagTreeItem("showAll","hideAll",&m_root)),
    m_showVisual(FlagTreeItem("showVisual","hideVisual",&m_showAll)),
    m_showVisualModels(FlagTreeItem("showVisualModels","hideVisualModels",&m_showVisual)),
    m_showBehavior(FlagTreeItem("showBehavior","hideBehavior",&m_showAll)),
    m_showBehaviorModels(FlagTreeItem("showBehaviorModels","hideBehaviorModels",&m_showBehavior)),
    m_showForceFields(FlagTreeItem("showForceFields","hideForceFields",&m_showBehavior)),
    m_showInteractionForceFields(FlagTreeItem("showInteractionForceFields","hideInteractionForceFields",&m_showBehavior)),
    m_showCollision(FlagTreeItem("showCollision","hideCollision",&m_showAll)),
    m_showCollisionModels(FlagTreeItem("showCollisionModels","hideCollisionModels",&m_showCollision)),
    m_showBoundingCollisionModels(FlagTreeItem("showBoundingCollisionModels","hideBoundingCollisionModels",&m_showCollision)),
    m_showDetectionOutputs(FlagTreeItem("showDetectionOutputs","hideDetectionOutputs",&m_showCollision)),
    m_showMapping(FlagTreeItem("showMapping","hideMapping",&m_showAll)),
    m_showVisualMappings(FlagTreeItem("showMappings","hideMappings",&m_showMapping)),
    m_showMechanicalMappings(FlagTreeItem("showMechanicalMappings","hideMechanicalMappings",&m_showMapping)),
    m_showOptions(FlagTreeItem("showOptions","hideOptions",&m_root)),
    m_showAdvancedRendering(FlagTreeItem("showAdvancedRendering","hideAdvancedRendering",&m_showOptions)),
    m_showWireframe(FlagTreeItem("showWireframe","hideWireframe",&m_showOptions)),
    m_showNormals(FlagTreeItem("showNormals","hideNormals",&m_showOptions))
{
    m_showVisualModels.setValue(other.m_showVisualModels.state());
    m_showBehaviorModels.setValue(other.m_showBehaviorModels.state());
    m_showForceFields.setValue(other.m_showForceFields.state());
    m_showInteractionForceFields.setValue(other.m_showInteractionForceFields.state());
    m_showCollisionModels.setValue(other.m_showCollisionModels.state());
    m_showBoundingCollisionModels.setValue(other.m_showBoundingCollisionModels.state());
    m_showDetectionOutputs.setValue(other.m_showDetectionOutputs.state());
    m_showVisualMappings.setValue(other.m_showVisualMappings.state());
    m_showMechanicalMappings.setValue(other.m_showMechanicalMappings.state());
    m_showAdvancedRendering.setValue(other.m_showAdvancedRendering.state());
    m_showWireframe.setValue(other.m_showWireframe.state());
    m_showNormals.setValue(other.m_showNormals.state());

    m_showAdvancedRendering.addAliasShow("showRendering");
    m_showAdvancedRendering.addAliasHide("hideRendering");
}

DisplayFlags& DisplayFlags::operator =(const DisplayFlags& other)
{
    if( this != &other)
    {
        m_showVisualModels.setValue(other.m_showVisualModels.state());
        m_showBehaviorModels.setValue(other.m_showBehaviorModels.state());
        m_showForceFields.setValue(other.m_showForceFields.state());
        m_showInteractionForceFields.setValue(other.m_showInteractionForceFields.state());
        m_showCollisionModels.setValue(other.m_showCollisionModels.state());
        m_showBoundingCollisionModels.setValue(other.m_showBoundingCollisionModels.state());
        m_showDetectionOutputs.setValue(other.m_showDetectionOutputs.state());
        m_showVisualMappings.setValue(other.m_showVisualMappings.state());
        m_showMechanicalMappings.setValue(other.m_showMechanicalMappings.state());
        m_showAdvancedRendering.setValue(other.m_showAdvancedRendering.state());
        m_showWireframe.setValue(other.m_showWireframe.state());
        m_showNormals.setValue(other.m_showNormals.state());
    }
    return *this;
}

std::istream& DisplayFlags::read(std::istream& in,
    const std::function<void(std::string)>& unknownFlagFunction,
    const std::function<void(std::string, std::string)>& incorrectLetterCaseFunction)
{
    return m_root.read(in,
        unknownFlagFunction,
        incorrectLetterCaseFunction);
}

bool DisplayFlags::isNeutral() const
{
    return m_showVisualModels.state().state == tristate::neutral_value
           && m_showBehaviorModels.state().state == tristate::neutral_value
           && m_showForceFields.state().state  == tristate::neutral_value
           && m_showInteractionForceFields.state().state == tristate::neutral_value
           && m_showBoundingCollisionModels.state().state == tristate::neutral_value
           && m_showDetectionOutputs.state().state == tristate::neutral_value
           && m_showCollisionModels.state().state == tristate::neutral_value
           && m_showVisualMappings.state().state == tristate::neutral_value
           && m_showMechanicalMappings.state().state == tristate::neutral_value
           && m_showAdvancedRendering.state().state == tristate::neutral_value
           && m_showWireframe.state().state == tristate::neutral_value
           && m_showNormals.state().state == tristate::neutral_value
        ;
}

sofa::type::vector<std::string> DisplayFlags::getAllFlagsLabels() const
{
    sofa::type::vector<std::string> labels;
    m_root.getLabels(labels);
    return labels;
}

DisplayFlags merge_displayFlags(const DisplayFlags &previous, const DisplayFlags &current)
{
    DisplayFlags merge;
    merge.m_showVisualModels.setValue( merge_tristate(previous.m_showVisualModels.state(),current.m_showVisualModels.state()) );
    merge.m_showBehaviorModels.setValue( merge_tristate(previous.m_showBehaviorModels.state(),current.m_showBehaviorModels.state()) );
    merge.m_showForceFields.setValue( merge_tristate(previous.m_showForceFields.state(),current.m_showForceFields.state()) );
    merge.m_showInteractionForceFields.setValue( merge_tristate(previous.m_showInteractionForceFields.state(),current.m_showInteractionForceFields.state()) );
    merge.m_showCollisionModels.setValue( merge_tristate(previous.m_showCollisionModels.state(),current.m_showCollisionModels.state()) );
    merge.m_showBoundingCollisionModels.setValue( merge_tristate(previous.m_showBoundingCollisionModels.state(),current.m_showBoundingCollisionModels.state()) );
    merge.m_showDetectionOutputs.setValue( merge_tristate(previous.m_showDetectionOutputs.state(),current.m_showDetectionOutputs.state()) );
    merge.m_showVisualMappings.setValue( merge_tristate(previous.m_showVisualMappings.state(),current.m_showVisualMappings.state()) );
    merge.m_showMechanicalMappings.setValue( merge_tristate(previous.m_showMechanicalMappings.state(),current.m_showMechanicalMappings.state()) );
    merge.m_showAdvancedRendering.setValue( merge_tristate(previous.m_showAdvancedRendering.state(),current.m_showAdvancedRendering.state()) );
    merge.m_showWireframe.setValue( merge_tristate(previous.m_showWireframe.state(),current.m_showWireframe.state()) );
    merge.m_showNormals.setValue( merge_tristate(previous.m_showNormals.state(),current.m_showNormals.state()) );
    return merge;
}

DisplayFlags difference_displayFlags(const DisplayFlags& previous, const DisplayFlags& current)
{
    DisplayFlags difference;
    difference.m_showVisualModels.setValue( difference_tristate(previous.m_showVisualModels.state(),current.m_showVisualModels.state()) );
    difference.m_showBehaviorModels.setValue( difference_tristate(previous.m_showBehaviorModels.state(),current.m_showBehaviorModels.state()) );
    difference.m_showForceFields.setValue( difference_tristate(previous.m_showForceFields.state(),current.m_showForceFields.state()) );
    difference.m_showInteractionForceFields.setValue( difference_tristate(previous.m_showInteractionForceFields.state(),current.m_showInteractionForceFields.state()) );
    difference.m_showCollisionModels.setValue( difference_tristate(previous.m_showCollisionModels.state(),current.m_showCollisionModels.state()) );
    difference.m_showBoundingCollisionModels.setValue( difference_tristate(previous.m_showBoundingCollisionModels.state(),current.m_showBoundingCollisionModels.state()) );
    difference.m_showDetectionOutputs.setValue( difference_tristate(previous.m_showDetectionOutputs.state(),current.m_showDetectionOutputs.state()) );
    difference.m_showVisualMappings.setValue( difference_tristate(previous.m_showVisualMappings.state(),current.m_showVisualMappings.state()) );
    difference.m_showMechanicalMappings.setValue( difference_tristate(previous.m_showMechanicalMappings.state(),current.m_showMechanicalMappings.state()) );
    difference.m_showAdvancedRendering.setValue( difference_tristate(previous.m_showAdvancedRendering.state(),current.m_showAdvancedRendering.state()) );
    difference.m_showWireframe.setValue( difference_tristate(previous.m_showWireframe.state(),current.m_showWireframe.state()) );
    difference.m_showNormals.setValue( difference_tristate(previous.m_showNormals.state(),current.m_showNormals.state()) );
    return difference;
}

std::ostream& operator<< ( std::ostream& os, const DisplayFlags& flags )
{
    return flags.m_root.write(os);
}

std::istream& operator>> ( std::istream& in, DisplayFlags& flags )
{
    return flags.m_root.read(in);
}

}
