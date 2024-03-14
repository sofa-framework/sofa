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
#include <sofa/core/visual/Data[DisplayFlags].h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <gtest/gtest.h>
#include <sofa/core/logging/PerComponentLoggingMessageHandler.h>

namespace sofa
{

TEST(DisplayFlags, getAllFlagsLabels)
{
    const core::visual::DisplayFlags displayFlags;
    const auto labels = displayFlags.getAllFlagsLabels();

    EXPECT_EQ(labels.size(), 40);

    const sofa::type::vector<std::string> expectedLabels
    { "showRoot", "hideRoot", "showAll", "hideAll", "showVisual", "hideVisual", "showVisualModels", "hideVisualModels", "showBehavior", "hideBehavior", "showBehaviorModels", "hideBehaviorModels", "showForceFields", "hideForceFields", "showInteractionForceFields", "hideInteractionForceFields", "showCollision", "hideCollision", "showCollisionModels", "hideCollisionModels", "showBoundingCollisionModels", "hideBoundingCollisionModels", "showDetectionOutputs", "hideDetectionOutputs", "showMapping", "hideMapping", "showMappings", "hideMappings", "showMechanicalMappings", "hideMechanicalMappings", "showOptions", "hideOptions", "showAdvancedRendering", "showRendering", "hideAdvancedRendering", "hideRendering", "showWireframe", "hideWireframe", "showNormals", "hideNormals"}
    ;
    for (const auto& label : expectedLabels)
    {
        EXPECT_NE(std::find(labels.begin(), labels.end(), label), labels.end())
            << label << " not found";
    }
}

class DummyDisplayFlagsOwner : public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(DummyDisplayFlagsOwner, sofa::core::objectmodel::BaseObject);

    Data<core::visual::DisplayFlags> m_displayFlags;

    DummyDisplayFlagsOwner() : m_displayFlags(initData(&m_displayFlags, "displayFlags", "help")) {}
};

class RAIIPerComponentLoggingMessageHandler : public helper::logging::PerComponentLoggingMessageHandler
{
public:
    ~RAIIPerComponentLoggingMessageHandler() override
    {
        helper::logging::MessageDispatcher::rmHandler(this);
    }
};

TEST(DisplayFlags, readFromObject)
{
    RAIIPerComponentLoggingMessageHandler logHandler;
    helper::logging::MessageDispatcher::addHandler(&logHandler) ;

    const auto object = core::objectmodel::New<DummyDisplayFlagsOwner>();

    object->m_displayFlags.read("showForceFields");
    EXPECT_TRUE(object->getLoggedMessages().empty());

    object->m_displayFlags.read("showForceFie");
    EXPECT_FALSE(object->getLoggedMessages().empty());
    EXPECT_EQ(object->getLoggedMessages().front().messageAsString(),
              "Unknown flag 'showForceFie'. The closest existing ones:  \n"
              "\t- showForceFields (88% match)  \n"
              "\t- hideForceFields (66% match)  \n"
              "\t- showInteractionForceFields (63% match)  \n"
              "Complete list is: [\"showRoot\", \"hideRoot\", \"showAll\", \"hideAll\", \"showVisual\", \"hideVisual\", \"showVisualModels\", \"hideVisualModels\", \"showBehavior\", \"hideBehavior\", \"showBehaviorModels\", \"hideBehaviorModels\", \"showForceFields\", \"hideForceFields\", \"showInteractionForceFields\", \"hideInteractionForceFields\", \"showCollision\", \"hideCollision\", \"showCollisionModels\", \"hideCollisionModels\", \"showBoundingCollisionModels\", \"hideBoundingCollisionModels\", \"showDetectionOutputs\", \"hideDetectionOutputs\", \"showMapping\", \"hideMapping\", \"showMappings\", \"hideMappings\", \"showMechanicalMappings\", \"hideMechanicalMappings\", \"showOptions\", \"hideOptions\", \"showAdvancedRendering\", \"showRendering\", \"hideAdvancedRendering\", \"hideRendering\", \"showWireframe\", \"hideWireframe\", \"showNormals\", \"hideNormals\"]");


    object->clearLoggedMessages();
    EXPECT_TRUE(object->getLoggedMessages().empty());

    object->m_displayFlags.read("showforcefields");
    EXPECT_FALSE(object->getLoggedMessages().empty());
    EXPECT_EQ(object->getLoggedMessages().front().messageAsString(),
        "Letter case of flag 'showforcefields' is not correct, please use 'showForceFields' instead");
}

TEST(DisplayFlags, readFromData)
{
    Data<core::visual::DisplayFlags> displayFlags;

    displayFlags.read("showForceFields");
    displayFlags.read("showForceFiel");
}

}
