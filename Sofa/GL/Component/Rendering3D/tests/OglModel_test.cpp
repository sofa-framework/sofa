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
#include <sofa/gl/component/rendering3d/OglModel.h>
#include <gtest/gtest.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/graph/SimpleApi.h>
#include <sofa/testing/TestMessageHandler.h>

namespace sofa
{

using gl::component::rendering3d::OglModel;

TEST(OglModel, templateName)
{
    EXPECT_EQ(sofa::core::objectmodel::BaseClassNameHelper::DefaultTypeTemplateName<OglModel>::Get(), "");
    EXPECT_EQ(sofa::core::objectmodel::BaseClassNameHelper::getTemplateName<OglModel>(), "Vec3d");

    const auto oglModel = core::objectmodel::New<OglModel>();
    EXPECT_EQ(oglModel->getTemplateName(), "Vec3d");

    helper::logging::MessageDispatcher::addHandler( sofa::testing::MainGtestMessageHandler::getInstance() );

    simulation::Simulation* simulation = sofa::simulation::getSimulation();
    const simulation::Node::SPtr root = simulation->createNewGraph("root");

    {
        EXPECT_MSG_NOEMIT(Error);

        simpleapi::createObject(root, "OglModel", {
            {"name", "Vec3"},
            {"template","Vec3"},
        });

        simpleapi::createObject(root, "OglModel", {
            {"name", "Vec3d"},
            {"template","Vec3d"},
        });

        simpleapi::createObject(root, "OglModel", {
            {"name", "empty"},
            {"template",""},
        });

        simpleapi::createObject(root, "OglModel", {
            {"name", "notemplate"},
        });
    }

    {
        EXPECT_MSG_EMIT(Error);

        simpleapi::createObject(root, "OglModel", {
            {"name", "fake"},
            {"template","fake"},
        });

        simpleapi::createObject(root, "OglModel", {
            {"name", "ExtVec3f"},
            {"template","ExtVec3f"},
        });
    }
}

}
