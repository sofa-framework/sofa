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

#include <SofaBase/initSofaBase.h>

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;

#include <SofaMiscCollision/DefaultCollisionGroupManager.h>
using sofa::component::collision::DefaultCollisionGroupManager;

#include <sofa/simulation/graph/DAGSimulation.h>

#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

struct DefaultCollisionGroupManager_test : public BaseTest
{
    void onSetUp() override;
    void onTearDown() override;

    bool combineSingleObject();

private:
    /// Root of the scene graph
    simulation::Node::SPtr root { nullptr };
    /// Simulation
    simulation::Simulation* simulation { nullptr };
};

void DefaultCollisionGroupManager_test::onSetUp()
{
    sofa::component::initSofaBase();

    static const std::string sceneFilename = std::string(SOFAMISCCOLLISION_TEST_SCENES_DIR)
            + "/DefaultCollisionGroupManager_singleObject_test.scn";

    // Init simulation
    simulation = sofa::simulation::getSimulation();
    root = sofa::simulation::node::load(sceneFilename.c_str());
}

void DefaultCollisionGroupManager_test::onTearDown()
{
    if (root != nullptr)
    {
        sofa::simulation::node::unload(root);
    }
}

bool DefaultCollisionGroupManager_test::combineSingleObject()
{
    EXPECT_TRUE(root != nullptr);
    EXPECT_TRUE(sofa::simulation::getSimulation() != nullptr);

    sofa::simulation::node::initRoot(root.get());

    // run 200 time steps
    // objectives:
    // 1) The simulation does not crash
    // 2) Collision prevents the cube to fall through the floor
    for (unsigned int i = 0; i < 200; ++i)
    {
        sofa::simulation::node::animate(root.get(), 0.01);
    }

    auto* baseObject = root->getTreeNode("Cube1")->getObject("mechanicalObject");
    EXPECT_NE(baseObject, nullptr);

    auto* mechanicalObject = dynamic_cast<sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Vec3Types>*>(baseObject);
    EXPECT_NE(mechanicalObject, nullptr);

    const auto position = mechanicalObject->readPositions();
    EXPECT_FALSE(position.empty());

    // Check that the position of the first DOF is not below the floor
    EXPECT_GT(position->front().y(), -15.);

    return true;
}

TEST_F(DefaultCollisionGroupManager_test, combine)
{
    ASSERT_TRUE(combineSingleObject());
}

}
