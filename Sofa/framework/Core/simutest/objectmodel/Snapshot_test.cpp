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
#include <iostream>
#include <vector>
#include <sofa/core/objectmodel/Base.h>

#include "gtest/gtest.h"
using sofa::core::objectmodel::Base ;
using sofa::core::objectmodel::ComponentState;

#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest ;
using sofa::simulation::Node ;

#include <sofa/core/objectmodel/BaseComponent.h>
using sofa::core::objectmodel::BaseComponent;

#include <sofa/core/objectmodel/Snapshot.h>
using sofa::core::objectmodel::Snapshot;

#include <sofa/simulation/SaveSnapshotVisitor.h>
using sofa::simulation::SaveSnapshotVisitor;

#include <sofa/core/objectmodel/SnapshotJSONExporter.h>

#include <sofa/core/objectmodel/Data.h>
using sofa::core::objectmodel::Data;
using sofa::core::objectmodel::BaseLink;
using sofa::core::objectmodel::SingleLink;
using sofa::core::objectmodel::MultiLink;
using sofa::core::objectmodel::Snapshot;
using sofa::core::objectmodel::BaseNode;

#include <filesystem>
#include <fstream>

class TestComponent : public BaseComponent
{
public:
    SOFA_CLASS(TestComponent, BaseComponent);

    Data<float> d_value;
    sofa::MultiLink<TestComponent, BaseComponent, BaseLink::FLAG_DOUBLELINK> l_target;

    TestComponent()
        : d_value(initData(&d_value, 3.14f, "pi", "test value"))
        , l_target(initLink("target","target test"))
    {
        this->setName("TestComponent");
    }

    void saveData(Snapshot::SnapshotObject& snapshot)
    {
        for (const auto& dataFields = this->getDataFields(); const auto& data : dataFields)
        {
            Snapshot::DataInfo dataInfo;
            dataInfo.name = data->getName();
            dataInfo.type = data->getValueTypeString();
            dataInfo.value = data->getValueString();

            snapshot.m_dataContainer.push_back(dataInfo);
        }
    }

    void saveLinks(Snapshot::SnapshotObject& snapshot)
    {
        for (const auto& links = this->getLinks(); const auto& link : links)
        {
            Snapshot::LinkInfo linkInfo;
            linkInfo.name = link->getName();
            linkInfo.type = link->getValueTypeString();
            linkInfo.value = link->getValueString();

            std::string search = "//";
            sofa::helper::replaceAll(linkInfo.value, search,"");
            snapshot.m_linkContainer.push_back(linkInfo);
        }
    }

    std::shared_ptr<Snapshot::SnapshotObject> createSnapshotObjectTest(std::vector<std::shared_ptr<Snapshot::SnapshotNode>>& parents) const
    {

        return this->createSnapshotObject(parents);
    }

    std::shared_ptr<Snapshot::SnapshotObject> findSnapshotObjectTest(const std::shared_ptr<Snapshot::SnapshotNode>& parents, const std::string& objectname)
    {
        return this->findSnapshotObject(parents, objectname);
    }

};

class Snapshot_test: public BaseSimulationTest
{
public:
    Snapshot_test() {}
    ~Snapshot_test() override {}
};

/**
 * @brief Test of saveDataIn
 *
 * This test verifies that saveDataIn save data correctly in a SnapshotObject.
 *
 * Test steps:
 * 1. Create a component (Component) and a snapshot
 * 2. Save component's data in the snapshot
 * 3. Check if the snapshot contains the component with expected data
 *
 */
TEST_F(Snapshot_test, saveDataIn)
{
    TestComponent Component;
    auto snapshot = std::make_shared<Snapshot::SnapshotObject>();
    Component.saveData(*snapshot);
    for (auto& data : snapshot->m_dataContainer)
    {
        if (data.name == "name")
        {
            EXPECT_EQ(data.value, "TestComponent");
        }
        if(data.name == "pi")
        {
            EXPECT_EQ(data.value, "3.14");
        }
    }
}

/**
 * @brief Test of saveLinksIn
 *
 * This test verifies that saveLinksIn save links correctly in a SnapshotObject.
 *
 * Test steps:
 * 1. Create a component (Component) and a snapshot
 * 2. Save component's links in the snapshot
 * 3. Check if the snapshot contains the component with expected links
 *
 */
TEST_F(Snapshot_test, saveLinkIn)
{

    TestComponent tComponent;
    auto snapshot = std::make_shared<Snapshot::SnapshotObject>();

    tComponent.saveLinks(*snapshot);
    for (auto& link : snapshot->m_linkContainer)
    {
        if (link.name == "name")
        {
            EXPECT_EQ(link.value, "@./");
        }
        if (link.name == "slaves")
        {
            EXPECT_EQ(link.value, "");
        }
        if (link.name == "master")
        {
            EXPECT_EQ(link.value, "");
        }
    }
}

/**
 * @brief Test of createSnapshotObject
 *
 * This test verifies that createSnapshotObject can find the SnapshotObject in a Snapshot with the component's name.
 *
 * Test steps:
 * 1. Create a component (Component)
 * 2. Create a SnapshotObject with the function createSnapshotObject
 * 3. Save Component's data in the SnapshotObject
 * 4. Verify if every data is correctly saved in the SnapshotObject
 *
 */
TEST_F(Snapshot_test, createSnapshotObject)
{
    TestComponent Component;

    std::vector<std::shared_ptr<Snapshot::SnapshotNode>> snapshotParents;
    auto snapshotObject = Component.createSnapshotObjectTest(snapshotParents);

    snapshotObject->m_name = "snapshotObject";
    Component.saveData(*snapshotObject);

    EXPECT_NE(snapshotObject, nullptr);
    EXPECT_EQ(snapshotObject->m_name, "snapshotObject");
    for (auto& data : snapshotObject->m_dataContainer)
    {
        if(data.name == "pi")
        {
            EXPECT_EQ(data.value, "3.14");
        }
    }
}

/**
 * @brief Test of findSnapshotObject
 *
 * This test verifies that findSnapshotObject can find the SnapshotObject in a Snapshot with the component's name.
 *
 * Test steps:
 * 1. Create a component (Component) and a graph of with a SnapshotNode
 * 2. Save Component in the SnapshotNode (as a SnapshotObject)
 * 3. Use findSnapshotObject to find the SnapshotObject corresponding to Component
 * 4. Verify if the SnapshotObject has been correctly found
 *
 */
TEST_F(Snapshot_test, findSnapshotObject)
{
    TestComponent Component;
    auto snapshotNode = std::make_shared<Snapshot::SnapshotNode>("root");
    std::vector<std::shared_ptr<Snapshot::SnapshotNode>> snapshotParents;
    snapshotParents.push_back(snapshotNode);

    auto snapshot = Component.saveSnapshot(snapshotParents);
    snapshotNode->components.push_back(*snapshot);

    auto expectedObject = Component.findSnapshotObjectTest(snapshotNode, "TestComponent");

    EXPECT_NE(expectedObject, nullptr);
    EXPECT_EQ(Component.getName(), expectedObject->m_name);
}

/**
 * @brief Test of saveSnapshot
 *
 * This test verifies that saveSnapshot save the data to a previously saved snapshot.
 *
 * Test steps:
 * 1. Create a component (Component) and a snapshot
 * 2. Save Component1's data in the snapshot
 * 3. Verify if the snapshot contains all the data from Component1
 *
 */
TEST_F(Snapshot_test, saveSnapshot)
{
    TestComponent Component;
    auto snapshot = std::make_shared<Snapshot::SnapshotObject>();
    std::vector<std::shared_ptr<Snapshot::SnapshotNode>> snapshotParents;

    snapshot = Component.saveSnapshot(snapshotParents);

    EXPECT_EQ(snapshot->m_name, "TestComponent");
    EXPECT_EQ(snapshot->m_dataContainer.size(), 7);
    EXPECT_EQ(snapshot->m_dataContainer[0].name, "name");
    EXPECT_EQ(snapshot->m_dataContainer[0].value, "TestComponent");
    EXPECT_EQ(snapshot->m_dataContainer.back().name, "pi");
    EXPECT_EQ(snapshot->m_dataContainer.back().value, "3.14");
    EXPECT_EQ(snapshot->m_linkContainer[0].name, "context");
    EXPECT_EQ(snapshot->m_linkContainer[0].value, "@./");
    EXPECT_EQ(snapshot->m_linkContainer.back().name, "target");
    EXPECT_EQ(snapshot->m_linkContainer.back().value, "");

}

/**
 * @brief Test of loadDataSnapshot
 *
 * This test verifies that loadLinkSnapshot restores the state of data to a previously saved snapshot
 *
 * Test steps:
 * 1. Create a component (Component1) and a snapshot
 * 2. Save Component1's data in the snapshot
 * 3. Create another component (Component2) and change the data d_value
 * 4. Load Component1's data into Component2 with loadDataSnapshot
 * 5. Verify if Component2 has same value as Component1
 *
 */
TEST_F(Snapshot_test, loadDataSnapshot)
{
    TestComponent Component1;
    auto snapshot = std::make_shared<Snapshot::SnapshotObject>();
    std::vector<std::shared_ptr<Snapshot::SnapshotNode>> snapshotParents;

    snapshot = Component1.saveSnapshot(snapshotParents);

    TestComponent Component2;
    Component2.d_value.setValue(0.0f);
    Component2.loadSnapshot(snapshot);

    EXPECT_EQ(Component2.d_value.getValue(), 3.14f);
}

/**
 * @brief Test of loadLinkSnapshot
 *
 * This test verifies that loadLinkSnapshot restores the state of a link to a previously saved snapshot.
 * First, it set up a graph "root" with 3 components: Component1, Component2 and Component3
 * Each component holds a multi-link l_target pointing to other components
 *
 * Test steps:
 * 1. Add Component2 to Component1's l_target link.
 * 2. Save a snapshot of Component1 (l_target points to @Component2 only).
 * 3. Add Component3 to Component1's l_target link.
 * 4. Verify that l_target now points to both @Component2 and@Component3.
 * 5. Restore Component1 from the snapshot.
 * 6. Verify that l_target is back to pointing to @Component2 only,
 *    confirming that it @Component3 was correctly removed by loadLinkSnapshot.
 *
 */
TEST_F(Snapshot_test, loadLinkSnapshot)
{
    const SceneInstance scene("root");
    auto Component1 = sofa::core::objectmodel::New<TestComponent>();
    Component1->setName("Component1");
    auto Component2 = sofa::core::objectmodel::New<TestComponent>();
    Component2->setName("Component2");
    auto Component3 = sofa::core::objectmodel::New<TestComponent>();
    Component3->setName("Component3");
    scene.root->addObject(Component1);
    scene.root->addObject(Component2);
    scene.root->addObject(Component3);

    auto snapshotNode = std::make_shared<Snapshot::SnapshotNode>("root");
    std::vector<std::shared_ptr<Snapshot::SnapshotNode>> snapshotParents;
    snapshotParents.push_back(snapshotNode);

    TestComponent* ptr = Component2.get();
    Component1->l_target.add(ptr);

    EXPECT_EQ(Component1->l_target.getValueString(), "@Component2");

    auto snapshotObject1 = std::make_shared<Snapshot::SnapshotObject>();
    snapshotObject1 = Component1->saveSnapshot(snapshotParents);

    ptr = Component3.get();
    Component1->l_target.add(ptr);

    EXPECT_EQ(Component1->l_target.getValueString(), "@Component2 @Component3");

    Component1->loadSnapshot(snapshotObject1);

    EXPECT_EQ(Component1->l_target.getValueString(), "@Component2");
}

/**
 * @brief Test of SnapshotJSONExporter
 *
 * This test verifies the behavior of the export and the import with SnapshotJSONExporter
 *
 * Test steps:
 * 1. Init the scene
 * 2. Create a snapshot (snapshot)
 * 3. Run SaveSnapshotVisitor to save the state in m_snapshot
 * 4. Export m_snapshot to a JSON file and check if the file is valid
 * 5. Create another snapshot (snapshot_import)
 * 6. Import the JSON file in snapshot_import
 * 7. Compare m_snapshot and snapshot_import
 *
 */
TEST_F(Snapshot_test, SnapshotJSONExporter)
{
    const std::string scene = R"(
        <?xml version='1.0'?>
        <Node name='Root' gravity='0 -9.81 0' time='0' animate='0' >
            <RequiredPlugin pluginName='Sofa.Component.StateContainer'/>
            <DefaultAnimationLoop />
            <DefaultVisualManagerLoop />
            <Node name='child1'>
                <MechanicalObject />
            </Node>
        </Node>
    )";

    SceneInstance c("xml", scene) ;
    c.initScene() ;

    Node* root = c.root.get() ;

    std::filesystem::path path = std::filesystem::temp_directory_path() / "test_file.json";
    auto snapshot = std::make_shared<Snapshot>();

    auto visitor = SaveSnapshotVisitor(nullptr, *snapshot);
    root->execute(visitor);

    exportToJSON(*snapshot, path.string());

    std::ifstream checkFile(path);
    EXPECT_TRUE(checkFile.good());
    checkFile.close();

    auto snapshot_import = std::make_shared<Snapshot>();
    importFrom(*snapshot_import, path.string());

    EXPECT_NE(snapshot_import->m_graphRoot,nullptr);

    EXPECT_EQ(snapshot_import->m_graphRoot->m_name,"Root");
    EXPECT_EQ(snapshot_import->m_graphRoot->components[1].m_name,"DefaultAnimationLoop1");
    EXPECT_EQ(snapshot_import->m_graphRoot->components[2].m_name,"DefaultVisualManagerLoop1");
    EXPECT_EQ(snapshot_import->m_graphRoot->children[0]->m_name,"child1");
    EXPECT_EQ(snapshot_import->m_graphRoot->children[0]->components[0].m_name,"MechanicalObject1");

    std::filesystem::remove(path);
}
