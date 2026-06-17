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

#include <sofa/simulation/LoadDataSnapshotVisitor.h>
using sofa::simulation::LoadDataSnapshotVisitor;

#include <sofa/simulation/LoadLinkSnapshotVisitor.h>
using sofa::simulation::LoadLinkSnapshotVisitor;

#include <sofa/core/objectmodel/SnapshotJSONExporter.h>

#include <sofa/core/objectmodel/Data.h>
using sofa::core::objectmodel::Data;
using sofa::core::objectmodel::BaseLink;
using sofa::core::objectmodel::SingleLink;
using sofa::core::objectmodel::Snapshot;
using sofa::core::objectmodel::BaseNode;

#include <filesystem>
#include <fstream>

class TestComponent : public Base
{

public:
    Data<float> d_value;
    
    TestComponent()
        : d_value(initData(&d_value, 3.14f, "value", "test value"))
    {
        this->setName("pi");
    }

    void saveData(Snapshot::SnapshotObject& snapshot)
    {
        this->saveDataIn(snapshot);
    }

    void saveLinks(Snapshot::SnapshotObject& snapshot)
    {
        this->saveLinksIn(snapshot);
    }

    std::shared_ptr<Snapshot::SnapshotObject> createSnapshotObjectTest(std::vector<std::shared_ptr<Snapshot::SnapshotNode>>& parents) const
    {
        
        return this->createSnapshotObject(parents);
    }

};

class TestComponentA : public BaseComponent
{
public:
    SOFA_CLASS(TestComponentA, BaseComponent);

    Data<float> d_value;
    sofa::MultiLink<TestComponentA, BaseComponent, BaseLink::FLAG_DOUBLELINK> l_target;

    // SingleLink<TestComponentA, BaseComponent, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_target;

    TestComponentA()
        : d_value(initData(&d_value, 3.14f, "value", "test value"))
        , l_target(initLink("target","target test"))
    {
        this->setName("pi");
    }

};

class Snapshot_test: public BaseSimulationTest
{
public:
    Snapshot_test() {}
    ~Snapshot_test() override {}
};


TEST_F(Snapshot_test, saveDataIn)
{
    // TEST of saveDataIn method
    // Check if the snapshot contains the component with expected data

    TestComponent tComponent;
    auto snapshot = std::make_shared<Snapshot::SnapshotObject>();
    tComponent.saveData(*snapshot);
    for (auto& data : snapshot->m_dataContainer)
    {
        if(data.name == "name")
        {
            EXPECT_EQ(data.value, "pi");
        }

        if(data.name == "value")
        {
            EXPECT_EQ(data.value, "3.14");
        }
    }
}

TEST_F(Snapshot_test, saveLinkIn)
{
    // TEST of saveLinksIn method
    // Check if the snapshot contains the component with expected data

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

TEST_F(Snapshot_test, createSnapshotObject)
{
    // TEST of createSnapshotObject
    // Check if createSnapshotObject can create a SnapshotObject
    // To verify, A name and some data are added to the SnapshotObject

    TestComponent tComponent;
    std::vector<std::shared_ptr<Snapshot::SnapshotNode>> snapshotParents;
    auto snapshotObject = tComponent.createSnapshotObjectTest(snapshotParents);

    snapshotObject->m_name = "snapshotObject";
    tComponent.saveData(*snapshotObject);

    EXPECT_NE(snapshotObject, nullptr);
    EXPECT_EQ(snapshotObject->m_name, "snapshotObject");

    for (auto& data : snapshotObject->m_dataContainer)
    {
        if(data.name == "name")
        {
            EXPECT_EQ(data.value, "pi");
        }

        if(data.name == "value")
        {
            EXPECT_EQ(data.value, "3.14");
        }
    }
}

TEST_F(Snapshot_test, findSnapshotObject)
{
    // TEST of findSnapshotObject
    // Check if findSnapshotObject can find the SnapshotObject in a Snapshot
    // with the component's name

    TestComponent tComponent;

    auto snapshotNode = std::make_shared<Snapshot::SnapshotNode>("root");
    std::vector<std::shared_ptr<Snapshot::SnapshotNode>> snapshotParents;
    snapshotParents.push_back(snapshotNode); 

    auto snapshot = tComponent.saveSnapshot(snapshotParents);
    snapshotNode->components.push_back(*snapshot);

    auto expectedObject = tComponent.findSnapshotObject(snapshotNode, "pi");

    EXPECT_NE(expectedObject, nullptr);

    EXPECT_EQ(tComponent.getName(), expectedObject->m_name);  

}

TEST_F(Snapshot_test, saveSnapshot)
{
    // Test of saveSnapshot

    TestComponent tComponent;

    auto snapshot = std::make_shared<Snapshot::SnapshotObject>();
    std::vector<std::shared_ptr<Snapshot::SnapshotNode>> snapshotParents;
    snapshot = tComponent.saveSnapshot(snapshotParents);

    EXPECT_EQ(snapshot->m_name, "pi");
    EXPECT_EQ(snapshot->m_dataContainer.size(), 6);
    EXPECT_EQ(snapshot->m_dataContainer[0].name, "name");
    EXPECT_EQ(snapshot->m_dataContainer[0].value, "pi");
    EXPECT_EQ(snapshot->m_dataContainer[5].name, "value");
    EXPECT_EQ(snapshot->m_dataContainer[5].value, "3.14");
}

TEST_F(Snapshot_test, loadSnapshot)
{
    // Test of loadSnapshot

    TestComponent tComponent;

    auto snapshot = std::make_shared<Snapshot::SnapshotObject>();
    std::vector<std::shared_ptr<Snapshot::SnapshotNode>> snapshotParents;
    snapshot = tComponent.saveSnapshot(snapshotParents);

    TestComponent tcomponent2;
    tcomponent2.d_value.setValue(0.0f);
    tcomponent2.loadDataSnapshot(snapshot);

    EXPECT_EQ(tcomponent2.d_value.getValue(), 3.14f);
}

TEST_F(Snapshot_test, loadLinkSnapshot)
{
    /// Test of loadLinkSnapshot
    /// wip
    TestComponentA Component1;
    Component1.setName("Component1");
    TestComponentA Component2;
    TestComponentA* ptr;
    ptr = &Component2;
    Component2.setName("Component2");
    TestComponentA Component3;
    Component3.setName("Component3");


    auto snapshotNode = std::make_shared<Snapshot::SnapshotNode>("root");
    std::vector<std::shared_ptr<Snapshot::SnapshotNode>> snapshotParents;
    snapshotParents.push_back(snapshotNode);

    std::cout << "Component1 l_target value : " <<Component1.l_target.getValueString() << std::endl;

    Component1.l_target.add(ptr);

    std::cout << "Component1 l_target value : " <<Component1.l_target.getValueString() << std::endl;
    EXPECT_EQ(Component1.l_target.getValueString(), "@Component2");

    auto snapshotObject1 = std::make_shared<Snapshot::SnapshotObject>();
    snapshotObject1 = Component1.saveSnapshot(snapshotParents);

    ptr = &Component3;
    Component1.l_target.add(ptr);

    std::cout << "Component1 l_target value : " <<Component1.l_target.getValueString() << std::endl;
    // EXPECT_EQ(Component1.l_target.getValueString(), "@Component2 @Component3");

    Component1.loadLinkSnapshot(snapshotObject1);

    std::cout << "Component1 l_target value : " <<Component1.l_target.getValueString() << std::endl;
    // EXPECT_EQ(Component1.l_target.getValueString(), "@Component2");

    ptr = &Component2;
    Component1.l_target.remove(ptr);
    std::cout << "Component1 l_target value : " <<Component1.l_target.getValueString() << std::endl;

}

TEST_F(Snapshot_test, SnapshotJSONExporter)
{
    // TEST SnapshotJSONExporter
    // Test the behavior of the export and the import with JSON

    const std::string scene = R"(
        <?xml version='1.0'?>
        <Node name='Root' gravity='0 -9.81 0' time='0' animate='0' >
            <RequiredPlugin name='Sofa.Component.StateContainer'/>
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
    auto m_snapshot = std::make_shared<Snapshot>();

    auto visitor = SaveSnapshotVisitor(nullptr, *m_snapshot);
    root->execute(visitor);

    exportToJSON(*m_snapshot,path);

    std::ifstream checkFile(path);
    EXPECT_TRUE(checkFile.good());
    checkFile.close();

    auto m_snapshot_import = std::make_shared<Snapshot>();
    importFrom(*m_snapshot_import, path);

    EXPECT_NE(m_snapshot_import->m_graphRoot,nullptr);

    EXPECT_EQ(m_snapshot_import->m_graphRoot->m_name,"Root");
    EXPECT_EQ(m_snapshot_import->m_graphRoot->components[0].m_name,"Sofa.Component.StateContainer");
    EXPECT_EQ(m_snapshot_import->m_graphRoot->components[1].m_name,"DefaultAnimationLoop1");
    EXPECT_EQ(m_snapshot_import->m_graphRoot->components[2].m_name,"DefaultVisualManagerLoop1");
    EXPECT_EQ(m_snapshot_import->m_graphRoot->children[0]->m_name,"child1");
    EXPECT_EQ(m_snapshot_import->m_graphRoot->children[0]->components[0].m_name,"MechanicalObject1");

    std::filesystem::remove(path);
}
