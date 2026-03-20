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
#include <sofa/core/objectmodel/Base.h>

#include "gtest/gtest.h"
using sofa::core::objectmodel::Base ;
using sofa::core::objectmodel::ComponentState;

#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest ;
using sofa::simulation::Node ;

#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject;

#include <sofa/simulation/SaveSnapshotVisitor.h>
using sofa::simulation::SaveSnapshotVisitor;

#include <sofa/simulation/LoadDataSnapshotVisitor.h>
using sofa::simulation::LoadDataSnapshotVisitor;

#include <sofa/simulation/LoadLinkSnapshotVisitor.h>
using sofa::simulation::LoadLinkSnapshotVisitor;

#include <sofa/core/objectmodel/Data.h>
using sofa::core::objectmodel::Data;
using sofa::core::objectmodel::BaseLink;
using sofa::core::objectmodel::SingleLink ;
using sofa::core::objectmodel::Snapshot;
using sofa::core::objectmodel::BaseNode;

#include <filesystem>
#include <fstream>


class TestComponent : public Base
{

public:
    
    SOFA_CLASS(TestComponent,Base);

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

class MockSnapshotTest : public Snapshot
{
public:
    MockSnapshotTest() {}
    ~MockSnapshotTest() = default;

    void setupSnapshot()
    {
        this->m_graphRoot = std::make_shared<Snapshot::SnapshotNode>("root");
        auto snapshotObject0 = std::make_shared<Snapshot::SnapshotObject>("snapshotObject0");
        this->m_graphRoot->components.push_back(*snapshotObject0);

        auto snapshotNode1 = std::make_shared<Snapshot::SnapshotNode>("snapshotNode1");
        auto snapshotNode2 = std::make_shared<Snapshot::SnapshotNode>("snapshotNode2");

        auto snapshotObject1 = std::make_shared<Snapshot::SnapshotObject>("snapshotObject1");
        auto snapshotObject2 = std::make_shared<Snapshot::SnapshotObject>("snapshotObject2");

        snapshotNode1->components.push_back(*snapshotObject1);
        snapshotNode2->components.push_back(*snapshotObject2);

        snapshotNode1->children.push_back(snapshotNode2);

        this->m_graphRoot->children.push_back(snapshotNode1);
    }
};

class Snapshot_test: public BaseSimulationTest
{
public:

    SceneInstance* c;
    Node* node {nullptr};
    Snapshot_test() {}
    ~Snapshot_test() override 
    {
        delete c;
    }


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

TEST_F(Snapshot_test, SnapshotJSONExporter)
{
    // TEST SnapshotJSONExporter
    // Test the behavior of the export and the import with JSON

    // auto SnapshotJSONExporterTest = createSnapshot(SnapshotType::JSON);

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

    std::string path = std::filesystem::temp_directory_path() / "testfile.json";
    // auto visitor = SaveSnapshotVisitor(nullptr, *SnapshotJSONExporterTest);
    // root->execute(visitor);
    // SnapshotJSONExporterTest->exportTo(path);
    // EXPECT_NE(SnapshotJSONExporterTest->m_graphRoot,nullptr);
    // std::cout << "SnapshotJSONExporterTest : " << SnapshotJSONExporterTest->m_graphRoot->m_name << std::endl;
    //
    // std::ifstream checkFile(path);
    // EXPECT_TRUE(checkFile.good());
    // checkFile.close();
    //
    // // auto SnapshotJSONExporterTest2 = createSnapshot(SnapshotType::JSON);
    // SnapshotJSONExporterTest2->importFrom(path);
    // EXPECT_NE(SnapshotJSONExporterTest2->m_graphRoot,nullptr);
    //
    // EXPECT_EQ(SnapshotJSONExporterTest2->m_graphRoot->m_name,"Root");
    // EXPECT_EQ(SnapshotJSONExporterTest2->m_graphRoot->components[0].m_name,"Sofa.Component.StateContainer");
    // EXPECT_EQ(SnapshotJSONExporterTest2->m_graphRoot->components[1].m_name,"DefaultAnimationLoop1");
    // EXPECT_EQ(SnapshotJSONExporterTest2->m_graphRoot->components[2].m_name,"DefaultVisualManagerLoop1");
    // EXPECT_EQ(SnapshotJSONExporterTest2->m_graphRoot->children[0]->m_name,"child1");
    // EXPECT_EQ(SnapshotJSONExporterTest2->m_graphRoot->children[0]->components[0].m_name,"MechanicalObject1");

}

TEST_F(Snapshot_test, LoadLinkVisitor)
{
    // auto SnapshotJSONExporterTest = createSnapshot(SnapshotType::JSON);

    // const std::string scene = R"(
    //     <?xml version='1.0'?>
    //     <Node name='Root' gravity='0 -9.81 0' time='0' animate='0' >
    //         <RequiredPlugin name='Sofa.Component.StateContainer'/>
    //         <DefaultAnimationLoop />
    //         <DefaultVisualManagerLoop />
    //         <Node name='child1'>
    //             <MechanicalObject />
    //         </Node>
    //     </Node>
    // )";

    const std::string scene = R"(
        <?xml version="1.0" ?>
        <!-- See http://wiki.sofa-framework.org/mediawiki/index.php/TutorialBasicPendulum -->
        <Node name="root" dt="0.1" gravity="0 0 0">
          <RequiredPlugin name="Sofa.Component.Collision.Geometry"/> <!-- Needed to use components [SphereCollisionModel] -->
          <RequiredPlugin name="Sofa.Component.Constraint.Projective"/> <!-- Needed to use components [FixedProjectiveConstraint] -->
          <RequiredPlugin name="Sofa.Component.LinearSolver.Iterative"/> <!-- Needed to use components [CGLinearSolver] -->
          <RequiredPlugin name="Sofa.Component.Mass"/> <!-- Needed to use components [UniformMass] -->
          <RequiredPlugin name="Sofa.Component.ODESolver.Backward"/> <!-- Needed to use components [EulerImplicitSolver] -->
          <RequiredPlugin name="Sofa.Component.SolidMechanics.Spring"/> <!-- Needed to use components [SpringForceField] -->
          <RequiredPlugin name="Sofa.Component.StateContainer"/> <!-- Needed to use components [MechanicalObject] -->
          <RequiredPlugin name="Sofa.Component.Visual"/> <!-- Needed to use components [VisualStyle] -->

          <DefaultAnimationLoop/>
          <VisualStyle displayFlags="showBehavior showCollisionModels"/>
         <!-- Try to test with different solver -->
          <EulerImplicitSolver name="EulerImplicit"  rayleighStiffness="0.1" rayleighMass="0.1" />
          <CGLinearSolver name="CGSolver" iterations="25" tolerance="1e-5" threshold="1e-5"/>

          <MechanicalObject name="Particles" template="Vec3"
                            position="0 0 0 0 0 1"
                            velocity="0 0 0 0 1 0"/>

          <UniformMass name="Mass" totalMass="1" />

          <FixedProjectiveConstraint indices="0"/>
          <SpringForceField name="Springs" stiffness="100" damping="1" spring="0 1 10 1 1"/>
          <SphereCollisionModel radius="0.1"/>
        </Node>
    )";

    SceneInstance c("xml", scene) ;
    c.initScene() ;

    Node* root = c.root.get() ;

    // std::string path = std::filesystem::temp_directory_path() / "testfile.json";
    // auto visitor = SaveSnapshotVisitor(nullptr, *SnapshotJSONExporterTest);
    // root->execute(visitor);
    // SnapshotJSONExporterTest->exportTo(path);
    // EXPECT_NE(SnapshotJSONExporterTest->m_graphRoot,nullptr);
    // std::cout << "SnapshotJSONExporterTest : " << SnapshotJSONExporterTest->m_graphRoot->m_name << std::endl;
    //
    // std::ifstream checkFile(path);
    // EXPECT_TRUE(checkFile.good());
    // checkFile.close();
    //
    // auto loadvisitor = LoadDataSnapshotVisitor(nullptr, *SnapshotJSONExporterTest);
    // root->execute(loadvisitor);
    // auto loadlinkvisitor = LoadLinkSnapshotVisitor(nullptr, *SnapshotJSONExporterTest);
    // root->execute(loadlinkvisitor);
    // EXPECT_NE(SnapshotJSONExporterTest->m_graphRoot,nullptr);
}