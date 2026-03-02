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
using sofa::core::objectmodel::Base ;
using sofa::core::objectmodel::ComponentState;

#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest ;
using sofa::simulation::Node ;

#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject;

#include <sofa/core/objectmodel/SnapshotFactory.h>
using sofa::core::objectmodel::SnapshotType;

#include <sofa/simulation/SnapshotVisitor.h>
using sofa::simulation::SnapshotVisitor;

#include <sofa/core/objectmodel/Data.h>
using sofa::core::objectmodel::Data;
using sofa::core::objectmodel::BaseLink;
using sofa::core::objectmodel::SingleLink ;
using sofa::core::objectmodel::BaseSnapshot;
using sofa::core::objectmodel::BaseNode;

class TestComponent : public Base
{
public:
    
    Data<float> d_value;
    
    TestComponent() 
        : d_value(initData(&d_value, 3.14f, "value", "test value"))
    {
        this->setName("pi");
    }

    void saveData(BaseSnapshot::SnapshotObject& snapshot)
    {
        this->saveDataIn(snapshot);
    }

    void saveLinks(BaseSnapshot::SnapshotObject& snapshot)
    {
        this->saveLinksIn(snapshot);
    }

    std::shared_ptr<BaseSnapshot::SnapshotObject> createSnapshotObjectTest(std::vector<std::shared_ptr<BaseSnapshot::SnapshotNode>>& parents) const
    {
        
        return this->createSnapshotObject(parents);
    }

    SOFA_CLASS(TestComponent,Base);
};

class MockSnapshotTest : public BaseSnapshot
{
public:
    void importSnapshot(const std::string filename) override
    {}
    void exportTo(const std::string filename) override
    {}
    void importFrom(std::string filename) override
    {}
    

    MockSnapshotTest() {}
    ~MockSnapshotTest() = default;
};

class Snapshot_test: public BaseSimulationTest
{
public:

    SceneInstance* c;
    Node* node {nullptr};
    Snapshot_test()
    {
        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "   <RequiredPlugin name='Sofa.Component.SceneUtility' />                   \n"
                 "   <DefaultAnimationLoop />                                                \n"
                 "   <DefaultVisualManagerLoop />                                            \n"
                 "   <MechanicalObject name='mstate0'/>                                      \n"
                 "   <InfoComponent name='obj'/>                                             \n"
                 "   <Node name='child1'>                                                    \n"
                 "      <MechanicalObject name='mstate1'/>                                   \n"
                 "      <Node name='child2'>                                                 \n"
                 "      </Node>                                                              \n"
                 "   </Node>                                                                 \n"
                 "</Node>                                                                    \n" ;
        c = new SceneInstance("xml", scene.str()) ;
        c->initScene() ;
        const Node* root = c->root.get() ;
        Base* b = sofa::core::PathResolver::FindBaseFromPath(root, "@/child1/child2");
        node = dynamic_cast<Node*>(b);
    }
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
    auto snapshot = std::make_shared<BaseSnapshot::SnapshotObject>();
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
    std::vector<std::shared_ptr<BaseSnapshot::SnapshotNode>> snapshotParents;
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

    auto snapshotNode = std::make_shared<BaseSnapshot::SnapshotNode>("root");
    std::vector<std::shared_ptr<BaseSnapshot::SnapshotNode>> snapshotParents;
    snapshotParents.push_back(snapshotNode); 

    auto snapshot = tComponent.saveSnapshot(snapshotParents);
    snapshotNode->components.push_back(*snapshot);

    auto expectedObject = tComponent.findSnapshotObject(snapshotNode, "pi");

    EXPECT_NE(expectedObject, nullptr);

    EXPECT_EQ(tComponent.getName(), expectedObject->m_name);  

}

TEST_F(Snapshot_test, saveSnapshot)
{
    TestComponent tComponent;

    auto snapshot = std::make_shared<BaseSnapshot::SnapshotObject>();
    std::vector<std::shared_ptr<BaseSnapshot::SnapshotNode>> snapshotParents;
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
    TestComponent tComponent;

    auto snapshot = std::make_shared<BaseSnapshot::SnapshotObject>();
    std::vector<std::shared_ptr<BaseSnapshot::SnapshotNode>> snapshotParents;
    snapshot = tComponent.saveSnapshot(snapshotParents);

    TestComponent tcomponent2;
    tcomponent2.d_value.setValue(0.0f);
    tcomponent2.loadSnapshot(snapshot);

    EXPECT_EQ(tcomponent2.d_value.getValue(), 3.14f);
}

