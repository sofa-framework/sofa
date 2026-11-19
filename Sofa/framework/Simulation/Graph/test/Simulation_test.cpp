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
#include <sofa/testing/NumericTest.h>
using sofa::testing::NumericTest ;

#include <sofa/defaulttype/VecTypes.h>
using sofa::defaulttype::Vec3Types ;

#include <sofa/component/statecontainer/MechanicalObject.h>
typedef sofa::component::statecontainer::MechanicalObject<Vec3Types> MechanicalObject3;

#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/DeleteVisitor.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/component/mass/UniformMass.h>

#include <sofa/simulation/DefaultAnimationLoop.h>

#include <sofa/simulation/Node.h>
#include <sofa/simulation/UpdateBoundingBoxVisitor.h>
#include <sofa/core/objectmodel/BaseObjectDescription.h>

namespace sofa {

static int objectCounter;

/// Component class with an instance counter
template <class Object>
struct InstrumentedObject : public Object
{
    InstrumentedObject()  { objectCounter++; }
    ~InstrumentedObject() override { objectCounter--; }
};

/// Component with a sub-component
template <class O1, class _O2>
struct ParentObject : public O1
{
    typedef InstrumentedObject<_O2> O2;
    typename O2::SPtr o2;
    ParentObject()  { o2 = core::objectmodel::New<O2>(); }
};




/** Test the Simulation class
*/
struct Scene_test: public NumericTest<SReal>
{
    // root
    simulation::Simulation* simulation;
    simulation::Node::SPtr root;

    Scene_test()
    {
        simulation = sofa::simulation::MainSimulation::getSimulation();
    }

    /// Test Simulation::computeBBox
    void computeBBox()
    {
        // Init Sofa
        root = simulation::MainSimulation::getSimulation()->createNewGraph("root");

        // create DOFs and its expected bounding box
        const MechanicalObject3::SPtr DOF = core::objectmodel::New<MechanicalObject3>();
        root->addObject(DOF);
        DOF->resize(4);
        MechanicalObject3::WriteVecCoord x = DOF->writePositions();
        x[0] = type::Vec3(0,0,0);
        x[1] = type::Vec3(1,0,0);
        x[2] = type::Vec3(0,1,0);
        x[3] = type::Vec3(0,0,1);
        type::Vec3 expectedMin(0,0,0), expectedMax(1,1,1);
        DOF->showObject.setValue(true); // bbox is updated only for drawn MO

        // end create scene
        //*********
        sofa::simulation::node::initRoot(root.get());
        //*********

        type::Vec3 sceneMinBBox, sceneMaxBBox;
        sofa::simulation::node::computeBBox(root.get(), sceneMinBBox.ptr(), sceneMaxBBox.ptr());

        if( vectorMaxDiff(sceneMinBBox,expectedMin)>this->epsilon() || vectorMaxDiff(sceneMaxBBox,expectedMax)>this->epsilon() )
        {
            ADD_FAILURE() << "Wrong bounding box, expected (" << expectedMin <<", "<<expectedMax<<") , got ("<< sceneMinBBox <<", "<<sceneMaxBBox << ")" << std::endl;
        }

    }

    /// create a component and replace it with an other one
    void objectDestruction_replace()
    {
        typedef InstrumentedObject<component::mass::UniformMass<defaulttype::Vec3Types> > Component;

        objectCounter = 0;
        Component::SPtr toto = core::objectmodel::New<Component>();
        toto = core::objectmodel::New<Component>(); // this should first delete the previous one
        if(objectCounter != 1)
            ADD_FAILURE() << objectCounter << " objects, should be only 1 ! " <<std::endl;
    }

    /// create a component and replace it with an other one
    void objectDestruction_delete()
    {
        typedef InstrumentedObject<component::mass::UniformMass<defaulttype::Vec3Types> > Component;

        objectCounter = 0;
        Component::SPtr toto = core::objectmodel::New<Component>();
    }

    /// create a component and set it to nullptr
    void objectDestruction_setNull()
    {
        typedef InstrumentedObject<component::mass::UniformMass<defaulttype::Vec3Types> > Component;

        objectCounter = 0;
        Component::SPtr toto = core::objectmodel::New<Component>();
        toto = nullptr;
        checkDeletions();
    }

    /// create a component and set it to nullptr
    void objectDestruction_reset()
    {
        typedef InstrumentedObject<component::mass::UniformMass<defaulttype::Vec3Types> > Component;

        objectCounter = 0;
        Component::SPtr toto = core::objectmodel::New<Component>();
        toto.reset();
        checkDeletions();
    }

    /// create and delete a component with a sub-component
    void objectDestruction_subObject()
    {
        typedef ParentObject<MechanicalObject3,MechanicalObject3> PO;

        objectCounter = 0;
        PO::SPtr toto = core::objectmodel::New<PO>();
        // deletion of toto at function exit
    }

    /// create a scene, remove a node then step the scene
    void objectDestruction_subNodeAndStep()
    {
        root = simulation::MainSimulation::getSimulation()->createNewGraph("root");
        root->addObject(core::objectmodel::New<sofa::simulation::DefaultAnimationLoop>());

        core::objectmodel::BaseNode* child  = root->createChild("child").get();
        child->addObject(core::objectmodel::New<MechanicalObject3>());

        sofa::simulation::node::initRoot(root.get());

        {
            const simulation::Node::SPtr nodeToRemove = static_cast<simulation::Node*>(child);
            nodeToRemove->detachFromGraph();
            nodeToRemove->execute<simulation::DeleteVisitor>(sofa::core::execparams::defaultInstance());
        }

        sofa::simulation::node::animate(root.get());
        sofa::simulation::node::unload(root);
    }

    /// create and unload a scene and check if all the objects have been destroyed.
    void sceneDestruction_unload()
    {
        createScene();
        sofa::simulation::node::unload(root);
        checkDeletions();
    }

    /// create and replace a scene and check if all the objects have been destroyed.
    void sceneDestruction_createnewgraph()
    {
        createScene();
        root = simulation::MainSimulation::getSimulation()->createNewGraph("root2");
        checkDeletions();
    }

    /// create a new scene and reset the root
    void sceneDestruction_reset()
    {
        createScene();
        root.reset();
        checkDeletions();
    }

    /// create a new scene and set the root to nullptr
    void sceneDestruction_setNull()
    {
        createScene();
        root = nullptr;
        checkDeletions();
    }

protected:
    void createScene()
    {
        typedef component::mass::UniformMass<defaulttype::Vec3Types> UniformMass3;

        objectCounter = 0;

        root = simulation::MainSimulation::getSimulation()->createNewGraph("root");
        root->addObject(core::objectmodel::New<InstrumentedObject<MechanicalObject3> >());

        typename UniformMass3::SPtr uniformMass = core::objectmodel::New<InstrumentedObject<UniformMass3>>();
        uniformMass->d_totalMass.setValue(1.0);
        root->addObject(uniformMass);

        const simulation::Node::SPtr child  = simulation::MainSimulation::getSimulation()->createNewNode("child");
        root->addChild(child);
        child->addObject(core::objectmodel::New<InstrumentedObject<MechanicalObject3> >());

    }
    void checkDeletions() {
        if(objectCounter>0)
            ADD_FAILURE() << objectCounter << " objects not deleted " <<std::endl;
    }
};

// run the tests
TEST_F( Scene_test,computeBBox) {
    EXPECT_MSG_NOEMIT(Error) ;
    this->computeBBox();
}

// component destruction
TEST_F( Scene_test,objectDestruction_replace) {
    EXPECT_MSG_NOEMIT(Error) ;
    this->objectDestruction_replace();
}

TEST_F( Scene_test,objectDestruction_delete) {
    EXPECT_MSG_NOEMIT(Error) ;
    this->objectDestruction_delete(); checkDeletions();
}

TEST_F( Scene_test,objectDestruction_setNull) {
    EXPECT_MSG_NOEMIT(Error) ;
    this->objectDestruction_setNull();
}

TEST_F( Scene_test,objectDestruction_reset) {
    EXPECT_MSG_NOEMIT(Error) ;
    this->objectDestruction_reset();
}

TEST_F( Scene_test,objectDestruction_subObject) {
    EXPECT_MSG_NOEMIT(Error) ;
    this->objectDestruction_subObject(); checkDeletions();
}

TEST_F( Scene_test,objectDestruction_subNodeAndStep) {
    EXPECT_MSG_NOEMIT(Error) ;
    this->objectDestruction_subNodeAndStep();
}

// graph destruction
TEST_F( Scene_test,sceneDestruction_unload) {
    EXPECT_MSG_NOEMIT(Error) ;
    this->sceneDestruction_unload();
}

TEST_F( Scene_test,sceneDestruction_createnewgraph) {
    EXPECT_MSG_NOEMIT(Error) ;
    this->sceneDestruction_createnewgraph();
}

// node bounding box: computed from the objects, unless fixed by the user

namespace
{
using sofa::type::BoundingBox;
using sofa::type::Vec3;

/// Two points; the state is drawn, the condition for a MechanicalObject to
/// take part in the bounding box.
MechanicalObject3::SPtr addPoints(simulation::Node* node, const Vec3& a, const Vec3& b)
{
    const MechanicalObject3::SPtr dofs = core::objectmodel::New<MechanicalObject3>();
    dofs->showObject.setValue(true);
    node->addObject(dofs);
    dofs->resize(2);
    MechanicalObject3::WriteVecCoord x = dofs->writePositions();
    x[0] = a;
    x[1] = b;
    return dofs;
}

void movePoints(MechanicalObject3* dofs, const Vec3& a, const Vec3& b)
{
    MechanicalObject3::WriteVecCoord x = dofs->writePositions();
    x[0] = a;
    x[1] = b;
}

/// What the animation loop runs after each step.
void updateBBox(simulation::Node* root)
{
    root->execute<simulation::UpdateBoundingBoxVisitor>(core::execparams::defaultInstance());
}

void expectBBox(const simulation::Node* node, const Vec3& expectedMin, const Vec3& expectedMax)
{
    const BoundingBox& bbox = node->f_bbox.getValue();
    EXPECT_TRUE(bbox.isValid()) << "node " << node->getName();
    EXPECT_EQ(bbox.minBBox(), expectedMin) << "node " << node->getName();
    EXPECT_EQ(bbox.maxBBox(), expectedMax) << "node " << node->getName();
}

simulation::Node::SPtr createRoot()
{
    return simulation::MainSimulation::getSimulation()->createNewGraph("root");
}
} // namespace

/// No bbox given: computed at init, and recomputed when the objects move.
/// Writing the computed value must not turn it into a fixed bbox.
TEST(NodeBBox_test, computedBBoxFollowsObjects)
{
    EXPECT_MSG_NOEMIT(Error);
    const simulation::Node::SPtr root = createRoot();
    const auto dofs = addPoints(root.get(), Vec3(0, 0, 0), Vec3(1, 1, 1));

    simulation::node::initRoot(root.get());
    EXPECT_FALSE(root->bboxIsFixed());
    expectBBox(root.get(), Vec3(0, 0, 0), Vec3(1, 1, 1));

    movePoints(dofs.get(), Vec3(5, 5, 5), Vec3(6, 6, 6));
    updateBBox(root.get());
    EXPECT_FALSE(root->bboxIsFixed());
    expectBBox(root.get(), Vec3(5, 5, 5), Vec3(6, 6, 6));
}

/// bbox written after the node creation and before init, e.g. `root.bbox = ...`
/// in a Python createScene: kept by init and by the updates.
TEST(NodeBBox_test, bboxSetBeforeInitIsKept)
{
    EXPECT_MSG_NOEMIT(Error);
    const simulation::Node::SPtr root = createRoot();
    const auto dofs = addPoints(root.get(), Vec3(0, 0, 0), Vec3(1, 1, 1));
    root->f_bbox.setValue(BoundingBox(Vec3(-0.5, -0.5, -0.5), Vec3(0.5, 0.5, 0.5)));
    EXPECT_TRUE(root->bboxIsFixed());

    simulation::node::initRoot(root.get());
    expectBBox(root.get(), Vec3(-0.5, -0.5, -0.5), Vec3(0.5, 0.5, 0.5));

    movePoints(dofs.get(), Vec3(5, 5, 5), Vec3(6, 6, 6));
    updateBBox(root.get());
    EXPECT_TRUE(root->bboxIsFixed());
    expectBBox(root.get(), Vec3(-0.5, -0.5, -0.5), Vec3(0.5, 0.5, 0.5));
}

/// bbox written while the simulation runs (e.g. from a controller): from then on
/// it is fixed.
TEST(NodeBBox_test, bboxSetAfterInitIsKept)
{
    EXPECT_MSG_NOEMIT(Error);
    const simulation::Node::SPtr root = createRoot();
    const auto dofs = addPoints(root.get(), Vec3(0, 0, 0), Vec3(1, 1, 1));
    simulation::node::initRoot(root.get());
    expectBBox(root.get(), Vec3(0, 0, 0), Vec3(1, 1, 1));

    root->f_bbox.setValue(BoundingBox(Vec3(-2, -2, -2), Vec3(2, 2, 2)));
    EXPECT_TRUE(root->bboxIsFixed());
    updateBBox(root.get());
    expectBBox(root.get(), Vec3(-2, -2, -2), Vec3(2, 2, 2));

    movePoints(dofs.get(), Vec3(5, 5, 5), Vec3(6, 6, 6));
    updateBBox(root.get());
    expectBBox(root.get(), Vec3(-2, -2, -2), Vec3(2, 2, 2));
}

/// bbox given as a scene attribute (what the XML loader and addChild(name, bbox=...)
/// do through Node::parse): the node keeps it, and the computed bbox of its parent
/// includes it along with the computed bbox of its siblings.
TEST(NodeBBox_test, bboxFromSceneAttributeIsKeptAndIncludedInParent)
{
    EXPECT_MSG_NOEMIT(Error);
    const simulation::Node::SPtr root = createRoot();

    const simulation::Node::SPtr fixedChild = root->createChild("fixedChild");
    core::objectmodel::BaseObjectDescription desc("fixedChild", "Node");
    desc.setAttribute("bbox", "-3 -3 -3 3 3 3");
    fixedChild->parse(&desc);
    const auto fixedDofs = addPoints(fixedChild.get(), Vec3(0, 0, 0), Vec3(1, 1, 1));

    const simulation::Node::SPtr computedChild = root->createChild("computedChild");
    const auto computedDofs = addPoints(computedChild.get(), Vec3(10, 10, 10), Vec3(11, 11, 11));

    simulation::node::initRoot(root.get());
    EXPECT_TRUE(fixedChild->bboxIsFixed());
    EXPECT_FALSE(computedChild->bboxIsFixed());
    EXPECT_FALSE(root->bboxIsFixed());
    expectBBox(fixedChild.get(), Vec3(-3, -3, -3), Vec3(3, 3, 3));
    expectBBox(computedChild.get(), Vec3(10, 10, 10), Vec3(11, 11, 11));
    expectBBox(root.get(), Vec3(-3, -3, -3), Vec3(11, 11, 11));

    movePoints(fixedDofs.get(), Vec3(20, 20, 20), Vec3(21, 21, 21));
    movePoints(computedDofs.get(), Vec3(4, 4, 4), Vec3(5, 5, 5));
    updateBBox(root.get());
    expectBBox(fixedChild.get(), Vec3(-3, -3, -3), Vec3(3, 3, 3));
    expectBBox(computedChild.get(), Vec3(4, 4, 4), Vec3(5, 5, 5));
    expectBBox(root.get(), Vec3(-3, -3, -3), Vec3(5, 5, 5));
}

}// namespace sofa







