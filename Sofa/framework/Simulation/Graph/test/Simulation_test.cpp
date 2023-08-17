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

#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/DeleteVisitor.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/component/mass/UniformMass.h>

#include <sofa/simulation/DefaultAnimationLoop.h>

#include <sofa/simulation/Node.h>

namespace sofa {

static int objectCounter;

/// Component class with an instance counter
template <class Object>
struct InstrumentedObject : public Object
{
    InstrumentedObject()  { objectCounter++; }
    ~InstrumentedObject() { objectCounter--; }
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
        simulation = sofa::simulation::getSimulation();
    }

    /// Test Simulation::computeBBox
    void computeBBox()
    {
        // Init Sofa
        root = simulation::getSimulation()->createNewGraph("root");

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
        root = simulation::getSimulation()->createNewGraph("root");
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
        root = simulation::getSimulation()->createNewGraph("root2");
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

        root = simulation::getSimulation()->createNewGraph("root");
        root->addObject(core::objectmodel::New<InstrumentedObject<MechanicalObject3> >());
        root->addObject(core::objectmodel::New<InstrumentedObject<UniformMass3> >());
        const simulation::Node::SPtr child  = simulation::getSimulation()->createNewNode("child");
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

}// namespace sofa







