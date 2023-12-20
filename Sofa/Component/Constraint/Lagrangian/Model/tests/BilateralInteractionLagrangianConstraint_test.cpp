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

#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/defaulttype/VecTypes.h>
#include <SofaBaseTopology/PointSetTopologyContainer.h>
#include <SofaConstraint/BilateralInteractionLangrangianConstraint.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/MechanicalParams.h>
using sofa::core::execparams::defaultInstance; 

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/testing/NumericTest.h>
using sofa::testing::NumericTest;
#include <sofa/simulation/Node.h>
#include <SofaSimulationGraph/SimpleApi.h>
#include <SofaSimulationCommon/SceneLoaderXML.h>
#include <sofa/helper/logging/Message.h>

#include <SofaConstraint/GenericConstraintSolver.h>

namespace sofa {

namespace {

using sofa::simulation::Node ;
using sofa::core::ExecParams ;
using sofa::simulation::SceneLoaderXML ;

using namespace component;
using namespace defaulttype;

template <typename _DataTypes>
struct BilateralInteractionLangrangianConstraint_test : public NumericTest<>
{
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::CPos CPos;
    typedef typename Coord::value_type Real;
    typedef constraintset::BilateralInteractionLangrangianConstraint<DataTypes> BilateralInteractionLangrangianConstraint;
    typedef component::topology::PointSetTopologyContainer PointSetTopologyContainer;
    typedef container::MechanicalObject<DataTypes> MechanicalObject;

    simulation::Node::SPtr root;                 ///< Root of the scene graph, created by the constructor an re-used in the tests
    simulation::Simulation* simulation {nullptr}; ///< created by the constructor an re-used in the tests

    /// Create the context for the tests.
    void SetUp() override
    {
        sofa::simpleapi::importPlugin("SofaComponentAll");
        sofa::simpleapi::importPlugin("SofaMiscCollision");
        sofa::simpleapi::importPlugin("SofaOpenglVisual");
        if(simulation==nullptr)
            sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());
    }

    void TearDown() override
    {
        if(root)
            simulation->unload(root);
    }

    void init_Vec3Setup()
    {
    }

    bool test_Vec3ConstrainedPositions()
    {
        return true;
    }

    /// It is important to freeze what are the available Data field
    /// of a component and rise warning/errors when some are removed.
    /// If you remove/renamed a data field please add a deprecation
    /// message as well as update this test.
    void attributesTests(){
        /// I'm using '\n' so that the XML parser correctly report the line number
        /// in case of problems.
        std::stringstream scene;
        scene << "<?xml version='1.0'?>                                       \n"
                 "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > \n"
                 "   <MechanicalObject name='o1' template='"<< DataTypes::Name() << "' position='1 2 3'/>   \n"
                 "   <MechanicalObject name='o2' template='"<< DataTypes::Name() << "' position='1 2 3'/>   \n"
                 "   <BilateralInteractionLangrangianConstraint template='"<< DataTypes::Name() << "' object1='@./o1' object2='@./o2'/>     \n"
                 "</Node>                                                     \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene.str().c_str(),
                                                          scene.str().size()) ;
        root->init(sofa::core::execparams::defaultInstance()) ;

        BilateralInteractionLangrangianConstraint* constraint = root->getTreeObject<BilateralInteractionLangrangianConstraint>() ;
        EXPECT_TRUE( constraint != nullptr ) ;

        EXPECT_TRUE( constraint->findData("first_point") != nullptr ) ;
        EXPECT_TRUE( constraint->findData("second_point") != nullptr ) ;
        EXPECT_TRUE( constraint->findData("rest_vector") != nullptr ) ;
        EXPECT_TRUE( constraint->findData("activateAtIteration") != nullptr ) ;
        EXPECT_TRUE( constraint->findData("numericalTolerance") != nullptr ) ;

        EXPECT_TRUE( constraint->findData("merge") != nullptr ) ;
        EXPECT_TRUE( constraint->findData("derivative") != nullptr ) ;
        return ;
    }


    /// This component requires to be used in conjonction with MechanicalObjects.
    void checkMstateRequiredAssumption(){
        EXPECT_MSG_EMIT(Error) ;

        /// I'm using '\n' so that the XML parser correctly report the line number
        /// in case of problems.
        std::stringstream scene;
        scene << "<?xml version='1.0'?>                                       \n"
                 "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > \n"
                 "   <BilateralInteractionLangrangianConstraint template='"<< DataTypes::Name() << "' />     \n"
                 "</Node>                                                     \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene.str().c_str(),
                                                          scene.str().size()) ;
        root->init(sofa::core::execparams::defaultInstance()) ;

        return ;
    }

    void checkRigid3fFixForBackwardCompatibility(){}
 };

template<>
void BilateralInteractionLangrangianConstraint_test<Rigid3fTypes>::checkRigid3fFixForBackwardCompatibility(){
    EXPECT_MSG_EMIT(Warning) ;

    /// I'm using '\n' so that the XML parser correctly report the line number
    /// in case of problems.
    std::stringstream scene;
    scene << "<?xml version='1.0'?>                                       \n"
             "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > \n"
             "   <MechanicalObject name='o1' template='"<< DataTypes::Name() << "' position='1 2 3'/>   \n"
             "   <MechanicalObject name='o2' template='"<< DataTypes::Name() << "' position='1 2 3'/>   \n"
             "   <BilateralInteractionLangrangianConstraint template='"<< DataTypes::Name() << "' object1='@./o1' object2='@./o2'/>     \n"
             "</Node>                                                     \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

}


template<>
void BilateralInteractionLangrangianConstraint_test<Vec3Types>::init_Vec3Setup()
{
    /// Load the scene
    //TODO(dmarchal): This general load should be updated... there is no reason to load
    // a scene independently of the data template to use.
    std::string sceneName = "BilateralInteractionLangrangianConstraint.scn";
    std::string fileName  = std::string(SOFATEST_SCENES_DIR) + "/" + sceneName;
    root = sofa::simulation::getSimulation()->load(fileName.c_str()).get();

    //TODO(dmarchal): I'm very surprised that scene.loadSucceed could contain
    // a state about the load results that happens "before" maybe a side effect
    // of the static variable.

    // Test if load has succeeded
    sofa::simulation::SceneLoaderXML scene;

    if(!root || !scene.loadSucceed)
    {
        ADD_FAILURE() << "Error while loading the scene: " << sceneName << std::endl;
    }


    /// Init
    sofa::simulation::getSimulation()->init(root.get());
}

template<>
bool BilateralInteractionLangrangianConstraint_test<Vec3Types>::test_Vec3ConstrainedPositions()
{
    std::vector<MechanicalObject*> meca;
    root->get<MechanicalObject>(&meca,std::string("mecaConstraint"),root->SearchDown);

    std::vector<Coord> points;
    points.resize(2);

    if(meca.size()!=2)
    {
        ADD_FAILURE() << "Error while searching mechanical object" << std::endl;
    }

    for(int i=0; i<10; i++)
        sofa::simulation::getSimulation()->animate(root.get(),(double)0.001);

    if(meca.size()==2)
    {
        for(unsigned int i=0; i<meca.size(); i++)
            points[i] = meca[i]->read(core::ConstVecCoordId::position())->getValue()[0];
    }

    component::constraintset::GenericConstraintSolver *test;
    root->get(test);
    if(vectorMaxDiff(points[0],points[1])<test->tolerance.getValue()) return true;
    else
    {
        ADD_FAILURE() << "Error while testing if two positions are correctly constrained" << std::endl;
    }

    return false;
}

// Define the list of DataTypes to instanciate
using ::testing::Types;
typedef Types<Vec3Types
              ,Rigid3dTypes
 //
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_SUITE(BilateralInteractionLangrangianConstraint_test, DataTypes);

//TODO(dmarchal): Needs a serious refactor !!!
TYPED_TEST( BilateralInteractionLangrangianConstraint_test , checkVec3ConstrainedPositions )
{
    this->init_Vec3Setup();
    ASSERT_TRUE(  this->test_Vec3ConstrainedPositions() );
}


TYPED_TEST( BilateralInteractionLangrangianConstraint_test , attributesTests )
{
    ASSERT_NO_THROW(  this->attributesTests() );
}

TYPED_TEST( BilateralInteractionLangrangianConstraint_test , checkMstateRequiredAssumption )
{
    ASSERT_NO_THROW(  this->checkMstateRequiredAssumption() );
}

TYPED_TEST( BilateralInteractionLangrangianConstraint_test ,  checkRigid3fFixForBackwardCompatibility)
{
    ASSERT_NO_THROW(  this->checkRigid3fFixForBackwardCompatibility() );
}


}

} // namespace sofa

