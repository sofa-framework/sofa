/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <string>
using std::string ;

#include <vector>
using std::vector ;

#include <SofaTest/Sofa_test.h>
using sofa::Sofa_test;

#include <SceneCreator/SceneCreator.h>

#include <SofaTest/TestMessageHandler.h>
using sofa::helper::logging::Message ;
using sofa::helper::logging::ExpectMessage ;
using sofa::helper::logging::MessageAsTestFailure;

#include <sofa/simulation/Simulation.h>
using sofa::simulation::Simulation ;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;
using sofa::simulation::Node ;

#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
using namespace sofa::defaulttype;

#include <SofaBoundaryCondition/ConstantForceField.h>
using sofa::component::forcefield::ConstantForceField ;
using sofa::core::ExecParams;


template <typename TDataType, typename TMassType>
struct TypeTuple
{
    typedef TDataType DataType ;
    typedef TMassType MassType ;
} ;

template <typename TTypeTuple>
struct ConstantForceField_test : public Sofa_test<>
{
    typedef typename TTypeTuple::DataType DataTypes ;
    typedef typename TTypeTuple::MassType MassType ;
    typedef ConstantForceField<DataTypes> TheConstantForceField ;

    void SetUp() {}
    void TearDown(){}

    void testSimpleBehavior()
    {
        MessageAsTestFailure raii(Message::Error);

        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "   <DefaultAnimationLoop/>                                                     \n"
                 "   <MechanicalObject name='mstate' template='"<<  DataTypes::Name() << "'/>    \n"
                 "   <ConstantForceField name='myForceField'/>                                   \n"
                 "</Node>                                                                        \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene.str().c_str(),
                                                          scene.str().size()) ;

        EXPECT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheConstantForceField* forcefield ;
        root->getTreeObject(forcefield) ;

        EXPECT_NE(nullptr, forcefield) ;

        Simulation* simulation = sofa::simulation::getSimulation() ;
        ASSERT_NE(nullptr, simulation) ;
        for(int i=0; i<100; i++){
            simulation->animate(root.get(),(double)0.01);
        }
    }

    void testBasicAttributes()
    {
        MessageAsTestFailure raii(Message::Error);

        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "   <DefaultAnimationLoop/>                                                     \n"
                 "   <MechanicalObject name='mstate' template='"<<  DataTypes::Name() << "'/>    \n"
                 "   <ConstantForceField name='myPlaneForceField'/>                              \n"
                 "</Node>                                                                        \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene.str().c_str(),
                                                          scene.str().size()) ;

        EXPECT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheConstantForceField* forcefield ;
        root->getTreeObject(forcefield) ;

        EXPECT_NE(nullptr, forcefield) ;

        /// List of the supported attributes the user expect to find
        /// This list needs to be updated if you add an attribute.
        vector<string> attrnames = {
            "points","forces","force","totalForce","arrowSizeCoef","showColor","indexFromEnd"
        };

        for(auto& attrname : attrnames)
            EXPECT_NE( nullptr, forcefield->findData(attrname) ) << "Missing attribute with name '"
                                                                 << attrname << "'." ;

        Simulation* simulation = sofa::simulation::getSimulation() ;
        ASSERT_NE(nullptr, simulation) ;
        for(int i=0; i<100; i++){
            simulation->animate(root.get(),(double)0.01);
        }

    }

    void testMissingMechanicalObject()
    {
        ExpectMessage raii(Message::Error);

        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "   <ConstantForceField name='myPlaneForceField'/>                              \n"
                 "</Node>                                                                        \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene.str().c_str(),
                                                          scene.str().size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;
    }
};

// Define the list of DataTypes to instanciate
using testing::Types;
typedef Types<
               TypeTuple<Rigid2Types, Rigid2Mass>
              ,TypeTuple<Rigid3Types, Rigid3Mass>
#ifdef SOFA_WITH_DOUBLE
              ,TypeTuple<Vec1dTypes, double>
              ,TypeTuple<Vec2dTypes, double>
              ,TypeTuple<Vec3dTypes, double>
              ,TypeTuple<Vec6dTypes, double>
              ,TypeTuple<Rigid3dTypes, Rigid3dMass>
              ,TypeTuple<Rigid2dTypes, Rigid2dMass>
#endif
#ifdef SOFA_WITH_FLOAT
             ,TypeTuple<Vec1fTypes, float>
             ,TypeTuple<Vec2fTypes, float>
             ,TypeTuple<Vec3fTypes, float>
             ,TypeTuple<Vec6fTypes, float>
             ,TypeTuple<Rigid3fTypes, Rigid3fMass>
             ,TypeTuple<Rigid2fTypes, Rigid2fMass>
#
#endif
> DataTypes;

// Test suite for all the instanciations
TYPED_TEST_CASE(ConstantForceField_test, DataTypes);// first test case
TYPED_TEST( ConstantForceField_test , testBasicAttributes )
{
    ASSERT_NO_THROW (this->testBasicAttributes());
}

TYPED_TEST( ConstantForceField_test , testMissingMechanicalObject )
{
    ASSERT_NO_THROW (this->testMissingMechanicalObject(););
}



