/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

#include <map>
using std::map ;
using std::pair ;

#include <SofaTest/Sofa_test.h>
using sofa::Sofa_test;

#include <SceneCreator/SceneCreator.h>

#include <sofa/simulation/Simulation.h>
using sofa::simulation::Simulation ;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;
using sofa::simulation::Node ;

#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
using namespace sofa::defaulttype;

#include <SofaBaseMechanics/MechanicalObject.h>
using sofa::component::container::MechanicalObject ;

#include <SofaBoundaryCondition/ConstantForceField.h>
using sofa::component::forcefield::ConstantForceField ;
using sofa::core::ExecParams ;


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
    typedef MechanicalObject<DataTypes>   TheMechanicalObject;

    void SetUp() {}
    void TearDown(){}

    void testSimpleBehavior()
    {
        EXPECT_MSG_NOEMIT(Error) ;

        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node 	name='Root' gravity='-9.81 0 0' time='0' animate='0' >               \n"
                 "   <DefaultAnimationLoop/>                                                     \n"
                 "   <CGLinearSolver/>                                                           \n"
                 "   <EulerImplicitSolver/>                                                      \n"
                 "   <MechanicalObject name='mstate' size='2' template='"<<  DataTypes::Name() << "'/> \n"
                 "   <UniformMass/>                                                                    \n"
                 "   <ConstantForceField name='myForceField' indices='0' force='100.0 0.0 0'/>         \n"
                 "</Node>                                                                                                                                                               \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene.str().c_str(),
                                                          scene.str().size()) ;

        EXPECT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        TheMechanicalObject* mechanicalobject ;
        root->getTreeObject(mechanicalobject) ;
        ASSERT_NE(nullptr, mechanicalobject) ;

        TheConstantForceField* forcefield ;
        root->getTreeObject(forcefield) ;
        ASSERT_NE(nullptr, forcefield) ;

        Simulation* simulation = sofa::simulation::getSimulation() ;
        ASSERT_NE(nullptr, simulation) ;

        Real xi = mechanicalobject->x.getValue()[0][0];
        Real ei = mechanicalobject->x.getValue()[1][0];
        EXPECT_GT(xi, -0.1) << "Initialization problem...before simulation first value should be 0";
        EXPECT_GT(ei, -0.1) << "Initialization problem...before simulation second value should be 0>";
        for(int i=0; i<100; i++)
        {
            simulation->animate(root.get(),(double)0.01);
        }
        Real xe=mechanicalobject->x.getValue()[0][0];
        Real ee = mechanicalobject->x.getValue()[1][0];
        EXPECT_GT(xe, -0.1) << "Simulation problem...after simulation the particle should not have fallen.";
        EXPECT_GT(xe, xi) << "Simulation problem...after simulation the particle should be higher than initial position.";

        EXPECT_LT(ee, -0.1) << "Simulation problem...after simulation the particle should have fallen.";
    }

    void testMonkeyValueForIndices()
    {
        map<string, vector< pair<string, string> >> values =
        {
            {"indices",   { {"0 1","0 1"}, {"1 0", "1 0"}, {"-1 5", "0 5"} } }
        };

        for(auto& kv : values){
            for(auto& v : kv.second){
                std::stringstream scene ;
                scene << "<?xml version='1.0'?>"
                         "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                         "   <DefaultAnimationLoop/>                                                     \n"
                         "   <MechanicalObject name='mstate' template='"<<  DataTypes::Name() << "'/>    \n"
                         "   <ConstantForceField name='myForceField' "<< kv.first << "='"<< v.first << "'/>  \n"
                         "</Node>                                                                        \n" ;

                Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                                  scene.str().c_str(),
                                                                  scene.str().size()) ;
                ASSERT_NE(root.get(), nullptr) << "Problem to load scene: " << scene.str() ;
                root->init(ExecParams::defaultInstance()) ;

                sofa::core::objectmodel::BaseObject* constantff = root->getObject("myForceField") ;
                ASSERT_NE( constantff, nullptr) ;

                ASSERT_NE( nullptr, constantff->findData(kv.first) ) << "Missing parameter '" << kv.first << "'";

                EXPECT_STREQ(  v.second.c_str(), constantff->findData(kv.first)->getValueString().c_str() )
                        << "When the attribute '"<<kv.first<< "' is set to the value '" << v.first.c_str()
                        << "' it should be corrected during the component init to the valid value '" << v.second.c_str() << "'."
                        << " If this is not the case this means that the init function is not working properly (or the default "
                        << "value have changed and thus the test need to be fixed)";
            }
        }
    }


    void testBasicAttributes()
    {
        EXPECT_MSG_NOEMIT(Error) ;

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
        EXPECT_MSG_EMIT(Error) ;

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
    ASSERT_NO_THROW (this->testMissingMechanicalObject());
}


TYPED_TEST( ConstantForceField_test , testSimpleBehavior )
{
    ASSERT_NO_THROW (this->testSimpleBehavior());
}

TYPED_TEST( ConstantForceField_test , testMonkeyValueForIndices )
{
    ASSERT_NO_THROW (this->testMonkeyValueForIndices());
}






