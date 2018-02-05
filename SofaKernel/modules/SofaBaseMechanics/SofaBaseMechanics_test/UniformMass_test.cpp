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
#include <SofaBaseMechanics/UniformMass.h>

#include <string>
using std::string ;

#include <gtest/gtest.h>
using testing::Types;

#include <sofa/helper/BackTrace.h>
#include <SofaBaseMechanics/MechanicalObject.h>
using namespace sofa::defaulttype ;

#include <SofaBaseMechanics/UniformMass.h>
using sofa::component::mass::UniformMass ;

#include <SofaBaseMechanics/initBaseMechanics.h>
using sofa::component::initBaseMechanics ;

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::Simulation ;
using sofa::simulation::graph::DAGSimulation ;
using sofa::simulation::Node ;
using sofa::simulation::setSimulation ;
using sofa::core::objectmodel::New ;
using sofa::core::objectmodel::BaseData ;
using sofa::core::ExecParams ;
using sofa::component::container::MechanicalObject ;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

template <class TDataTypes, class TMassTypes>
struct TemplateTypes
{
    typedef TDataTypes DataTypes ;
    typedef TMassTypes MassTypes ;
};

template <typename TTemplateTypes>
struct UniformMassTest :  public ::testing::Test
{
    typedef UniformMass<typename TTemplateTypes::DataTypes,
    typename TTemplateTypes::MassTypes> TheUniformMass ;
    typedef UniformMass<Rigid3Types, Rigid3Mass> UniformMassRigid;

    /// Bring parents members in the current lookup context.
    /// more info at: https://gcc.gnu.org/onlinedocs/gcc/Name-lookup.html
    typedef typename TTemplateTypes::DataTypes DataTypes ;
    typedef typename TTemplateTypes::MassTypes MassTypes ;

    Simulation* m_simu  {nullptr} ;
    Node::SPtr m_root ;
    Node::SPtr m_node ;
    typename TheUniformMass::SPtr m_mass ;
    typename MechanicalObject<DataTypes>::SPtr m_mecaobject;
    bool todo  {true} ;

    virtual void SetUp()
    {
        todo = true ;
        initBaseMechanics();
        setSimulation( m_simu = new DAGSimulation() );
        m_root = m_simu->createNewGraph("root");
    }

    void TearDown()
    {
        if (m_root != NULL){
            m_simu->unload(m_root);
        }
    }

    /// It is important to freeze what are the available Data field
    /// of a component and rise warning/errors when some one removed.
    ///
    void attributesTests(){
        m_node = m_root->createChild("node") ;
        m_mass = New< TheUniformMass >() ;
        m_node->addObject(m_mass) ;

        EXPECT_TRUE( m_mass->findData("mass") != nullptr ) ;
        EXPECT_TRUE( m_mass->findData("totalmass") != nullptr ) ;
        EXPECT_TRUE( m_mass->findData("filename") != nullptr ) ;
        EXPECT_TRUE( m_mass->findData("localRange") != nullptr ) ;

        EXPECT_TRUE( m_mass->findData("showGravityCenter") != nullptr ) ;
        EXPECT_TRUE( m_mass->findData("showAxisSizeFactor") != nullptr ) ;
        EXPECT_TRUE( m_mass->findData("showInitialCenterOfGravity") != nullptr ) ;

        EXPECT_TRUE( m_mass->findData("indices") != nullptr ) ;
        EXPECT_TRUE( m_mass->findData("handleTopoChange") != nullptr ) ;
        EXPECT_TRUE( m_mass->findData("preserveTotalMass") != nullptr ) ;

        EXPECT_TRUE( m_mass->findData("compute_mapping_inertia") != nullptr ) ;
        EXPECT_TRUE( m_mass->findData("totalMass") != nullptr ) ;
        return ;
    }

    /// totalMass, mass and localRange..
    /// totalMass & mass are exclusive.
    /// si mass and total mass set c'est total mass le plus fort.
    void checkDefaultValuesForAttributes(){
        string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "   <MechanicalObject position='0 0 0 4 5 6'/>               "
                "   <UniformMass name='m_mass'/>                             "
                "</Node>                                                     " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        root->init(ExecParams::defaultInstance()) ;

        TheUniformMass* mass = root->getTreeObject<TheUniformMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMass(), 1.0 ) ;
            EXPECT_EQ( mass->getTotalMass(), 2.0 ) ;
        }
    }

    /// totalMass, mass and localRange..
    void checkMassTotalFromMass(){
        string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "   <MechanicalObject position='0 0 0 4 5 6'/>               "
                "   <UniformMass name='m_mass' mass='4.0' />                 "
                "</Node>                                                     " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          (int)scene.size()) ;

        root->init(ExecParams::defaultInstance()) ;

        TheUniformMass* mass = root->getTreeObject<TheUniformMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMass(), 4.0 ) ;
            EXPECT_EQ( mass->getTotalMass(), 8.0 ) ;
        }
    }

    /// totalMass, mass and localRange..
    void checkMassFromMassTotal(){
        string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "   <MechanicalObject position='0 0 0 4 5 6'/>               "
                "   <UniformMass name='m_mass' totalmass='4.0' />            "
                "</Node>                                                     " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          (int)scene.size()) ;

        root->init(ExecParams::defaultInstance()) ;

        TheUniformMass* mass = root->getTreeObject<TheUniformMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMass(), 2.0 ) ;
            EXPECT_EQ( mass->getTotalMass(), 4.0 ) ;
        }
    }

    /// totalMass, mass and localRange..
    void checkMassAndMassTotal(){
        string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "   <MechanicalObject position='0 0 0 4 5 6'/>               "
                "   <UniformMass name='m_mass' totalmass='91.0' mass=2.0/>   "
                "</Node>                                                     " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          (int)scene.size()) ;

        root->init(ExecParams::defaultInstance()) ;

        TheUniformMass* mass = root->getTreeObject<TheUniformMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMass(), 45.5 ) ;
            EXPECT_EQ( mass->getTotalMass(), 91.0 ) ;
        }
    }

    /// if masses are negative we refuse them and use the default values.
    void checkNegativeMassNotAllowed(){
        string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "   <MechanicalObject position='0 0 0 4 5 6'/>               "
                "   <UniformMass name='m_mass' totalmass='-1.0' mass=-3.0/>   "
                "</Node>                                                     " ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadWithNoParam",
                                                          scene.c_str(),
                                                          scene.size()) ;

        root->init(ExecParams::defaultInstance()) ;

        TheUniformMass* mass = root->getTreeObject<TheUniformMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMass(), 1.0 ) ;
            EXPECT_EQ( mass->getTotalMass(), 2.0 ) ;
        }
    }

    void loadFromAFileForRigid(){
        string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "   <MechanicalObject template='Rigid3' position='0 0 0 1 0 0 1 0 0 0 1 0 0 1'/>                     "
                "   <UniformMass filename='BehaviorModels/card.rigid'/>        "
                "</Node>                                                     " ;
        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadFromAValidFile",
                                                          scene.c_str(), (int)scene.size()) ;
        root->init(ExecParams::defaultInstance()) ;

        UniformMassRigid* mass = root->getTreeObject<UniformMassRigid>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getMass(), 40.0 ) ;
            EXPECT_EQ( mass->getTotalMass(), 80.0 ) ;
        }
    }

    void loadFromAFileForNonRigid(){
        string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "   <MechanicalObject position='0 0 0'/>                     "
                "   <UniformMass filename='BehaviorModels/card.rigid'/>        "
                "</Node>                                                     " ;
        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadFromAValidFile",
                                                          scene.c_str(), (int)scene.size()) ;
        root->init(ExecParams::defaultInstance()) ;
    }

    void loadFromAnInvalidFile(){
        string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "   <MechanicalObject position='0 0 0'/>                     "
                "   <UniformMass filename='invalid_uniformmatrix.txt'/>        "
                "</Node>                                                     " ;
        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadFromAnInValidFile",
                                                          scene.c_str(), (int)scene.size()) ;
        root->init(ExecParams::defaultInstance()) ;
    }

    void loadFromAnInvalidPathname(){
        string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "   <MechanicalObject position='0 0 0'/>                     "
                "   <UniformMass filename='invalid_uniformmatrix.txt'/>        "
                "</Node>                                                     " ;
        Node::SPtr root = SceneLoaderXML::loadFromMemory ("loadFromAnInValidFile",
                                                          scene.c_str(), (int)scene.size()) ;
        root->init(ExecParams::defaultInstance()) ;
    }

    void reinitTest(){
        // TODO
        EXPECT_TRUE(todo == false) ;
    }

};


typedef Types<
TemplateTypes<Vec3Types, Vec3Types::Real>
/*#ifdef SOFA_WITH_DOUBLE
,TemplateTypes<Vec3dTypes, Vec3dTypes::Real>
,TemplateTypes<Vec2dTypes, Vec2dTypes::Real>
,TemplateTypes<Vec1dTypes, Vec1dTypes::Real>
,TemplateTypes<Vec6dTypes, Vec6dTypes::Real>
,TemplateTypes<Rigid3dTypes, Rigid3dMass>
,TemplateTypes<Rigid2dTypes, Rigid2dMass>
#endif
#ifdef SOFA_WITH_FLOAT
,TemplateTypes<Vec3dTypes, Vec3dTypes::Real>
,TemplateTypes<Vec2dTypes, Vec2dTypes::Real>
,TemplateTypes<Vec1dTypes, Vec1dTypes::Real>
,TemplateTypes<Vec6dTypes, Vec6dTypes::Real>
,TemplateTypes<Rigid3dTypes, Rigid3dMass>
,TemplateTypes<Rigid2dTypes, Rigid2dMass>
#endif*/
> DataTypes;

TYPED_TEST_CASE(UniformMassTest, DataTypes);


TYPED_TEST(UniformMassTest, attributesTests) {
    ASSERT_NO_THROW(this->attributesTests()) ;
}

TYPED_TEST(UniformMassTest, checkMassTotalFromMass)
{
    ASSERT_NO_THROW(this->checkMassTotalFromMass()) ;
}

TYPED_TEST(UniformMassTest, checkMassFromMassTotal)
{
    ASSERT_NO_THROW(this->checkMassFromMassTotal()) ;
}

TYPED_TEST(UniformMassTest, checkMassAndMassTotal)
{
    ASSERT_NO_THROW(this->checkMassAndMassTotal()) ;
}

TYPED_TEST(UniformMassTest, checkNegativeMassNotAllowed)
{
    ASSERT_NO_THROW(this->checkNegativeMassNotAllowed()) ;
}

TYPED_TEST(UniformMassTest, checkDefaultValuesForAttributes) {
    ASSERT_NO_THROW(this->checkDefaultValuesForAttributes()) ;
}

TYPED_TEST(UniformMassTest, loadFromAFileForNonRigid) {
    ASSERT_NO_THROW(this->loadFromAFileForNonRigid()) ;
}

TYPED_TEST(UniformMassTest, loadFromAnInvalidFile) {
    ASSERT_NO_THROW(this->loadFromAnInvalidFile()) ;
}

TYPED_TEST(UniformMassTest, loadFromAnInvalidPathname) {
    ASSERT_NO_THROW(this->loadFromAnInvalidPathname()) ;
}

TYPED_TEST(UniformMassTest, loadFromAFileForRigid) {
    ASSERT_NO_THROW(this->loadFromAFileForRigid()) ;
}

TYPED_TEST(UniformMassTest, reinitTest) {
    //ASSERT_NO_THROW(this->reinitTest()) ;
}

