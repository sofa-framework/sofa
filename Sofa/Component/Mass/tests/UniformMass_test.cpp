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
#include <sofa/component/mass/UniformMass.h>

#include <string>
using std::string ;

#include <sofa/component/statecontainer/MechanicalObject.h>
using namespace sofa::defaulttype ;

using sofa::component::mass::UniformMass ;

#include <sofa/simpleapi/SimpleApi.h>

#include <sofa/simulation/graph/DAGSimulation.h>
using sofa::simulation::Simulation ;
using sofa::simulation::graph::DAGSimulation ;

#include <sofa/simulation/Node.h>
using sofa::simulation::Node;

using sofa::core::objectmodel::New ;
using sofa::core::objectmodel::BaseData ;
using sofa::core::ExecParams ;
using sofa::component::statecontainer::MechanicalObject ;

#include <sofa/simulation/common/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;
using testing::Types;

#include <sofa/core/ExecParams.h>
#include <sofa/core/VecId.h>

template <class TDataTypes, class TMassTypes>
struct TemplateTypes
{
    typedef TDataTypes DataTypes ;
    typedef TMassTypes MassTypes ;
};

template <typename TTemplateTypes>
struct UniformMassTest :  public BaseTest
{
    typedef UniformMass<typename TTemplateTypes::DataTypes> TheUniformMass ;
    typedef UniformMass<Rigid3Types> UniformMassRigid;

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

    void SetUp() override
    {
        todo = true ;
        m_simu = sofa::simulation::getSimulation();
        m_root = m_simu->createNewGraph("root");
    }

    void TearDown() override
    {
        if (m_root != nullptr){
            sofa::simulation::node::unload(m_root);
        }
    }

    /// It is important to freeze what are the available Data field
    /// of a component and rise warning/errors when some one removed.
    ///
    void attributesTests(){
        m_node = m_root->createChild("node") ;
        m_mass = New< TheUniformMass >() ;
        m_node->addObject(m_mass) ;

        EXPECT_TRUE( m_mass->findData("vertexMass") != nullptr ) ;
        EXPECT_TRUE( m_mass->findData("totalMass") != nullptr ) ;
        EXPECT_TRUE( m_mass->findData("filename") != nullptr ) ;
        EXPECT_TRUE( m_mass->findData("localRange") != nullptr ) ;

        EXPECT_TRUE( m_mass->findData("showGravityCenter") != nullptr ) ;
        EXPECT_TRUE( m_mass->findData("showAxisSizeFactor") != nullptr ) ;
        EXPECT_TRUE( m_mass->findData("showInitialCenterOfGravity") != nullptr ) ;

        EXPECT_TRUE( m_mass->findData("indices") != nullptr ) ;
        EXPECT_TRUE( m_mass->findData("preserveTotalMass") != nullptr ) ;

        EXPECT_TRUE( m_mass->findData("compute_mapping_inertia") != nullptr ) ;
        return ;
    }

    /// totalMass, mass and d_localRange..
    /// case where NO mass info give, default totalMass = 1.0
    void checkNoAttributes(){
        const string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "   <MechanicalObject position='0 0 0 4 5 6'/>               "
                "   <UniformMass name='m_mass'/>                             "
                "</Node>                                                     " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        root->init(sofa::core::execparams::defaultInstance()) ;

        TheUniformMass* mass = root->getTreeObject<TheUniformMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getTotalMass(), 1.0 ) ;
            EXPECT_EQ( mass->getVertexMass(), 0.5 ) ;
        }
    }

    /// totalMass, mass and d_localRange..
    /// case where NO mass info give, default totalMass = 1.0
    void checkRigidAttribute()
    {
        EXPECT_MSG_NOEMIT(Error, Warning);
        const string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "   <MechanicalObject template='Rigid3' position='0 0 0 0 0 0 1'/>"
                "   <UniformMass name='mass' vertexMass='1.0 1.0 2.0 0.0 0.0 0.0 4.0 0.0 7.0 8.0 9.0'/>"
                "</Node>                                                     " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        root->init(sofa::core::execparams::defaultInstance()) ;

        const UniformMassRigid* mass = root->getTreeObject<UniformMassRigid>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        const std::vector<double> values={2.0,0.0,0.0,0.0,4.0,0.0,7.0,8.0,9.0};
        for(unsigned int i=0;i<3;i++)
        {
            for(unsigned int j=0;j<3;j++)
            {
                ASSERT_EQ(mass->d_vertexMass.getValue().inertiaMatrix[i][j],
                          values[i*3+j]);
            }
        }
    }

    /// totalMass is well defined
    /// vertexMass will be computed from it using the formulat :  vertexMass = totalMass / number of particules
    void checkVertexMassFromTotalMass(){
        const string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "   <MechanicalObject position='0 0 0 4 5 6'/>               "
                "   <UniformMass name='m_mass' totalMass='8.0'/>                             "
                "</Node>                                                     " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        root->init(sofa::core::execparams::defaultInstance()) ;

        TheUniformMass* mass = root->getTreeObject<TheUniformMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getVertexMass(), 4.0 ) ;
            EXPECT_EQ( mass->getTotalMass(), 8.0 ) ;
        }
    }

    /// vertexMass is well defined
    /// totalMass will be computed from it using the formulat : totalMass = vertexMass * number of particules
    void checkTotalMassFromVertexMass(){
        const string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "   <MechanicalObject position='0 0 0 4 5 6'/>               "
                "   <UniformMass name='m_mass' vertexMass='4.0' />                 "
                "</Node>                                                     " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        root->init(sofa::core::execparams::defaultInstance()) ;

        TheUniformMass* mass = root->getTreeObject<TheUniformMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getVertexMass(), 4.0 ) ;
            EXPECT_EQ( mass->getTotalMass(), 8.0 ) ;
        }
    }

    /// totalMass is defined but negative
    /// Ignore value and use default value of totalMass = 1.0
    void checkNegativeTotalMass(){
        const string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "   <MechanicalObject position='0 0 0 4 5 6'/>               "
                "   <UniformMass name='m_mass' totalMass='-8.0' />                 "
                "</Node>                                                     " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        root->init(sofa::core::execparams::defaultInstance()) ;

        TheUniformMass* mass = root->getTreeObject<TheUniformMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getTotalMass(), 1.0 ) ;
            EXPECT_EQ( mass->getVertexMass(), 0.5 ) ;
        }
    }

    /// vertexMass is defined but negative
    /// Ignore value and use default value of totalMass = 1.0
    void checkNegativeVertexMass(){
        const string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "   <MechanicalObject position='0 0 0 4 5 6'/>               "
                "   <UniformMass name='m_mass' vertexMass='-4.0' />                 "
                "</Node>                                                     " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        root->init(sofa::core::execparams::defaultInstance()) ;

        TheUniformMass* mass = root->getTreeObject<TheUniformMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getTotalMass(), 1.0 ) ;
            EXPECT_EQ( mass->getVertexMass(), 0.5 ) ;
        }
    }

    /// totalMass & mass are exclusive.
    /// if both totalMass and vertexMass are user-defined, by default use the totalMass
    void checkDoubleDeclarationVertexAndTotalMass(){
        const string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "   <MechanicalObject position='0 0 0 4 5 6'/>               "
                "   <UniformMass name='m_mass' vertexMass='10.0' totalMass='8.0'/>                 "
                "</Node>                                                     " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        root->init(sofa::core::execparams::defaultInstance()) ;

        TheUniformMass* mass = root->getTreeObject<TheUniformMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getVertexMass(), 4.0 ) ;
            EXPECT_EQ( mass->getTotalMass(), 8.0 ) ;
        }
    }


    /// Both vertexMass and totalMass information are defined but totalMass is negative
    /// Due to double declaration, by default the totalMass is used
    /// Due to negative value, the default value of totalMass overwrites totalMass = 1.0
    void checkDoubleDeclarationNegativeTotalMass(){
        const string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "   <MechanicalObject position='0 0 0 4 5 6'/>               "
                "   <UniformMass name='m_mass' vertexMass='4.0' totalMass='-8.0'/>                 "
                "</Node>                                                     " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        root->init(sofa::core::execparams::defaultInstance()) ;

        TheUniformMass* mass = root->getTreeObject<TheUniformMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getTotalMass(), 1.0 ) ;
            EXPECT_EQ( mass->getVertexMass(), 0.5 ) ;
        }
    }

    /// Both vertexMass and totalMass information are defined but vertexMass is negative
    /// By default use the totalMass
    void checkDoubleDeclarationNegativeVertexMass(){
        const string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "   <MechanicalObject position='0 0 0 4 5 6'/>               "
                "   <UniformMass name='m_mass' vertexMass='-4.0' totalMass='8.0'/>                 "
                "</Node>                                                     " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        root->init(sofa::core::execparams::defaultInstance()) ;

        TheUniformMass* mass = root->getTreeObject<TheUniformMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getVertexMass(), 4.0 ) ;
            EXPECT_EQ( mass->getTotalMass(), 8.0 ) ;
        }
    }

    /// Both vertexMass and totalMass information are negative
    /// Ignore them and use default value of totalMass = 1.0
    void checkDoubleDeclarationBothNegative(){
        const string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "   <MechanicalObject position='0 0 0 4 5 6'/>               "
                "   <UniformMass name='m_mass' totalMass='-8.0' vertexMass='-4.0'/>   "
                "</Node>                                                     " ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());

        root->init(sofa::core::execparams::defaultInstance()) ;

        TheUniformMass* mass = root->getTreeObject<TheUniformMass>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getVertexMass(), 0.5 ) ;
            EXPECT_EQ( mass->getTotalMass(), 1.0 ) ;
        }
    }

    /// check mass initialization for rigids
    void loadFromAFileForRigid(){
        const string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "   <MechanicalObject template='Rigid3' position='0 0 0 1 0 0 1 0 0 0 1 0 0 1'/>                     "
                "   <UniformMass filename='BehaviorModels/card.rigid'/>        "
                "</Node>                                                     " ;
        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadFromAValidFile", scene.c_str());
        root->init(sofa::core::execparams::defaultInstance()) ;

        const UniformMassRigid* mass = root->getTreeObject<UniformMassRigid>() ;
        EXPECT_TRUE( mass != nullptr ) ;

        if(mass!=nullptr){
            EXPECT_EQ( mass->getVertexMass(), 40.0 ) ;
            EXPECT_EQ( mass->getTotalMass(), 80.0 ) ;
        }
    }

    void loadFromAFileForNonRigid(){
        const string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "   <MechanicalObject position='0 0 0'/>                     "
                "   <UniformMass filename='BehaviorModels/card.rigid'/>        "
                "</Node>                                                     " ;
        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadFromAValidFile", scene.c_str());
        root->init(sofa::core::execparams::defaultInstance()) ;
    }

    void loadFromAnInvalidFile(){
        const string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "   <MechanicalObject position='0 0 0'/>                     "
                "   <UniformMass filename='invalid_uniformmatrix.txt'/>        "
                "</Node>                                                     " ;
        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadFromAnInValidFile", scene.c_str());
        root->init(sofa::core::execparams::defaultInstance()) ;
    }

    void loadFromAnInvalidPathname(){
        const string scene =
                "<?xml version='1.0'?>"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   > "
                "   <MechanicalObject position='0 0 0'/>                     "
                "   <UniformMass filename='invalid_uniformmatrix.txt'/>        "
                "</Node>                                                     " ;
        const Node::SPtr root = SceneLoaderXML::loadFromMemory("loadFromAnInValidFile", scene.c_str());
        root->init(sofa::core::execparams::defaultInstance()) ;
    }

    void reinitTest(){
        // TODO
        EXPECT_TRUE(todo == false) ;
    }


    static Node::SPtr generateRigidScene()
    {
        static const string scene =
        "<?xml version='1.0' ?>"
        "<Node name='root' dt='0.01' gravity='0 0 0'>"
        "    <RequiredPlugin name='Sofa.Component.Mass'/>"
        "    <RequiredPlugin name='Sofa.Component.StateContainer'/>"
        "    <RequiredPlugin name='Sofa.Component.Topology.Container.Grid'/>"
        "    <RequiredPlugin name='Sofa.Component.Visual'/>"
        "    <RequiredPlugin name='Sofa.Component.ODESolver.Backward'/>"
        "    <RequiredPlugin name='Sofa.Component.LinearSolver.Direct'/>"
        "    <RequiredPlugin name='Sofa.Component.Engine.Select'/>"
        "    <RequiredPlugin name='Sofa.Component.Constraint.Projective'/>"
        "    <RequiredPlugin name='Sofa.Component.SolidMechanics.FEM.Elastic'/>"
        "    <RequiredPlugin name='Sofa.Component.MechanicalLoad'/>"
        "    <DefaultAnimationLoop />"
        "    <EulerImplicitSolver rayleighStiffness='0.'  rayleighMass='0.0'/>"
        "    <SparseLDLSolver template='CompressedRowSparseMatrixd'/>"
        "    <Node name='Aligned' >"
        "        <MechanicalObject  name='Mstate1' template='Rigid3' position='0 0 0 0 0 0 1' showObject='true' showObjectScale='0.1'/>"
        "        <UniformMass name='mass' vertexMass='300 0.0158 [0.0427 0.0 0.0 0.0 0.0427 0.0 0.0 0.0 0.00375]'/>"
        "        <ConstantForceField name='ConstantForceField1' forces='0 0 0 0 0 0'/>"
        "    </Node>"
        "    <Node name='Rotated' >"
        "        <MechanicalObject name='Mstate2' template='Rigid3' position='1 0 0 0 0 0 1' showObject='true' showObjectScale='0.1'/>"
        "        <UniformMass name='mass' vertexMass='300 0.0158 [0.0427 0.0 0.0 0.0 0.0427 0.0 0.0 0.0 0.00375]'/>"
        "        <ConstantForceField name='ConstantForceField2' forces='0 0 0 0 0 0'/>"
        "    </Node>"
        "</Node>";



        Node::SPtr root = SceneLoaderXML::loadFromMemory("loadWithNoParam", scene.c_str());
        sofa::simulation::node::initRoot(root.get());

        return root;
    }

    void nonIdentityInertiaMatrix_DifferentRotationDirection()
    {
        Node::SPtr root = generateRigidScene();
        Rigid3Types::VecDeriv* CF1_force = reinterpret_cast<Rigid3Types::VecDeriv*>(root->getChild("Aligned")->getObject("ConstantForceField1")->findData("forces")->beginEditVoidPtr());
        Rigid3Types::VecDeriv* CF2_force = reinterpret_cast<Rigid3Types::VecDeriv*>(root->getChild("Rotated")->getObject("ConstantForceField2")->findData("forces")->beginEditVoidPtr());

        (*CF1_force)[0][5] = 1.0;
        (*CF2_force)[0][3] = 1.0;

        root->getChild("Aligned")->getObject("ConstantForceField1")->findData("forces")->endEditVoidPtr();
        root->getChild("Rotated")->getObject("ConstantForceField2")->findData("forces")->endEditVoidPtr();

        auto mstate1 = root->getChild("Aligned")->getNodeObject<MechanicalObject<Rigid3Types>>();
        auto mstate2 = root->getChild("Rotated")->getNodeObject<MechanicalObject<Rigid3Types>>();

        sofa::simulation::node::animate(root.get(),50);

        //Because the inertia is smaller along z, we expect different velocity after some times with a ratio equivalent to the inverse ratio between the inertia
        EXPECT_GT(mstate2->read(sofa::core::ConstVecDerivId::velocity())->getValue()[0][3],0.0);
        EXPECT_NEAR(mstate1->read(sofa::core::ConstVecDerivId::velocity())->getValue()[0][5] /
                    mstate2->read(sofa::core::ConstVecDerivId::velocity())->getValue()[0][3],
                    0.0427/0.00375, 1.0e-5 );


    }

    void nonIdentityInertiaMatrix_RotationOfOneRigid()
    {
        Node::SPtr root = generateRigidScene();

        sofa::simulation::node::animate(root.get(),1);


        auto mstate1 = root->getChild("Aligned")->getNodeObject<MechanicalObject<Rigid3Types>>();
        auto mstate2 = root->getChild("Rotated")->getNodeObject<MechanicalObject<Rigid3Types>>();

        //Rotate the rigid
        mstate2->write(sofa::core::VecCoordId::position())->setValue({Rigid3Types::Coord(Rigid3Types::Coord::Pos(1,0,0),Rigid3Types::Coord::Rot (0.707106781,0,0.707106781,0))});

        Rigid3Types::VecDeriv* CF1_force = reinterpret_cast<Rigid3Types::VecDeriv*>(root->getChild("Aligned")->getObject("ConstantForceField1")->findData("forces")->beginEditVoidPtr());
        Rigid3Types::VecDeriv* CF2_force = reinterpret_cast<Rigid3Types::VecDeriv*>(root->getChild("Rotated")->getObject("ConstantForceField2")->findData("forces")->beginEditVoidPtr());

        //With rotated state, we now apply rotation along the Z axis of both rigids, this should result in the same acceleration if the inertia matrix is also rotated
        (*CF1_force)[0][5] = 1.0;
        (*CF2_force)[0][3] = 1.0;

        root->getChild("Aligned")->getObject("ConstantForceField1")->findData("forces")->endEditVoidPtr();
        root->getChild("Rotated")->getObject("ConstantForceField2")->findData("forces")->endEditVoidPtr();

        sofa::simulation::node::animate(root.get(),50);
        EXPECT_GT(mstate1->read(sofa::core::ConstVecDerivId::velocity())->getValue()[0][5],0.0);
        EXPECT_GT(mstate2->read(sofa::core::ConstVecDerivId::velocity())->getValue()[0][3],0.0);
        EXPECT_NEAR(mstate1->read(sofa::core::ConstVecDerivId::velocity())->getValue()[0][5] /
                    mstate2->read(sofa::core::ConstVecDerivId::velocity())->getValue()[0][3],
                    1.0, 1.0e-5 );
    }

    void nonIdentityInertiaMatrix_CentrifugalForces()
    {
        Node::SPtr root = generateRigidScene();

        sofa::simulation::node::animate(root.get(), 1);

        constexpr sofa::type::Vec3 zAxis(0, 0, 1);
        auto * mstate1 = root->getChild("Aligned")->getNodeObject<MechanicalObject<Rigid3Types>>();
        const auto * DataPos = mstate1->read(sofa::core::ConstVecId::position());
        const auto * DataVel = mstate1->read(sofa::core::ConstVecDerivId::velocity());

        Rigid3Types::VecDeriv* CF1_force = reinterpret_cast<Rigid3Types::VecDeriv*>(root->getChild("Aligned")->getObject("ConstantForceField1")->findData("forces")->beginEditVoidPtr());

        //We apply two different rotation, one exactly normal to z and one slightly along z
        (*CF1_force)[0][3] = 1.0;
        (*CF1_force)[0][5] = 0.1;

        root->getChild("Aligned")->getObject("ConstantForceField1")->findData("forces")->endEditVoidPtr();

        sofa::simulation::node::animate(root.get(),100);

        //After rotating for some time, the centrifugal forces should have made the Z axis (along which most of the mass is located) normal to the axis of rotation
        sofa::type::Vec3 Vel1 = DataVel->getValue()[0].getVOrientation();
        sofa::type::Vec3 ori1_z = DataPos->getValue()[0].getOrientation().rotate(zAxis);
        EXPECT_GT(norm(Vel1),0.0);
        EXPECT_NEAR(dot(Vel1/norm(Vel1),ori1_z), 0.0, 1.0e-5 );

        //To make sure the first try wasn't a fluke, we continue the rotation a bit and we check again
        sofa::simulation::node::animate(root.get(),5);
        Vel1 = DataVel->getValue()[0].getVOrientation();
        ori1_z = DataPos->getValue()[0].getOrientation().rotate(zAxis);
        EXPECT_GT(norm(Vel1),0.0);
        EXPECT_NEAR(dot(Vel1/norm(Vel1),ori1_z), 0.0, 1.0e-5 );

        //and again
        sofa::simulation::node::animate(root.get(),5);
        Vel1 = DataVel->getValue()[0].getVOrientation();
        ori1_z = DataPos->getValue()[0].getOrientation().rotate(zAxis);
        EXPECT_GT(norm(Vel1),0.0);
        EXPECT_NEAR(dot(Vel1/norm(Vel1),ori1_z), 0.0, 1.0e-5 );

        //and one final time
        sofa::simulation::node::animate(root.get(),5);
        Vel1 = DataVel->getValue()[0].getVOrientation();
        ori1_z = DataPos->getValue()[0].getOrientation().rotate(zAxis);
        EXPECT_GT(norm(Vel1),0.0);
        EXPECT_NEAR(dot(Vel1/norm(Vel1),ori1_z), 0.0, 1.0e-5 );
    }

};


typedef Types<
TemplateTypes<Vec3Types, Vec3Types::Real>> DataTypes;

TYPED_TEST_SUITE(UniformMassTest, DataTypes);


TYPED_TEST(UniformMassTest, attributesTests) {
    ASSERT_NO_THROW(this->attributesTests()) ;
}

TYPED_TEST(UniformMassTest, checkNoAttributes)
{
    ASSERT_NO_THROW(this->checkNoAttributes()) ;
}

TYPED_TEST(UniformMassTest, checkVertexMassFromTotalMass)
{
    ASSERT_NO_THROW(this->checkVertexMassFromTotalMass()) ;
}

TYPED_TEST(UniformMassTest, checkTotalMassFromVertexMass)
{
    ASSERT_NO_THROW(this->checkTotalMassFromVertexMass()) ;
}

TYPED_TEST(UniformMassTest, checkNegativeVertexMass)
{
    ASSERT_NO_THROW(this->checkNegativeVertexMass()) ;
}

TYPED_TEST(UniformMassTest, checkNegativeTotalMass)
{
    ASSERT_NO_THROW(this->checkNegativeTotalMass()) ;
}

TYPED_TEST(UniformMassTest, checkDoubleDeclarationVertexAndTotalMass)
{
    ASSERT_NO_THROW(this->checkDoubleDeclarationVertexAndTotalMass()) ;
}

TYPED_TEST(UniformMassTest, checkDoubleDeclarationNegativeTotalMass)
{
    ASSERT_NO_THROW(this->checkDoubleDeclarationNegativeTotalMass()) ;
}

TYPED_TEST(UniformMassTest, checkDoubleDeclarationNegativeVertexMass)
{
    ASSERT_NO_THROW(this->checkDoubleDeclarationNegativeVertexMass()) ;
}

TYPED_TEST(UniformMassTest, checkDoubleDeclarationBothNegative)
{
    ASSERT_NO_THROW(this->checkDoubleDeclarationBothNegative()) ;
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

TYPED_TEST(UniformMassTest, checkRigidAttribute) {
    ASSERT_NO_THROW(this->checkRigidAttribute()) ;
}

TYPED_TEST(UniformMassTest, reinitTest) {
    //ASSERT_NO_THROW(this->reinitTest()) ;
}

TYPED_TEST(UniformMassTest, nonIdentityInertiaMatrix_DifferentRotationDirection){
    EXPECT_NO_THROW(this->nonIdentityInertiaMatrix_DifferentRotationDirection());
}

TYPED_TEST(UniformMassTest, nonIdentityInertiaMatrix_RotationOfOneRigid){
    EXPECT_NO_THROW(this->nonIdentityInertiaMatrix_RotationOfOneRigid());
}

TYPED_TEST(UniformMassTest, nonIdentityInertiaMatrix_CentrifugalForces){
    EXPECT_NO_THROW(this->nonIdentityInertiaMatrix_CentrifugalForces());
}


