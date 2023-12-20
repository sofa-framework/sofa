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
#include <sofa/component/solidmechanics/fem/elastic/TetrahedronFEMForceField.h>

#include <sofa/simulation/graph/DAGSimulation.h>
using sofa::core::objectmodel::ComponentState ;
using sofa::core::objectmodel::BaseObject ;
using sofa::simulation::Simulation ;
using sofa::simulation::Node ;

#include <sofa/simulation/common/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

#include <sofa/component/solidmechanics/testing/ForceFieldTestCreation.h>
#include <sofa/testing/BaseTest.h>
#include <sofa/simulation/graph/SimpleApi.h>
#include <sofa/helper/system/thread/CTime.h>
#include <limits>

#include <sofa/simulation/graph/SimpleApi.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/Node.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/component/solidmechanics/fem/elastic/TetrahedralCorotationalFEMForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/FastTetrahedralCorotationalForceField.h>

using sofa::core::execparams::defaultInstance; 

namespace sofa {


/**  Test suite for TetrahedronFEMForceField.
  */
template <typename _TetrahedronFEMForceField>
struct TetrahedronFEMForceField_stepTest : public ForceField_test<_TetrahedronFEMForceField>
{

    typedef _TetrahedronFEMForceField ForceType;
    typedef ForceField_test<_TetrahedronFEMForceField> Inherited;
    typedef typename ForceType::DataTypes DataTypes;

    typedef typename ForceType::VecCoord VecCoord;
    typedef typename ForceType::VecDeriv VecDeriv;
    typedef typename ForceType::Coord Coord;
    typedef typename ForceType::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef type::Vec<3,Real> Vec3;

    VecCoord x;
    VecDeriv v,f;

    /** @name Test_Cases
      For each of these cases, we check if the accurate forces are computed
    */
    TetrahedronFEMForceField_stepTest():Inherited::ForceField_test(std::string(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_TEST_SCENES_DIR) + "/" + "TetrahedronFEMForceFieldRegularTetra.scn")
    {
        //Position
        x.resize(4);
        DataTypes::set( x[0], 0,0,0);
        DataTypes::set( x[1], 1,0,0);
        DataTypes::set( x[2], (Real)0.5, (Real)0.8660254037, (Real)0);
        //DataTypes::set( x[3], (SReal)0.5, (SReal)0.28867,(SReal)1.632993);
        DataTypes::set( x[3], (Real)0.5, (Real)0.28867513,(Real)2);
        //Velocity
        v.resize(4);
        DataTypes::set( v[0], 0,0,0);
        DataTypes::set( v[1], 0,0,0);
        DataTypes::set( v[2], 0,0,0);
        DataTypes::set( v[3], 0,0,0);

        //Force e*E*S*1/3  = 1*40*sqrt(3)/4*1/3
        f.resize(4);
        Vec3 fup(0,0,std::sqrt(3.)*10.0/3.0);
        Vec3 fdown(0,0,std::sqrt(3.)*10.0/9.0);
        DataTypes::set( f[0],  fdown[0], fdown[1], (Real)fdown[2]);
        DataTypes::set( f[1],  fdown[0], fdown[1], (Real)fdown[2]);
        DataTypes::set( f[2],  fdown[0], fdown[1], (Real)fdown[2]);
        DataTypes::set( f[3],  -fup[0], -fup[1], -(Real)fup[2]);

        // Set force parameters
        Inherited::force->_poissonRatio.setValue(0);
        type::vector<Real> youngModulusVec;youngModulusVec.push_back(40);
        Inherited::force->_youngModulus.setValue(youngModulusVec);
        Inherited::force->f_method.setValue("small");

        // Init simulation
        sofa::simulation::node::initRoot(Inherited::node.get());
    }

    //Test the value of the force it should be equal for each vertex to Pressure*area/4
    void test_valueForce()
    {
        // run the forcefield_test
        Inherited::run_test( x, v, f );
    }

    void checkGracefullHandlingWhenTopologyIsMissing()
    {
        modeling::clearScene();

        // This is a RAII message.
        EXPECT_MSG_EMIT(Error) ;

        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node 	name='Root'>                                \n"
                 "  <RequiredPlugin name=\"Sofa.Component.StateContainer\"/>"
                 "  <RequiredPlugin name=\"Sofa.Component.SolidMechanics.FEM.Elastic\"/>"
                 "  <DefaultAnimationLoop/>"
                 "  <Node name='FEMnode'>                               \n"
                 "    <MechanicalObject/>                               \n"
                 "    <TetrahedronFEMForceField name='fem' youngModulus='5000' poissonRatio='0.07'/>\n"
                 "  </Node>                                             \n"
                 "</Node>                                               \n" ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                                scene.str().c_str()) ;
        root->init(sofa::core::execparams::defaultInstance()) ;

        BaseObject* fem = root->getTreeNode("FEMnode")->getObject("fem") ;
        EXPECT_NE(fem, nullptr) ;

        EXPECT_EQ(fem->getComponentState(), ComponentState::Invalid) ;
    }
};





using sofa::helper::system::thread::ctime_t;
using namespace sofa::type;
using namespace sofa::defaulttype;
using namespace sofa::simpleapi;
using sofa::testing::BaseTest;

template <class DataTypes>
class TetrahedronFEMForceField_test : public BaseTest
{
public:
    using Real = typename DataTypes::Real;
    using Coord = typename DataTypes::Coord;
    using VecCoord = typename DataTypes::VecCoord;

    using MState = sofa::component::statecontainer::MechanicalObject<DataTypes>;
    using TetrahedronFEM = sofa::component::solidmechanics::fem::elastic::TetrahedronFEMForceField<DataTypes>;
    using TetraCorotationalFEM = sofa::component::solidmechanics::fem::elastic::TetrahedralCorotationalFEMForceField<DataTypes>;
    using FastTetraCorotationalFEM = sofa::component::solidmechanics::fem::elastic::FastTetrahedralCorotationalForceField<DataTypes>;

    using Transformation = type::Mat<3, 3, Real>;
    using MaterialStiffness = type::Mat<6, 6, Real>;
    using StrainDisplacement = type::Mat<12, 6, Real>;
    using TetraCoord = type::fixed_array<Coord, 4>;
    using Vec6 = type::Vec6;

    static constexpr const char* dataTypeName = DataTypes::Name();

protected:
    simulation::Simulation* m_simulation = nullptr;
    simulation::Node::SPtr m_root;

    ctime_t timeTicks = sofa::helper::system::thread::CTime::getRefTicksPerSec();

public:

    void SetUp() override
    {
        m_simulation = sofa::simulation::getSimulation();
    }

    void TearDown() override
    {
        if (m_root != nullptr)
            sofa::simulation::node::unload(m_root);
    }

    void addTetraFEMForceField(Node::SPtr node, int FEMType, Real young, Real poisson, std::string method)
    {
        if (FEMType == 0) // TetrahedronFEMForceField
        {
            createObject(node, "TetrahedronFEMForceField", {
                {"name","FEM"}, {"youngModulus", str(young)}, {"poissonRatio", str(poisson)}, {"method", method} });
        }
        else if (FEMType == 1)
        {
            createObject(node, "TetrahedralCorotationalFEMForceField", {
                {"name","FEM"}, {"youngModulus", str(young)}, {"poissonRatio", str(poisson)}, {"method", method} });
        }
        else
        {
            createObject(node, "FastTetrahedralCorotationalForceField", {
                {"name","FEM"}, {"youngModulus", str(young)}, {"poissonRatio", str(poisson)}, {"method", method} });
        }
    }

    
    void createSingleTetrahedronFEMScene(int FEMType, Real young, Real poisson, std::string method)
    {
        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");

        createObject(m_root, "DefaultAnimationLoop");
        createObject(m_root, "DefaultVisualManagerLoop");

        sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");
        sofa::simpleapi::importPlugin("Sofa.Component.Topology.Container.Dynamic");
        sofa::simpleapi::importPlugin("Sofa.Component.SolidMechanics.FEM.Elastic");
        sofa::simpleapi::importPlugin("Sofa.Component.Mass");

        createObject(m_root, "MechanicalObject", { {"template", dataTypeName}, {"position", "0 0 0  1 0 0  0 1 0  0 0 1"} });
        createObject(m_root, "TetrahedronSetTopologyContainer", { {"tetrahedra","2 3 1 0"} });
        createObject(m_root, "TetrahedronSetTopologyModifier");
        createObject(m_root, "TetrahedronSetGeometryAlgorithms", { {"template", dataTypeName} });

        addTetraFEMForceField(m_root, FEMType, young, poisson, method);
        
        createObject(m_root, "DiagonalMass", {
            {"name","mass"}, {"massDensity","0.1"} });
        /// Init simulation
        sofa::simulation::node::initRoot(m_root.get());
    }


    void createGridFEMScene(int FEMType, type::Vec3 nbrGrid)
    {
        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");
        m_root->setGravity(type::Vec3(0.0, 10.0, 0.0));
        m_root->setDt(0.01);

        createObject(m_root, "DefaultAnimationLoop");
        createObject(m_root, "DefaultVisualManagerLoop");


        sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");
        sofa::simpleapi::importPlugin("Sofa.Component.Topology.Container.Dynamic");
        sofa::simpleapi::importPlugin("Sofa.Component.SolidMechanics.FEM.Elastic");
        sofa::simpleapi::importPlugin("Sofa.Component.Mass");
        sofa::simpleapi::importPlugin("Sofa.Component.LinearSolver.Direct");
        sofa::simpleapi::importPlugin("Sofa.Component.ODESolver.Backward");
        sofa::simpleapi::importPlugin("Sofa.Component.Constraint.Lagrangian");
        sofa::simpleapi::importPlugin("Sofa.Component.Topology.Mapping");
        sofa::simpleapi::importPlugin("Sofa.Component.Engine.Select");
        sofa::simpleapi::importPlugin("Sofa.Component.Constraint.Projective");

        createObject(m_root, "GenericConstraintSolver", { {"tolerance", "1e-3"}, {"maxIt", "1000"} });
        
        createObject(m_root, "RegularGridTopology", { {"name", "grid"},
                    {"n", str(nbrGrid)}, {"min", "0 0 20"}, {"max", "10 40 30"} });


        Node::SPtr FEMNode = sofa::simpleapi::createChild(m_root, "Beam");
        createObject(FEMNode, "EulerImplicitSolver");
        //createObject(FEMNode, "SparseLDLSolver", { {"name","solver"}, { "template", "CompressedRowSparseMatrixd" } });
        createObject(FEMNode, "CGLinearSolver", { { "iterations", "20" }, { "tolerance", "1e-5" }, {"threshold", "1e-6"} });

        createObject(FEMNode, "MechanicalObject", {
            {"name","dof"}, {"template",dataTypeName}, {"position", "@../grid.position"} });

        createObject(FEMNode, "TetrahedronSetTopologyContainer", {
            {"name","topo"} });
        createObject(FEMNode, "TetrahedronSetTopologyModifier", {
            {"name","Modifier"} });
        createObject(FEMNode, "TetrahedronSetGeometryAlgorithms", {
            {"name","GeomAlgo"}, {"template",dataTypeName} });
        createObject(FEMNode, "Hexa2TetraTopologicalMapping", {
            {"input","@../grid"}, {"output","@topo"} });

        createObject(FEMNode, "BoxROI", {
            {"name","ROI1"}, {"box","-1 -1 0 10 1 50"} });
        createObject(FEMNode, "FixedProjectiveConstraint", { {"mstate", "@dof"},
            {"name","fixC"}, {"indices","@ROI1.indices"} });

        createObject(FEMNode, "DiagonalMass", {
            {"name","mass"}, {"massDensity","1.0"} });

        addTetraFEMForceField(FEMNode, FEMType, 600, 0.3, "large");

        ASSERT_NE(m_root.get(), nullptr);

        /// Init simulation
        sofa::simulation::node::initRoot(m_root.get());
    }



    void checkCreation(int FEMType)
    {
        createSingleTetrahedronFEMScene(FEMType, static_cast<Real>(10000), static_cast<Real>(0.4), "large");

        typename MState::SPtr dofs = m_root->getTreeObject<MState>();
        ASSERT_TRUE(dofs.get() != nullptr);
        ASSERT_EQ(dofs->getSize(), 4);

        if (FEMType == 0)
        {
            typename TetrahedronFEM::SPtr tetraFEM = m_root->getTreeObject<TetrahedronFEM>();
            ASSERT_TRUE(tetraFEM.get() != nullptr);
            ASSERT_FLOATINGPOINT_EQ(tetraFEM->_poissonRatio.getValue(), static_cast<Real>(0.4));
            ASSERT_FLOATINGPOINT_EQ(tetraFEM->_youngModulus.getValue()[0], static_cast<Real>(10000));
            ASSERT_EQ(tetraFEM->f_method.getValue(), "large");
        }
        else if (FEMType == 1)
        {
            typename TetraCorotationalFEM::SPtr tetraFEM = m_root->getTreeObject<TetraCorotationalFEM>();
            ASSERT_TRUE(tetraFEM.get() != nullptr);
            ASSERT_FLOATINGPOINT_EQ(tetraFEM->_poissonRatio.getValue(), static_cast<Real>(0.4));
            ASSERT_FLOATINGPOINT_EQ(tetraFEM->_youngModulus.getValue(), static_cast<Real>(10000));
            ASSERT_EQ(tetraFEM->f_method.getValue(), "large");
        }
        else
        {
            typename FastTetraCorotationalFEM::SPtr tetraFEM = m_root->getTreeObject<FastTetraCorotationalFEM>();
            ASSERT_TRUE(tetraFEM.get() != nullptr);
            ASSERT_FLOATINGPOINT_EQ(tetraFEM->f_poissonRatio.getValue(), static_cast<Real>(0.4));
            ASSERT_FLOATINGPOINT_EQ(tetraFEM->f_youngModulus.getValue(), static_cast<Real>(10000));
            ASSERT_EQ(tetraFEM->f_method.getValue(), "large");
        }
    }

    void checkNoTopology(int FEMType)
    {
        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");
        createObject(m_root, "DefaultAnimationLoop");
        createObject(m_root, "DefaultVisualManagerLoop");

        sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");

        createObject(m_root, "MechanicalObject", { {"template","Vec3"}, {"position", "0 0 0  1 0 0  0 1 0  0 0 1"} });
        addTetraFEMForceField(m_root, FEMType, 100, 0.3, "large");
        
        EXPECT_MSG_EMIT(Error);

        /// Init simulation
        sofa::simulation::node::initRoot(m_root.get());
    }

    void checkEmptyTopology(int FEMType)
    {
        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");
        createObject(m_root, "DefaultAnimationLoop");
        createObject(m_root, "DefaultVisualManagerLoop");

        sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");
        sofa::simpleapi::importPlugin("Sofa.Component.Topology.Container.Dynamic");

        createObject(m_root, "MechanicalObject", { {"template","Vec3"} });
        createObject(m_root, "TetrahedronSetTopologyContainer");
        addTetraFEMForceField(m_root, FEMType, 100, 0.3, "large");

        if (FEMType == 0)
        {
            EXPECT_MSG_EMIT(Error); // TODO: Need to change this behavior
            sofa::simulation::node::initRoot(m_root.get());
        }
        else
        {
            EXPECT_MSG_EMIT(Warning);
            /// Init simulation
            sofa::simulation::node::initRoot(m_root.get());
        }
    }


    void checkDefaultAttributes(int FEMType)
    {
        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");
        createObject(m_root, "DefaultAnimationLoop");
        createObject(m_root, "DefaultVisualManagerLoop");

        sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");
        sofa::simpleapi::importPlugin("Sofa.Component.Topology.Container.Dynamic");
        sofa::simpleapi::importPlugin("Sofa.Component.SolidMechanics.FEM.Elastic");

        createObject(m_root, "MechanicalObject", { {"template", dataTypeName}, {"position", "0 0 0  1 0 0  0 1 0  0 0 1"} });
        createObject(m_root, "TetrahedronSetTopologyContainer", { {"tetrahedra","2 3 1 0"} });
        createObject(m_root, "TetrahedronSetTopologyModifier");
        createObject(m_root, "TetrahedronSetGeometryAlgorithms", { {"template", dataTypeName} });
       
        if (FEMType == 0) 
        {
            createObject(m_root, "TetrahedronFEMForceField");
        }
        else if (FEMType == 1)
        {
            createObject(m_root, "TetrahedralCorotationalFEMForceField");
        }
        else
        {
            createObject(m_root, "FastTetrahedralCorotationalForceField");
        }


        if (FEMType == 0)
        {
            EXPECT_MSG_EMIT(Error); // TODO: Need to unify this behavior
            sofa::simulation::node::initRoot(m_root.get());
        }
        else
        {
            EXPECT_MSG_EMIT(Warning);
            /// Init simulation
            sofa::simulation::node::initRoot(m_root.get());
        }

       
        if (FEMType == 0)
        {
            typename TetrahedronFEM::SPtr tetraFEM = m_root->getTreeObject<TetrahedronFEM>();
            ASSERT_TRUE(tetraFEM.get() != nullptr);
            
            ASSERT_FLOATINGPOINT_EQ(tetraFEM->_poissonRatio.getValue(), static_cast<Real>(0.45));
            ASSERT_FLOATINGPOINT_EQ(tetraFEM->_youngModulus.getValue()[0], static_cast<Real>(5000));
            ASSERT_EQ(tetraFEM->f_method.getValue(), "large");
        }
        else if (FEMType == 1)
        {
            typename TetraCorotationalFEM::SPtr tetraFEM = m_root->getTreeObject<TetraCorotationalFEM>();
            ASSERT_TRUE(tetraFEM.get() != nullptr);
            ASSERT_FLOATINGPOINT_EQ(tetraFEM->_poissonRatio.getValue(), static_cast<Real>(0.45));
            ASSERT_FLOATINGPOINT_EQ(tetraFEM->_youngModulus.getValue(), static_cast<Real>(5000));
            ASSERT_EQ(tetraFEM->f_method.getValue(), "large");
        }
        else
        {
            typename FastTetraCorotationalFEM::SPtr tetraFEM = m_root->getTreeObject<FastTetraCorotationalFEM>();
            ASSERT_TRUE(tetraFEM.get() != nullptr);
            ASSERT_FLOATINGPOINT_EQ(tetraFEM->f_poissonRatio.getValue(), static_cast<Real>(0.45));
            ASSERT_FLOATINGPOINT_EQ(tetraFEM->f_youngModulus.getValue(), static_cast<Real>(5000));
            ASSERT_EQ(tetraFEM->f_method.getValue(), "qr");
        }
    }


    void checkWrongAttributes(int FEMType)
    {
        EXPECT_MSG_EMIT(Warning);
        createSingleTetrahedronFEMScene(FEMType, -100, -0.3, "toto");
    }


    void checkInit(int FEMType)
    {
        createSingleTetrahedronFEMScene(FEMType, static_cast<Real>(1000), static_cast<Real>(0.3), "large");

        // Expected values
        Transformation exp_initRot = { Vec3(0, 0.816497, 0.57735), Vec3(-0.707107, -0.408248, 0.57735), Vec3(0.707107, -0.408248, 0.57735) };
        TetraCoord exp_initPos = { Coord(0, 0, 0), Coord(1.41421, 0, 0), Coord(0.707107, 1.22474, 0), Coord(0.707107, 0.408248, -0.57735) };

        Transformation exp_curRot = { Vec3(0, 0.816497, 0.57735), Vec3(-0.707107, -0.408248, 0.57735), Vec3(0.707107, -0.408248, 0.57735) };

        MaterialStiffness exp_stiffnessMat = { Vec6(224.359, 96.1538, 96.1538, 0, 0, 0), Vec6(96.1538, 224.359, 96.1538, 0, 0, 0), Vec6(96.1538, 96.1538, 224.359, 0, 0, 0), 
            Vec6(0, 0, 0, 64.1026, 0, 0), Vec6(0, 0, 0, 0, 64.1026, 0), Vec6(0, 0, 0, 0, 0, 64.1026) };

        StrainDisplacement exp_strainD = { Vec6(0.707107, 0, 0, 0.408248, 0, -0.57735),
            Vec6(0, 0.408248, 0, 0.707107, -0.57735, 0),
            Vec6(0, 0, -0.57735, 0, 0.408248, 0.707107),
            Vec6(-0.707107, 0, 0, 0.408248, 0, -0.57735),
            Vec6(0, 0.408248, 0, -0.707107, -0.57735, 0),
            Vec6(0, 0, -0.57735, 0, 0.408248, -0.707107),
            Vec6(-0, 0, 0, -0.816497, 0, -0.57735),
            Vec6(0, -0.816497, 0, -0, -0.57735, 0),
            Vec6(0, 0, -0.57735, 0, -0.816497, 0),
            Vec6(0, 0, 0, -0, 0, 1.73205),
            Vec6(0, 0, 0, 0, 1.73205, 0),
            Vec6(0, 0, 1.73205, 0, 0, 0) };

        Transformation initRot (type::NOINIT);
        Transformation curRot(type::NOINIT);
        MaterialStiffness stiffnessMat(type::NOINIT);
        StrainDisplacement strainD(type::NOINIT);
        TetraCoord initPosition;

        if (FEMType == 0)
        {
            typename TetrahedronFEM::SPtr tetraFEM = m_root->getTreeObject<TetrahedronFEM>();
            ASSERT_TRUE(tetraFEM.get() != nullptr);

            initRot = tetraFEM->getInitialTetraRotation(0);
            initPosition = tetraFEM->getRotatedInitialElements(0);

            curRot = tetraFEM->getActualTetraRotation(0);

            stiffnessMat = tetraFEM->getMaterialStiffness(0);
            strainD = tetraFEM->getStrainDisplacement(0);
        }
        else if (FEMType == 1)
        {
            typename TetraCorotationalFEM::SPtr tetraFEM = m_root->getTreeObject<TetraCorotationalFEM>();
            ASSERT_TRUE(tetraFEM.get() != nullptr);

            const typename TetraCorotationalFEM::TetrahedronInformation& tetraInfo = tetraFEM->tetrahedronInfo.getValue()[0];
            initRot.transpose(tetraInfo.initialTransformation); // TODO check why transposed is stored in this version
            initPosition = tetraInfo.rotatedInitialElements;

            curRot = initRot; // TODO check why this is not computed at start

            stiffnessMat = tetraInfo.materialMatrix;
            strainD = tetraInfo.strainDisplacementTransposedMatrix;
        }
        else
        {
            typename FastTetraCorotationalFEM::SPtr tetraFEM = m_root->getTreeObject<FastTetraCorotationalFEM>();
            ASSERT_TRUE(tetraFEM.get() != nullptr);

            const typename FastTetraCorotationalFEM::TetrahedronRestInformation& tetraInfo = tetraFEM->tetrahedronInfo.getValue()[0];
            initRot.transpose(tetraInfo.restRotation); // TODO check why transposed is stored in this version
            curRot = initRot; // not needed at init.

            // Expected specific values
            TetraCoord exp_shapeVector = { Coord(0, 1, 0), Coord(0, 0, 1), Coord(1, 0, 0), Coord(-1, -1, -1) };
            Transformation exp_linearDfDxDiag[4]; 
            exp_linearDfDxDiag[0] = { Vec3(64.1026, 0, 0), Vec3(0, 224.359, 0), Vec3(0, 0, 64.1026) };
            exp_linearDfDxDiag[1] = { Vec3(64.1026, 0, -0), Vec3(0, 64.1026, -0), Vec3(-0, -0, 224.359) };
            exp_linearDfDxDiag[2] = { Vec3(224.359, 0, 0), Vec3(0, 64.1026, 0), Vec3(0, 0, 64.1026) };
            exp_linearDfDxDiag[3] = { Vec3(352.5641, 160.25641, 160.25641), Vec3(160.25641, 352.5641, 160.25641), Vec3(160.25641, 160.25641, 352.5641) };

            Transformation exp_linearDfDx[6];
            exp_linearDfDx[0] = { Vec3(0, -0, 0), Vec3(0, 0, 64.1026), Vec3(0, 96.1538, 0) };
            exp_linearDfDx[1] = { Vec3(0, 96.1538, 0), Vec3(64.1026, 0, 0), Vec3(0, 0, 0) };
            exp_linearDfDx[2] = { Vec3(-64.1026, -96.1538, -0), Vec3(-64.1026, -224.359, -64.1026), Vec3(-0, -96.1538, -64.1026) };
            exp_linearDfDx[3] = { Vec3(0, -0, 96.1538), Vec3(-0, 0, 0), Vec3(64.1026, 0, 0) };
            exp_linearDfDx[4] = { Vec3(-64.1026, 0, -96.1538), Vec3(0, -64.1026, -96.1538), Vec3(-64.1026, -64.1026, -224.359) };
            exp_linearDfDx[5] = { Vec3(-224.359, -64.1026, -64.1026), Vec3(-96.1538, -64.1026, -0), Vec3(-96.1538, -0, -64.1026) };
   

            // check rotations
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    EXPECT_NEAR(exp_initRot[i][j], initRot[i][j], 1e-4);
                    EXPECT_NEAR(exp_curRot[i][j], curRot[i][j], 1e-4);
                }
            }

            // check shapeVector
            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    EXPECT_NEAR(exp_shapeVector[i][j], tetraInfo.shapeVector[i][j], 1e-4);
                }
            }

            // check DfDx
            for (int id = 0; id < 4; ++id)
            {
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        EXPECT_NEAR(exp_linearDfDxDiag[id][i][j], tetraInfo.linearDfDxDiag[id][i][j], 1e-4);
                        EXPECT_NEAR(exp_linearDfDx[id][i][j], tetraInfo.linearDfDx[id][i][j], 1e-4);
                    }
                }
            }

            // Fast method do not share the other data.
            return;
        }
            


        // check rotations
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                EXPECT_NEAR(exp_initRot[i][j], initRot[i][j], 1e-4);
                EXPECT_NEAR(exp_curRot[i][j], curRot[i][j], 1e-4);
            }
        }

        // check position
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                EXPECT_NEAR(exp_initPos[i][j], initPosition[i][j], 1e-4);
            }
        }

        // check stiffness
        for (int i = 0; i < 6; ++i)
        {
            for (int j = 0; j < 6; ++j)
            {
                EXPECT_NEAR(exp_stiffnessMat[i][j], stiffnessMat[i][j], 1e-4);
            }
        }

        // check strain displacement
        for (int i = 0; i < 12; ++i)
        {
            for (int j = 0; j < 6; ++j)
            {
                EXPECT_NEAR(exp_strainD[i][j], strainD[i][j], 1e-4);
            }
        }
        
    }


    void checkFEMValues(int FEMType)
    {
        type::Vec3 grid = type::Vec3(4, 10, 4);

        // load TetrahedronFEMForceField grid
        createGridFEMScene(FEMType, grid);
        if (m_root.get() == nullptr)
            return;

        // Access mstate
        typename MState::SPtr dofs = m_root->getTreeObject<MState>();
        ASSERT_TRUE(dofs.get() != nullptr);

        // Access dofs
        const VecCoord& positions = dofs->x.getValue();
        ASSERT_EQ(positions.size(), 4 * 10 * 4);

        EXPECT_NEAR(positions[159][0], 10, 1e-4);
        EXPECT_NEAR(positions[159][1], 40, 1e-4);
        EXPECT_NEAR(positions[159][2], 30, 1e-4);

        // perform some steps
        for (int i = 0; i < 100; i++)
        {
            sofa::simulation::node::animate(m_root.get(), 0.01_sreal);
        }

        EXPECT_NEAR(positions[159][0], 9.99985, 1e-4);
        EXPECT_NEAR(positions[159][1], 45.0487, 1e-4);
        EXPECT_NEAR(positions[159][2], 30.0011, 1e-4);

        // Expected values
        Transformation exp_initRot = { Vec3(-1, 0, 0), Vec3(0, -0.8, -0.6), Vec3(0, -0.6, 0.8) };
        TetraCoord exp_initPos = { Coord(0, 0, 0), Coord(3.33333, 0, 0), Coord(3.33333, 5.55556, 0), Coord(0, 3.55556, 2.66667) };

        Transformation exp_curRot = { Vec3(-1, 8.01488e-06, 0.000541687), Vec3(-0.000320764, -0.814541, -0.580106), Vec3(0.000436576, -0.580106, 0.814541) };

        MaterialStiffness exp_stiffnessMat = { Vec6(2.72596, 1.16827, 1.16827, 0, 0, 0), Vec6(1.16827, 2.72596, 1.16827, 0, 0, 0), Vec6(1.16827, 1.16827, 2.72596, 0, 0, 0),
            Vec6(0, 0, 0, 0.778846, 0, 0), Vec6(0, 0, 0, 0, 0.778846, 0), Vec6(0, 0, 0, 0, 0, 0.778846) };

        StrainDisplacement exp_strainD = { Vec6(-14.8148, 0, 0, 1.18424e-14, 0, -18.5185),
            Vec6(0, 1.18424e-14, 0, -14.8148, -18.5185, 0),
            Vec6(0, 0, -18.5185, 0, 1.18424e-14, -14.8148),
            Vec6(14.8148, 0, 0, -8.88889, 0, 11.8519),
            Vec6(0, -8.88889, 0, 14.8148, 11.8519, 0),
            Vec6(0, 0, 11.8519, 0, -8.88889, 14.8148),
            Vec6(-0, 0, 0, 8.88889, 0, -11.8519),
            Vec6(0, 8.88889, 0, -0, -11.8519, 0),
            Vec6(0, 0, -11.8519, 0, 8.88889, -0),
            Vec6(0, 0, 0, -1.18424e-14, 0, 18.5185),
            Vec6(0, -1.18424e-14, 0, 0, 18.5185, 0),
            Vec6(0, 0, 18.5185, 0, -1.18424e-14, 0) };


        Transformation initRot(type::NOINIT);
        Transformation curRot(type::NOINIT);
        MaterialStiffness stiffnessMat(type::NOINIT);
        StrainDisplacement strainD(type::NOINIT);
        TetraCoord initPosition;

        if (FEMType == 0)
        {
            typename TetrahedronFEM::SPtr tetraFEM = m_root->getTreeObject<TetrahedronFEM>();
            ASSERT_TRUE(tetraFEM.get() != nullptr);

            initRot = tetraFEM->getInitialTetraRotation(100);
            initPosition = tetraFEM->getRotatedInitialElements(100);

            curRot = tetraFEM->getActualTetraRotation(100);

            stiffnessMat = tetraFEM->getMaterialStiffness(100);
            strainD = tetraFEM->getStrainDisplacement(100);
        }
        else if (FEMType == 1)
        {
            typename TetraCorotationalFEM::SPtr tetraFEM = m_root->getTreeObject<TetraCorotationalFEM>();
            ASSERT_TRUE(tetraFEM.get() != nullptr);

            const typename TetraCorotationalFEM::TetrahedronInformation& tetraInfo = tetraFEM->tetrahedronInfo.getValue()[100];
            initRot.transpose(tetraInfo.initialTransformation); // TODO check why transposed is stored in this version
            initPosition = tetraInfo.rotatedInitialElements;

            curRot = tetraInfo.rotation;

            stiffnessMat = tetraInfo.materialMatrix;
            strainD = tetraInfo.strainDisplacementTransposedMatrix;
        }
        else
        {
            typename FastTetraCorotationalFEM::SPtr tetraFEM = m_root->getTreeObject<FastTetraCorotationalFEM>();
            ASSERT_TRUE(tetraFEM.get() != nullptr);

            const typename FastTetraCorotationalFEM::TetrahedronRestInformation& tetraInfo = tetraFEM->tetrahedronInfo.getValue()[100];
            initRot.transpose(tetraInfo.restRotation); // TODO check why transposed is stored in this version
            curRot = tetraInfo.rotation; 
            
            // Expected specific values
            exp_curRot = { Vec3(0.99999985, 0.00032076406, -0.00043657642), Vec3(-0.00033142383, 0.99969634, -0.024639719), Vec3(0.00042854031, 0.024639861, 0.9996963) };
            TetraCoord exp_shapeVector = { Coord(0.3, 0.224999, -0.3), Coord(-0.3, 0, 0.3), Coord(0, 0, -0.3), Coord(0, -0.224999, 0.3) };

            Transformation exp_linearDfDxDiag[4];
            exp_linearDfDxDiag[0] = { Vec3(865.38462, 320.51282, -427.35043), Vec3(320.51282, 678.4188, -320.51282), Vec3(-427.35043, -320.51282, 865.38462) };
            exp_linearDfDxDiag[1] = { Vec3(769.23077, -0, -427.35043), Vec3(-0, 341.88034, 0), Vec3(-427.35043, 0, 769.23077) };
            exp_linearDfDxDiag[2] = { Vec3(170.94017, -0, 0), Vec3(-0, 170.94017, -0), Vec3(0, -0, 598.2906) };
            exp_linearDfDxDiag[3] = { Vec3(267.09402, -0, 0), Vec3(-0, 507.47863, -320.51282), Vec3(0, -320.51282, 694.44444) };

            Transformation exp_linearDfDx[6];
            exp_linearDfDx[0] = { Vec3(-769.23077, -192.30769, 427.35043), Vec3(-128.20513, -341.88034, 128.20513), Vec3(427.35043, 192.30769, -769.23077) };
            exp_linearDfDx[1] = { Vec3(170.94017, 0, -170.94017), Vec3(0, 170.94017, -128.20513), Vec3(-256.41026, -192.30769, 598.2906) };
            exp_linearDfDx[2] = { Vec3(-267.09402, -128.20513, 170.94017), Vec3(-192.30769, -507.47863, 320.51282), Vec3(256.41026, 320.51282, -694.44444) };
            exp_linearDfDx[3] = { Vec3(-170.94017, -0, 170.94017), Vec3(-0, -170.94017, 0), Vec3(256.41026, 0, -598.2906) };
            exp_linearDfDx[4] = { Vec3(170.94017, 128.20513, -170.94017), Vec3(192.30769, 170.94017, -192.30769), Vec3(-256.41026, -128.20513, 598.2906) };
            exp_linearDfDx[5] = { Vec3(-170.94017, 0, -0), Vec3(0, -170.94017, 192.30769), Vec3(-0, 128.20513, -598.2906) };


            // check rotations
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    EXPECT_NEAR(exp_initRot[i][j], initRot[i][j], 1e-4);
                    EXPECT_NEAR(exp_curRot[i][j], curRot[i][j], 1e-4);
                }
            }

            // check shapeVector
            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    EXPECT_NEAR(exp_shapeVector[i][j], tetraInfo.shapeVector[i][j], 1e-4);
                }
            }

            // check DfDx
            for (int id = 0; id < 4; ++id)
            {
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        EXPECT_NEAR(exp_linearDfDxDiag[id][i][j], tetraInfo.linearDfDxDiag[id][i][j], 1e-4);
                        EXPECT_NEAR(exp_linearDfDx[id][i][j], tetraInfo.linearDfDx[id][i][j], 1e-4);
                    }
                }
            }

            // Fast method do not share the other data.
            return;
        }
        

       

        // check rotations
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                EXPECT_NEAR(exp_initRot[i][j], initRot[i][j], 1e-4);
                EXPECT_NEAR(exp_curRot[i][j], curRot[i][j], 1e-4);
            }
        }

        // check position
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                EXPECT_NEAR(exp_initPos[i][j], initPosition[i][j], 1e-4);
            }
        }

        // check stiffness
        for (int i = 0; i < 6; ++i)
        {
            for (int j = 0; j < 6; ++j)
            {
                EXPECT_NEAR(exp_stiffnessMat[i][j], stiffnessMat[i][j], 1e-4);
            }
        }

        // check strain displacement
        for (int i = 0; i < 12; ++i)
        {
            for (int j = 0; j < 6; ++j)
            {
                EXPECT_NEAR(exp_strainD[i][j], strainD[i][j], 1e-4);
            }
        }
    }


    void testFEMPerformance(int FEMType)
    {
        const type::Vec3 grid = type::Vec3(8, 26, 8);

        // load TetrahedronFEMForceField grid
        createGridFEMScene(FEMType, grid);
        if (m_root.get() == nullptr)
            return;

        const int nbrStep = 1000;
        const int nbrTest = 4;
        double diffTimeMs = 0;
        double timeMin = std::numeric_limits<double>::max();
        double timeMax = std::numeric_limits<double>::min();

        if (m_simulation == nullptr)
            return;

        for (int i = 0; i < nbrTest; ++i)
        {
            const ctime_t startTime = sofa::helper::system::thread::CTime::getRefTime();
            for (int i = 0; i < nbrStep; i++)
            {
                sofa::simulation::node::animate(m_root.get(), 0.01_sreal);
            }

            const ctime_t diffTime = sofa::helper::system::thread::CTime::getRefTime() - startTime;
            const double diffTimed = sofa::helper::system::thread::CTime::toSecond(diffTime);

            if (timeMin > diffTimed)
                timeMin = diffTimed;
            if (timeMax < diffTimed)
                timeMax = diffTimed;

            diffTimeMs += diffTimed;
            sofa::simulation::node::reset(m_root.get());
        }

        std::cout << "timeMean: " << diffTimeMs / nbrTest << std::endl;
        std::cout << "timeMin: " << timeMin << std::endl;
        std::cout << "timeMax: " << timeMax << std::endl;

        //TetrahedronFEM
        //timeMean: 7.40746
        //timeMin : 7.37514
        //timeMax : 7.46645

        //TetrahedralCorotational
        //timeMean : 14.0486
        //timeMin : 13.9016
        //timeMax : 14.4603

        // FastTetrahedralCorotationalForceField
        //timeMean: 6.01042
        //timeMin : 5.95179
        //timeMax : 6.16263
    }


};



// ========= Define the list of types to instanciate.
//using ::testing::Types;
typedef ::testing::Types<
sofa::component::solidmechanics::fem::elastic::TetrahedronFEMForceField<defaulttype::Vec3Types>
> TestTypes; // the types to instanciate.



// ========= Tests to run for each instanciated type
TYPED_TEST_SUITE(TetrahedronFEMForceField_stepTest, TestTypes);

// test case
TYPED_TEST(TetrahedronFEMForceField_stepTest, extension )
{
    this->errorMax *= 1e6;
    this->deltaRange = std::make_pair( 1, this->errorMax * 10 );
    this->debug = false;

    // Young modulus, poisson ratio method

    // run test
    this->test_valueForce();
}

TYPED_TEST(TetrahedronFEMForceField_stepTest, checkGracefullHandlingWhenTopologyIsMissing)
{
    this->checkGracefullHandlingWhenTopologyIsMissing();
}





/// Tests for TriangleFEMForceField
typedef TetrahedronFEMForceField_test<Vec3Types> TetrahedronFEMForceField3_test;

TEST_F(TetrahedronFEMForceField3_test, checkCreation)
{
    this->checkCreation(0);
}

TEST_F(TetrahedronFEMForceField3_test, checkNoTopology)
{
    this->checkNoTopology(0);
}

TEST_F(TetrahedronFEMForceField3_test, checkEmptyTopology)
{
    this->checkEmptyTopology(0);
}

TEST_F(TetrahedronFEMForceField3_test, checkDefaultAttributes)
{
    this->checkDefaultAttributes(0);
}

//TEST_F(TetrahedronFEMForceField3_test, checkWrongAttributes)
//{
//    this->checkWrongAttributes(0);
//}

TEST_F(TetrahedronFEMForceField3_test, checkInit)
{
    this->checkInit(0);
}

TEST_F(TetrahedronFEMForceField3_test, checkFEMValues)
{
    this->checkFEMValues(0);
}



typedef TetrahedronFEMForceField_test<Vec3Types> TetrahedralCorotationalFEMForceField3_test;

TEST_F(TetrahedralCorotationalFEMForceField3_test, checkCreation)
{
    this->checkCreation(1);
}

TEST_F(TetrahedralCorotationalFEMForceField3_test, checkNoTopology)
{
    this->checkNoTopology(1);
}

TEST_F(TetrahedralCorotationalFEMForceField3_test, checkEmptyTopology)
{
    this->checkEmptyTopology(1);
}

TEST_F(TetrahedralCorotationalFEMForceField3_test, checkDefaultAttributes)
{
    this->checkDefaultAttributes(1);
}

//TEST_F(TetrahedralCorotationalFEMForceField3_test, checkWrongAttributes)
//{
//    this->checkWrongAttributes(1);
//}

TEST_F(TetrahedralCorotationalFEMForceField3_test, checkInit)
{
    this->checkInit(1);
}

TEST_F(TetrahedralCorotationalFEMForceField3_test, checkFEMValues)
{
    this->checkFEMValues(1);
}



typedef TetrahedronFEMForceField_test<Vec3Types> FastTetrahedralCorotationalForceField3_test;

TEST_F(FastTetrahedralCorotationalForceField3_test, checkCreation)
{
    this->checkCreation(2);
}

TEST_F(FastTetrahedralCorotationalForceField3_test, checkNoTopology)
{
    this->checkNoTopology(2);
}

TEST_F(FastTetrahedralCorotationalForceField3_test, checkEmptyTopology)
{
    this->checkEmptyTopology(2);
}

TEST_F(FastTetrahedralCorotationalForceField3_test, checkDefaultAttributes)
{
    this->checkDefaultAttributes(2);
}

//TEST_F(FastTetrahedralCorotationalForceField3_test, checkWrongAttributes)
//{
//    this->checkWrongAttributes(2);
//}

TEST_F(FastTetrahedralCorotationalForceField3_test, checkInit)
{
    this->checkInit(2);
}

TEST_F(FastTetrahedralCorotationalForceField3_test, checkFEMValues)
{
    this->checkFEMValues(2);
}


// performances tests. Disabled by default
TEST_F(TetrahedronFEMForceField3_test, DISABLED_testFEMPerformance)
{
    this->testFEMPerformance(0);
}

TEST_F(TetrahedralCorotationalFEMForceField3_test, DISABLED_testFEMPerformance)
{
    this->testFEMPerformance(1);
}


TEST_F(FastTetrahedralCorotationalForceField3_test, DISABLED_testFEMPerformance)
{
    this->testFEMPerformance(2);
}


} // namespace sofa
