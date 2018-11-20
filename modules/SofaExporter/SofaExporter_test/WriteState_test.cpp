/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <fstream>
#include <iterator>
#include <algorithm>

#include <SofaTest/Sofa_test.h>
#include <SofaTest/TestMessageHandler.h>
#include <SofaSimulationGraph/DAGSimulation.h>

#include <SofaImplicitOdeSolver/EulerImplicitSolver.h>
#include <SofaGeneralImplicitOdeSolver/VariationalSymplecticSolver.h>
#include <SofaBaseLinearSolver/CGLinearSolver.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaBaseMechanics/UniformMass.h>
#include <SofaExporter/WriteState.h>

namespace sofa {

    using namespace component;
    using namespace defaulttype;
    using sofa::core::objectmodel::New;


    template <typename _DataTypes>
    struct WriteState_test : public Sofa_test<typename _DataTypes::Real>
    {
        typedef _DataTypes DataTypes;
        typedef typename DataTypes::CPos CPos;
        typedef typename DataTypes::Real Real;
        typedef typename DataTypes::Coord Coord;
        typedef typename DataTypes::VecCoord VecCoord;
        typedef typename DataTypes::VecDeriv VecDeriv;
        typedef container::MechanicalObject<DataTypes> MechanicalObject;
        typedef component::mass::UniformMass<DataTypes, Real> UniformMass;
        typedef component::linearsolver::CGLinearSolver<component::linearsolver::GraphScatteredMatrix, component::linearsolver::GraphScatteredVector> CGLinearSolver;

        /// Root of the scene graph
        simulation::Node::SPtr root=NULL;
        /// Simulation
        simulation::Simulation* simulation=NULL;
        /// MechanicalObject
        typename MechanicalObject::SPtr mecaObj=NULL;
        /// Time step
        double timeStep=0.01;
        /// Gravity
        double gravity=-9.81;
        /// Expected result
        double final_expected_value=0.0;

        /// Create the context for the scene
        void SetUp()
        {
            // Init simulation
            sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());
            root = simulation::getSimulation()->createNewGraph("root");
        }

        // Create the scene and the components
        void createScene(bool symplectic)
        {
            timeStep = 0.01;
            root->setGravity(Coord(0.0,0.0,gravity));
            root->setDt(timeStep);

            if(symplectic)
            {
                sofa::component::odesolver::VariationalSymplecticSolver::SPtr variationalSolver = New<sofa::component::odesolver::VariationalSymplecticSolver>();
                root->addObject(variationalSolver);
            }
            else
            {
                sofa::component::odesolver::EulerImplicitSolver::SPtr eulerSolver = New<sofa::component::odesolver::EulerImplicitSolver>();
                root->addObject(eulerSolver);
            }
            CGLinearSolver::SPtr cgLinearSolver = New<CGLinearSolver> ();
            root->addObject(cgLinearSolver);

            simulation::Node::SPtr childNode = root->createChild("Particle");

            mecaObj = New<MechanicalObject>();
            mecaObj->resize(1);
            childNode->addObject(mecaObj);
            typename UniformMass::SPtr mass = New<UniformMass>();
            mass->setTotalMass(1.0);
            childNode->addObject(mass);

            sofa::component::misc::WriteState::SPtr writeState =New<sofa::component::misc::WriteState>();
            helper::vector<double> time;
            time.resize(1);
            time[0] = 0.0;
            writeState->d_period.setValue(timeStep);

            std::cout<<"SOFAEXPORTER_BUILD_DIR = "<<SOFAEXPORTER_BUILD_DIR<<std::endl;

            if(symplectic)
            {
                writeState->d_filename.setValue(std::string(SOFAEXPORTER_BUILD_DIR)+"particleGravityX.data");
                writeState->d_writeX.setValue(true);
                writeState->d_writeV.setValue(false);
            }
            else
            {
                writeState->d_filename.setValue(std::string(SOFAEXPORTER_BUILD_DIR)+"particleGravityV.data");
                writeState->d_writeX.setValue(false);
                writeState->d_writeV.setValue(true);
            }
            writeState->d_writeF.setValue(false);
            writeState->d_time.setValue(time);
            childNode->addObject(writeState);

            EXPECT_TRUE(childNode);
        }

        // Initialization of the scene
        void initScene()
        {
            sofa::simulation::getSimulation()->init(this->root.get());
        }

        // Run five steps of simulation
        void runScene()
        {
            for(int i=0; i<7; i++)
            {
                sofa::simulation::getSimulation()->animate(root.get(),timeStep);
            }
        }



        /// Function where you can implement the test you want to do
        bool simulation_result_test(bool symplectic)
        {
            double time = root->getTime();
            double result;

            // Compute the ANALYTICAL solution in POSITION
            if(symplectic)
            {
                final_expected_value = gravity*time*time/2.0; // position of a particle under gravity
                result = mecaObj->x.getValue()[0][2];
            }
            // Compute the ANALYTICAL solution in VELOCITY
            else
            {
                final_expected_value = gravity*time; // velocity of a particle under gravity
                result = mecaObj->v.getValue()[0][2];
            }

            EXPECT_TRUE( fabs(final_expected_value-result)<std::numeric_limits<double>::epsilon() );
            return true;
        }

        bool test_export(bool symplectic)
        {
            // Check the written file by WriteState : should be exactly the same as reference file
            std::string createdFile, referenceFile;
            if(symplectic)
            {
                createdFile = std::string(SOFAEXPORTER_BUILD_DIR)+"particleGravityX.data";
                referenceFile = std::string(SOFAEXPORTER_TESTFILES_DIR)+"particleGravityX-reference.data";
            }
            else
            {
                createdFile = std::string(SOFAEXPORTER_BUILD_DIR)+"particleGravityV.data";
                referenceFile = std::string(SOFAEXPORTER_TESTFILES_DIR)+"particleGravityV-reference.data";
            }

            std::ifstream f1(createdFile, std::ifstream::binary|std::ifstream::ate);
            std::ifstream f2(referenceFile, std::ifstream::binary|std::ifstream::ate);

            if (f1.fail())
            {
                std::cout<<"Problem opening file "+createdFile<<std::endl;
                return false;
            }
            if(f2.fail())
            {
                std::cout<<"Problem opening file "+referenceFile<<std::endl;
                return false;
            }

            if (f1.tellg() != f2.tellg())
            {
                std::cout<<"File size mismatch "<<std::endl;
                return false;
            }

            //seek back to beginning and use std::equal to compare contents
            f1.seekg(0, std::ifstream::beg);
            f2.seekg(0, std::ifstream::beg);
            return std::equal(std::istreambuf_iterator<char>(f1.rdbuf()),
                              std::istreambuf_iterator<char>(),
                              std::istreambuf_iterator<char>(f2.rdbuf()));
        }


        /// Unload the scene
        void TearDown()
        {
            if (root!=NULL)
                sofa::simulation::getSimulation()->unload(root);
        }

    };

    // Define the list of DataTypes to instantiate
    typedef testing::Types< Vec3Types > DataTypes;

    // Test suite for all the instantiations
    TYPED_TEST_CASE(WriteState_test, DataTypes);

    // Test 1 : write position of a particle falling under gravity (required to use SymplecticSolver
    TYPED_TEST( WriteState_test , test_write_position)
    {
        this->SetUp();
        this->createScene(true);
        this->initScene();
        this->runScene();

        ASSERT_TRUE( this->simulation_result_test(true) );
        ASSERT_TRUE( this->test_export(true) );
        this->TearDown();
    }

    // Test 2 : write velocity of a particle falling under gravity
    TYPED_TEST( WriteState_test , test_write_velocity)
    {
        this->SetUp();
        this->createScene(false);
        this->initScene();
        this->runScene();

        ASSERT_TRUE( this->simulation_result_test(false) );
        ASSERT_TRUE( this->test_export(false) );
        this->TearDown();
    }
}
