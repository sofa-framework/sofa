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

#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaGeneralLoader/ReadState.h>

namespace sofa {

    using namespace component;
    using namespace defaulttype;
    using sofa::core::objectmodel::New;


    template <typename _DataTypes>
    struct ReadState_test : public Sofa_test<typename _DataTypes::Real>
    {
        typedef _DataTypes DataTypes;
        typedef typename DataTypes::CPos CPos;
        typedef typename DataTypes::Real Real;
        typedef typename DataTypes::Coord Coord;
        typedef typename DataTypes::VecCoord VecCoord;
        typedef typename DataTypes::VecDeriv VecDeriv;
        typedef container::MechanicalObject<DataTypes> MechanicalObject;

        /// Root of the scene graph
        simulation::Node::SPtr root=NULL;
        /// Simulation
        simulation::Simulation* simulation=NULL;
        /// MechanicalObject
        typename MechanicalObject::SPtr mecaObj=NULL;
        /// Time step
        double timeStep=0.01;
        /// Expected result
        double final_expected_value=0.0;

        /// Create the context for the scene
        void SetUp()
        {
            // Init simulation
            sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());
            root = simulation::getSimulation()->createNewGraph("root");
        }

        // Create the scene and the components: export velocity, VariationalSymplecticSolver is exact is velocity
        void createScene()
        {
            timeStep = 0.01;
            root->setGravity(Coord(0.0,0.0,0.0)); // no need of gravity, the file .data is just read
            root->setDt(timeStep);

            simulation::Node::SPtr childNode = root->createChild("Particle");

            mecaObj = New<MechanicalObject>();
            mecaObj->resize(1);
            childNode->addObject(mecaObj);

            sofa::component::misc::ReadState::SPtr readState =New<sofa::component::misc::ReadState>();
            readState->d_filename.setValue(std::string(SOFAGENERALLOADER_TESTFILES_DIR)+"particleGravityX.data");
            childNode->addObject(readState);

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
        bool simulation_result_test()
        {
            double result;
            result = mecaObj->x.getValue()[0][2];
            final_expected_value = -0.017658;

            EXPECT_TRUE( fabs(final_expected_value-result)<std::numeric_limits<double>::epsilon() );
            return true;
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
    TYPED_TEST_CASE(ReadState_test, DataTypes);

    // Test : read positions of a particle falling under gravity
    TYPED_TEST( ReadState_test , test_read_position)
    {
        this->SetUp();
        this->createScene();
        this->initScene();
        this->runScene();

        ASSERT_TRUE( this->simulation_result_test() );
        this->TearDown();
    }
}
