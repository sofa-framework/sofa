/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include <SofaTest/Sofa_test.h>
#include <SofaTest/TestMessageHandler.h>


#include <sofa/core/ExecParams.h>

//Including Simulation
#include <sofa/simulation/Simulation.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/simulation/Node.h>

// Including mechanical object
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/MechanicalParams.h>

#include <sofa/defaulttype/VecTypes.h>

namespace sofa {

using namespace component;

/**  Dynamic solver test.
Test the dynamic behavior of solver: study a mass-spring system under gravity initialize with spring rest length it will oscillate around its equilibrium position if there is no damping.
The movement follows the equation:
x(t)= A cos(wt + phi) with w the pulsation w=sqrt(K/M), K the stiffness, M the mass and phi the phase.
In this test: x(t=0)= 1 and v(t=0)=0 and K = spring stiffness and phi = 0 of material thus x(t)= cos(wt)
This test compares siumlated mass position to analytic mass position during 2s every time step (here dt=0.001).
*/

template <typename _DataTypes>
struct SpringSolverDynamic_test : public Sofa_test<typename _DataTypes::Real>
{
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::Coord Coord;

    typedef container::MechanicalObject<DataTypes> MechanicalObject;

    /// Root of the scene graph
    simulation::Node::SPtr root;
    /// Tested simulation
    simulation::Simulation* simulation;


    /// Create the context for the scene
    void SetUp()
    {
        // Init simulation
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());
        root = simulation::getSimulation()->createNewGraph("root");
    }

    void loadScene(std::string sceneName)
    {
        // Load the scene from the xml file
        std::string fileName = std::string(SOFAIMPLICITODESOLVER_TEST_SCENES_DIR) + "/" + sceneName;
        root = sofa::simulation::getSimulation()->load(fileName.c_str());
    }

    /// After simulation compare the positions of points to the theoretical positions.
    bool compareSimulatedToTheoreticalPositions(double tolerance)
    {
        // Init simulation
        sofa::simulation::getSimulation()->init(root.get());
        double time = root->getTime();
        double stiffnessSpring = 100;
        double mass = 10;
        double w = sqrt(stiffnessSpring/mass);

        // Get mechanical object
        simulation::Node::SPtr massNode = root->getChild("MassNode");
        typename MechanicalObject::SPtr dofs = massNode->get<MechanicalObject>(root->SearchDown);

        // Animate
        do
        {
            // Record the mass position
            Coord p0=dofs.get()->read(sofa::core::ConstVecCoordId::position())->getValue()[0];

            // Absolute error
            double absoluteError = fabs(p0[1]-(cos(w*time)));

            // Compare mass position to the theoretical position
            if( absoluteError > tolerance )
            {
                ADD_FAILURE() << "Position of mass at time " << time << " is wrong: "  << std::endl
                    <<" expected Position is " << cos(sqrt(stiffnessSpring/mass)*time) << std::endl
                    <<" actual Position is   " << p0[1] << std::endl
                    << "absolute error     = " << absoluteError << std::endl;
                return false;
            }

            //Animate
            sofa::simulation::getSimulation()->animate(root.get(),0.001);
            time = root->getTime();
        }
        while (time < 2);
        return true;
    }

};

// Define the list of DataTypes to instanciate
using testing::Types;
typedef Types<
    defaulttype::Vec3Types
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(SpringSolverDynamic_test, DataTypes);

// Test case EulerImplicit Solver
TYPED_TEST( SpringSolverDynamic_test , EulerImplicitSolverDynamicTest )
{
   EXPECT_MSG_NOEMIT(Error) ;
   this->loadScene("EulerImplicitSpringDynamicTest.xml");
   ASSERT_TRUE( this->compareSimulatedToTheoreticalPositions(0.01));
}

// Test case NewmarkImplicit Solver
TYPED_TEST( SpringSolverDynamic_test , NewmarkImplicitSolverDynamicTest )
{
   this->loadScene("NewmarkSpringDynamicTest.xml");
   ASSERT_TRUE( this->compareSimulatedToTheoreticalPositions(0.004));
}

} // namespace sofa
