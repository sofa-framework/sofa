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
#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest;
#include <sofa/testing/NumericTest.h>
using sofa::testing::NumericTest;

#include <sofa/component/odesolver/testing/ODESolverSpringTest.h>

//Including Simulation
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/Node.h>

// Including mechanical object
#include <sofa/component/statecontainer/MechanicalObject.h>
using MechanicalObject3 = sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Vec3Types> ;

#include <sofa/defaulttype/VecTypes.h>

namespace sofa {

using namespace component;
using namespace defaulttype;
using namespace simulation;
using type::vector;

/**  Dynamic solver test.
Test the dynamic behavior of solver: study a mass-spring system under gravity initialize with spring rest length it will oscillate around its equilibrium position if there is no damping.
The movement follows the equation:
x(t)= A cos(wt + phi) with w the pulsation w=sqrt(K/M), K the stiffness, M the mass and phi the phase.
In this test: x(t=0)= 1 and v(t=0)=0 and K = spring stiffness and phi = 0 of material thus x(t)= cos(wt)
This tests generates the discrete mass position obtained with rungeKutta2 solver with different parameter values (K,M,h).
Then it compares the effective mass position to the computed mass position every time step.
*/

template <typename _DataTypes>
struct RungeKutta2ExplicitSolverDynamic_test : public component::odesolver::testing::ODESolverSpringTest
{
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::Coord Coord;

    typedef statecontainer::MechanicalObject<DataTypes> MechanicalObject;

    /// Position, velocity and acceleration array
    vector<double> positionsArray;
    vector<double> velocitiesArray;
    vector<double> accelerationsArray;
    
    /// Create the context for the scene
    void createScene(double K, double m, double l0)
    {
        this->prepareScene(K, m, l0);
        // add ODE Solver to test
        simpleapi::createObject(m_si.root, "RungeKutta2Solver", {
            { }
        });

    }


    /// Generate discrete mass position values with euler implicit solver
    void generateDiscreteMassPositions (double h, double K, double m, double z0, double v0,double g, double finalTime)
    {
        int size = 0 ;

        // During t=finalTime
        if((finalTime/h) > 0)
        {
            size = int(finalTime/h);
            positionsArray.reserve(size);
            velocitiesArray.reserve(size);
            accelerationsArray.reserve(size);
        }

        // First acceleration
        accelerationsArray.push_back((-K*(z0-z0)-m*g)/m);

        // First position is z0
        positionsArray.push_back(double(z0));

        // First velocity is v0
        velocitiesArray.push_back(v0);

        for(int i=1;i< size+1; i++)
        {
            // At time t + h/2
            const double newPos = positionsArray[i-1]+0.5*h*velocitiesArray[i-1];
            const double newVel = velocitiesArray[i-1]+0.5*h*accelerationsArray[i-1];
            const double acc    = (-K*(newPos-z0)-m*g)/m;

            // At time t + h
            positionsArray.push_back(positionsArray[i-1]+h*newVel);
            velocitiesArray.push_back(velocitiesArray[i-1]+h*acc);
            accelerationsArray.push_back((-K*(positionsArray[i]-z0)-m*g)/m);

        }

    }

    /// After simulation compare the positions of points to the theoretical positions.
    bool compareSimulatedToTheoreticalPositions(double tolerance,double h)
    {
        int i = 0;
        // Init simulation
        m_si.initScene();
        double time = m_si.root->getTime();

        // Get mechanical object
        const simulation::Node::SPtr massNode = m_si.root->getChild("MassNode");
        typename MechanicalObject::SPtr dofs = massNode->get<MechanicalObject>(m_si.root->SearchDown);

        // Animate
        do
        {              
            // Record the mass position
            Coord p0=dofs.get()->read(sofa::core::vec_id::read_access::position)->getValue()[0];

            double absoluteError = fabs(p0[1]-positionsArray[i]);

            // Compare mass position to the theoretical position
            if( absoluteError > tolerance )
            {
                ADD_FAILURE() << "Position of mass at time " << time << " is wrong: "  << std::endl
                    <<" expected Position is " << positionsArray[i] << std::endl
                    <<" actual Position is   " << p0[1] << std::endl
                    << "absolute error     = " << absoluteError << std::endl;
                return false;
            }

            //Animate
            m_si.simulate(h);
            time = m_si.root->getTime();
            // Iterate
            i++;
        }
        while (time < 2);
        return true;
    }

};

// Define the list of DataTypes to instantiate
using ::testing::Types;
typedef Types<
    Vec3Types
> DataTypes; // the types to instantiate.

// Test suite for all the instantiations
TYPED_TEST_SUITE(RungeKutta2ExplicitSolverDynamic_test, DataTypes);

// Test case: h=0.1 k=100 m =10 rm=0.1 rk=0.1
TYPED_TEST( RungeKutta2ExplicitSolverDynamic_test , rungeKutta2ExplicitSolverDynamicTest_medium_dt)
{
   this->createScene(100,10,1); // k,m,l0
   this->generateDiscreteMassPositions (0.01, 100, 10, 1, 0,10, 2);
   this-> compareSimulatedToTheoreticalPositions(5e-16,0.01);
}

// Test case: h=0.1 K=1000 m = 10
TYPED_TEST( RungeKutta2ExplicitSolverDynamic_test , rungeKutta2ExplicitSolverDynamicTest_high_dt)
{
   this->createScene(1000,10,1); // k,m,l0
   this->generateDiscreteMassPositions (0.1, 1000, 10, 1, 0, 10, 2);
   this-> compareSimulatedToTheoreticalPositions(5e-16,0.1);
}

// Test case: h=0.001 k=100 m=100
TYPED_TEST( RungeKutta2ExplicitSolverDynamic_test , rungeKutta2ExplicitSolverDynamicTest_small_dt)
{
   this->createScene(100,100,1); // k,m,l0
   this->generateDiscreteMassPositions (0.001, 100, 100, 1, 0, 10, 2);
   this-> compareSimulatedToTheoreticalPositions(9e-16,0.001);
}

} // namespace sofa
