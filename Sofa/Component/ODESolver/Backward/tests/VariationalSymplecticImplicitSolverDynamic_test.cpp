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

// Solvers
#include <sofa/component/odesolver/backward/VariationalSymplecticSolver.h>

#include <sofa/defaulttype/VecTypes.h>

namespace sofa {

using namespace component;
using namespace defaulttype;
using namespace simulation;

/**  Dynamic solver test.
Test the dynamic behavior of solver: study a mass-spring system under gravity initialize with spring rest length it will oscillate around its equilibrium position if there is no damping.
The movement follows the equation:
x(t)= A cos(wt + phi) with w the pulsation w=sqrt(K/M), K the stiffness, M the mass and phi the phase.
In this test: x(t=0)= 1 and v(t=0)=0 and K = spring stiffness and phi = 0 of material thus x(t)= cos(wt)
This tests generates the discrete mass position obtained with variational symplectic solver with different parameter values (K,M,h).
Then it compares the effective mass position to the computed mass position every time step.
*/

template <typename _DataTypes>
struct VariationalSymplecticImplicitSolverDynamic_test : public component::odesolver::testing::ODESolverSpringTest
{
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::Coord Coord;

    typedef statecontainer::MechanicalObject<DataTypes> MechanicalObject;
    typedef component::odesolver::backward::VariationalSymplecticSolver VariationalSymplecticSolver;

    /// Position and velocity array
    type::vector<double> positionsArray;
    type::vector<double> velocitiesArray;
    type::vector<double> energiesArray;

    // totalEnergy
    double totalEnergy;

    /// Create the context for the scene
    void createScene(double K, double m, double l0, double rm=0, double rk=0)
    {
        this->prepareScene(K, m, l0);

        // add ODE Solver to test
        simpleapi::createObject(m_si.root, "VariationalSymplecticSolver", {
            { "rayleighStiffness", simpleapi::str(rk)},
            { "rayleighMass", simpleapi::str(rm)},
            { "computeHamiltonian", "1"},
            { "newtonError", simpleapi::str(1e-12)},
            { "newtonSteps", simpleapi::str(4)}
            });
    }


    /// Generate discrete mass position values with variational sympletic implicit solver
    void generateDiscreteMassPositions (double h, double K, double m, double z0, double v0,double g, double finalTime, double rm,double rk)
    {
        int size = 0 ;

        // During t=finalTime
        if((finalTime/h) > 0)
        {
            size = int(finalTime/h);
            positionsArray.reserve(size);
            velocitiesArray.reserve(size);
            energiesArray.reserve(size);
        }

        // First velocity is v0
        velocitiesArray.push_back(v0);

        // First acceleration
        energiesArray.push_back(m*v0);

        // First position is z0
        positionsArray.push_back(double(z0));

        // Compute totalEnergy
        totalEnergy = m*g*z0; // energy at initial time

        // Set constants
        const double denominator = 4*m+h*h*K+4*h*(rm*m + rk*K);//4*h*(-rk*K+rm*m);
        const double constant = -h*K;

        // Compute velocities, energies and positions
        for(int i=1;i< size+1; i++)
        {
            velocitiesArray.push_back(2*(-m*g*h+constant*(positionsArray[i-1]-z0)+2*energiesArray[i-1])/denominator);
            energiesArray.push_back(m*velocitiesArray[i]+h*(-K*(positionsArray[i-1]- z0+velocitiesArray[i]*h/2) -m*g)/2);
            positionsArray.push_back(positionsArray[i-1]+h*velocitiesArray[i]);
        }

    }

    /// After simulation compare the positions of points to the theoretical positions.
    bool compareSimulatedToTheoreticalPositions( double h, double tolerancePosition, double toleranceEnergy = 1e-13, bool checkEnergyConservation=false)
    {
        int i = 0;
        // Init simulation
        m_si.initScene();
        double time = m_si.root->getTime();

        // Get mechanical object
        const simulation::Node::SPtr massNode = m_si.root->getChild("MassNode");
        typename statecontainer::MechanicalObject<_DataTypes>::SPtr dofs = massNode->get<statecontainer::MechanicalObject<_DataTypes>>(m_si.root->SearchDown);
        const typename VariationalSymplecticSolver::SPtr variationalSolver = m_si.root->get<VariationalSymplecticSolver>(m_si.root->SearchDown);
        

        // Animate
        do
        {
            // Record the mass position
            Coord p0=dofs.get()->read(sofa::core::ConstVecCoordId::position())->getValue()[0];

            double absoluteError = fabs(p0[1]-positionsArray[i]);

            // Compare mass position to the theoretical position
            if( absoluteError > tolerancePosition )
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


            // Check if hamiltonian energy is constant when there is no damping
            if(checkEnergyConservation && fabs(variationalSolver->f_hamiltonianEnergy.getValue() -totalEnergy) > toleranceEnergy )
            {
                ADD_FAILURE() << "Hamiltonian energy at time " << time << " is wrong: "  << std::endl
                    <<" expected Energy is " << totalEnergy << std::endl
                    <<" actual Energy is   " << variationalSolver->f_hamiltonianEnergy.getValue() << std::endl
                    << "absolute error     = " << fabs(variationalSolver->f_hamiltonianEnergy.getValue() -totalEnergy) << std::endl;
                return false;
            }

            // Iterate
            i++;
        }
        while (time < 2);
        return true;
    }

};

// Define the list of DataTypes to instanciate
using ::testing::Types;
typedef Types<
    Vec3Types
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_SUITE(VariationalSymplecticImplicitSolverDynamic_test, DataTypes);

// Test case: h=0.1 k=100 m =1 rm=0.1 rk=0.1
TYPED_TEST( VariationalSymplecticImplicitSolverDynamic_test , variationalSymplecticImplicitSolverDynamicTest_high_dt_without_damping)
{
   this->createScene(100,1,1,0,0); // k,m,l0
   // rm = 0.1  and rk=0
   this->generateDiscreteMassPositions (0.01, 100, 1, 1, 0,10, 2, 0 ,0);
   // With incremental potential energy
   this-> compareSimulatedToTheoreticalPositions(0.01,9e-15,8e-15,true);
}

// Test case: h=0.01 K=100 m = 10 rm=0.1 rk=0
TYPED_TEST( VariationalSymplecticImplicitSolverDynamic_test , variationalSymplecticImplicitSolverDynamicTest_medium_dt_with_rayleigh_mass)
{
   this->createScene(100,10,1,0.1,0); // k,m,l0
   this->generateDiscreteMassPositions (0.01, 100, 10, 1, 0,10, 2, 0.1,0);
   this-> compareSimulatedToTheoreticalPositions(0.01, 7e-15);
}

// Test case: h=0.01 k=100 m=10 rm=0 rk=0.1
TYPED_TEST( VariationalSymplecticImplicitSolverDynamic_test , variationalSymplecticImplicitSolverDynamicTest_medium_dt_with_rayleigh_stiffness)
{
   this->createScene(100,10,1,0,0.1); // k,m,l0
   this->generateDiscreteMassPositions (0.01, 100, 10, 1, 0,10, 2, 0, 0.1);
   this-> compareSimulatedToTheoreticalPositions(0.01, 3.3e-15);
}

// Test case: h=0.001 k=100 m =10 rm=0.1 rk=0.1
TYPED_TEST( VariationalSymplecticImplicitSolverDynamic_test , variationalSymplecticImplicitSolverDynamicTest_small_dt_with_damping)
{
   this->createScene(100,10,1,0.1,0.1); // k,m,l0
   // rm = 0.1  and rk=0
   this->generateDiscreteMassPositions (0.001, 100, 10, 1, 0,10, 2, 0.1 ,0.1);
   this-> compareSimulatedToTheoreticalPositions(0.001, 6.3e-15);
}

} // namespace sofa
