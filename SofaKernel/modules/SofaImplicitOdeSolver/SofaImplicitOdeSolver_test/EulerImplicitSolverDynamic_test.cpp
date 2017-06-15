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
#include <SofaTest/Elasticity_test.h>
#include <SceneCreator/SceneCreator.h>

#include <sofa/core/ExecParams.h>

//Including Simulation
#include <sofa/simulation/Simulation.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/simulation/Node.h>

// Including mechanical object
#include <SofaBaseMechanics/MechanicalObject.h>

// Solvers
#include <SofaImplicitOdeSolver/EulerImplicitSolver.h>
#include <SofaBaseLinearSolver/CGLinearSolver.h>

#include <sofa/defaulttype/VecTypes.h>

namespace sofa {

using namespace component;
using namespace defaulttype;
using namespace simulation;
using namespace modeling;
using helper::vector;

/**  Dynamic solver test.
Test the dynamic behavior of solver: study a mass-spring system under gravity initialize with spring rest length it will oscillate around its equilibrium position if there is no damping.
The movement follows the equation:
x(t)= A cos(wt + phi) with w the pulsation w=sqrt(K/M), K the stiffness, M the mass and phi the phase.
In this test: x(t=0)= 1 and v(t=0)=0 and K = spring stiffness and phi = 0 of material thus x(t)= cos(wt)
This tests generates the discrete mass position obtained with euler implicit solver with different parameter values (K,M,h).
Then it compares the effective mass position to the computed mass position every time step.
*/

template <typename _DataTypes>
struct EulerImplicitDynamic_test : public Elasticity_test<_DataTypes>
{
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::Coord Coord;

    typedef container::MechanicalObject<DataTypes> MechanicalObject;
    typedef component::odesolver::EulerImplicitSolver EulerImplicitSolver;
    typedef component::linearsolver::CGLinearSolver<component::linearsolver::GraphScatteredMatrix, component::linearsolver::GraphScatteredVector> CGLinearSolver;


    /// Root of the scene graph
    simulation::Node::SPtr root;      
    /// Tested simulation
    simulation::Simulation* simulation;  
    /// Position and velocity array
    vector<double> positionsArray;
    vector<double> velocitiesArray;

    
    /// Create the context for the scene
    void createScene(double K, double m, double l0, double rm = 0, double rk=0)
    { 
        // Init simulation
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());
        root = simulation::getSimulation()->createNewGraph("root");

        // Create the scene
        root->setGravity(Coord(0,-10,0));

        // Solver
        EulerImplicitSolver::SPtr eulerSolver = addNew<EulerImplicitSolver> (root);
        eulerSolver->f_rayleighStiffness.setValue(rk);
        eulerSolver->f_rayleighMass.setValue(rm);

        CGLinearSolver::SPtr cgLinearSolver = addNew<CGLinearSolver>   (root);
        cgLinearSolver->f_maxIter=3000;
        cgLinearSolver->f_tolerance =1e-9;
        cgLinearSolver->f_smallDenominatorThreshold=1e-9;

        // Set initial positions and velocities of fixed point and mass
        MechanicalObject3::VecCoord xFixed(1);
        MechanicalObject3::DataTypes::set( xFixed[0], 0., 2.,0.);
        MechanicalObject3::VecDeriv vFixed(1);
        MechanicalObject3::DataTypes::set( vFixed[0], 0.,0.,0.);
        MechanicalObject3::VecCoord xMass(1);
        MechanicalObject3::DataTypes::set( xMass[0], 0., 1.,0.);
        MechanicalObject3::VecDeriv vMass(1);
        MechanicalObject3::DataTypes::set( vFixed[0], 0.,0.,0.);

        // Add mass spring system
        root = this-> createMassSpringSystem(
                root,   // add mass spring system to the node containing solver
                K,      // stiffness
                m,      // mass
                l0,     // spring rest length
                xFixed, // Initial position of fixed point
                vFixed, // Initial velocity of fixed point
                xMass,  // Initial position of mass
                vMass); // Initial velocity of mass

    }

    /// Generate discrete mass position values with euler implicit solver
    void generateDiscreteMassPositions (double h, double K, double m, double z0, double v0,double g, double finalTime, double rm, double rk)
    {
        int size = 0 ;

        // During t=finalTime
        if((finalTime/h) > 0)
        {
            size = int(finalTime/h);
            positionsArray.reserve(size);
            velocitiesArray.reserve(size);
        }

        // First position is z0
        positionsArray.push_back(double(z0));

        // First velocity is v0
        velocitiesArray.push_back(v0);

        // Compute velocities
        double denominator = h*(h+rk)*K+(1+h*rm)*m;
        double constant = (-(rk+h)*K-rm*m);

        for(int i=1;i< size+1; i++)
        {
            velocitiesArray.push_back(velocitiesArray[i-1]+h*(-K*(positionsArray[i-1]-z0)-m*g+velocitiesArray[i-1]*constant)/(denominator));
            positionsArray.push_back(positionsArray[i-1]+h*velocitiesArray[i]);
        }

    }

    /// After simulation compare the positions of points to the theoretical positions.
    bool compareSimulatedToTheoreticalPositions(double tolerance,double h)
    {
        int i = 0;
        // Init simulation
        sofa::simulation::getSimulation()->init(root.get());
        double time = root->getTime();

        // Get mechanical object
        simulation::Node::SPtr massNode = root->getChild("MassNode");
        typename MechanicalObject::SPtr dofs = massNode->get<MechanicalObject>(root->SearchDown);

        // Animate
        do
        {              
            // Record the mass position
            Coord p0=dofs.get()->read(sofa::core::ConstVecCoordId::position())->getValue()[0];

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
            sofa::simulation::getSimulation()->animate(root.get(),h);
            time = root->getTime();
            // Iterate
            i++;
        }
        while (time < 2);
        return true;
    }

};

// Define the list of DataTypes to instanciate
using testing::Types;
typedef Types<
    Vec3Types
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(EulerImplicitDynamic_test, DataTypes);

// Test case: h=0.1 k=100 m =10 rm=0 rk=0
TYPED_TEST( EulerImplicitDynamic_test , eulerImplicitSolverDynamicTest_high_dt_without_damping)
{
   this->createScene(100,10,1,0,0); // k,m,l0,rm,rk
   this->generateDiscreteMassPositions (0.1, 100, 10, 1, 0, 10, 2, 0, 0);
   this-> compareSimulatedToTheoreticalPositions(5e-16,0.1);
}

// Test case: h=0.1 k=100 m =10 rm=0.1 rk=0.1
TYPED_TEST( EulerImplicitDynamic_test , eulerImplicitSolverDynamicTest_high_dt_with_damping)
{
   this->createScene(100,10,1,0.1,0.1); // k,m,l0,rm,rk
   this->generateDiscreteMassPositions (0.1, 100, 10,1, 0, 10, 2, 0.1, 0.1);
   this-> compareSimulatedToTheoreticalPositions(5e-16,0.1);
}

// Test case: h=0.01 K=10 m=10 rm=0 rk=0.1
TYPED_TEST( EulerImplicitDynamic_test , eulerImplicitSolverDynamicTest_medium_dt_with_rayleigh_stiffness)
{
   this->createScene(10,10,1,0,0.1); // k,m,l0,rm,rk
   this->generateDiscreteMassPositions (0.01, 10, 10, 1, 0, 10, 2, 0, 0.1);
   this-> compareSimulatedToTheoreticalPositions(6e-15,0.01);
}

// Test case: h=0.001 K=10 m = 100 rm=0.1 rk=0
TYPED_TEST( EulerImplicitDynamic_test , eulerImplicitSolverDynamicTest_small_dt_with_rayleigh_mass)
{
   this->createScene(10,100,1,0.1,0); // k,m,l0,rm,rk
   this->generateDiscreteMassPositions (0.001, 10, 100, 1, 0, 10, 2, 0.1, 0);
   this-> compareSimulatedToTheoreticalPositions(5e-16,0.001);
}

} // namespace sofa
