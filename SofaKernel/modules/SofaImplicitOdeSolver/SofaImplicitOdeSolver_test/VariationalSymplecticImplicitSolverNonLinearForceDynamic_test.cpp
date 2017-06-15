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
#include <SofaGeneralImplicitOdeSolver/VariationalSymplecticSolver.h>
#include <SofaBaseLinearSolver/CGLinearSolver.h>

#include <sofa/defaulttype/VecTypes.h>

// For atan2 ??
//#include <sofa/helper/rmath.h>

namespace sofa {

using namespace component;
using namespace defaulttype;
using namespace simulation;
using namespace modeling;
using helper::vector;

/**  Dynamic solver test.
Test the dynamic behavior of solver with non linear force : study a sun-planet system with gravitational force.
Create a simulation with 2 mechanical objects: one MO fixed (the sun) and one MO (the planet) with a
initial velocity perpendicular to the initial sun-planet axis.
The time step h is computed with the period T of the sun-planet system:
h = T/nbIter.
From simulated positions (simu_x;simu_y) compute the angle of the ellipse
angle = atan(simu_y/simu_x)
then compute the radius
radius = (1-e)*r0/(1-e*cos(angle)) with r0= initialRadius and e = eccentricity = ABS(r0*v0*v0/(g*M)-1) (v0 initial velocity)
and finally the expected trajectory
expected_x = radius*cos(angle)
expected-y = radius*sin(angle)
Check if simulated positions are on the right trajectory, that is to say compare simulated and expected positions.
Check if positions are correct by comparing the expected planet position to the simulated position at period T.
Check if Hamiltonian energy is constant during simulation: variational solver is used with incremental potential energy.
*/

template <typename _DataTypes>
struct VariationalSymplecticImplicitSolverNonLinearForceDynamic_test : public Elasticity_test<_DataTypes>
{
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

    typedef container::MechanicalObject<DataTypes> MechanicalObject;
    typedef component::odesolver::VariationalSymplecticSolver VariationalSymplecticSolver;
    typedef component::linearsolver::CGLinearSolver<component::linearsolver::GraphScatteredMatrix, component::linearsolver::GraphScatteredVector> CGLinearSolver;


    /// Root of the scene graph
    simulation::Node::SPtr root;      
    /// Tested simulation
    simulation::Simulation* simulation;  
    /// Position and velocity array
    vector<Real> positionsArray;
    vector<Real> velocitiesArray;
    vector<Real> energiesArray;

    // Variational solver
    VariationalSymplecticSolver::SPtr variationalSolver;

    // totalEnergy
    double totalEnergy;
    
    /// Create the context for the scene
    void createScene(double g, double M, double m, double x0_x, double v0_y)
    {
        // Init simulation
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());
        root = simulation::getSimulation()->createNewGraph("root");

        // Create the scene
        root->setGravity(Coord(0,0,0));

        // Solver Variational
        variationalSolver = addNew<VariationalSymplecticSolver> (root);
        variationalSolver->f_rayleighStiffness.setValue(0);
        variationalSolver->f_rayleighMass.setValue(0);
        variationalSolver->f_newtonError.setValue(1e-12);//1e-18
        variationalSolver->f_newtonSteps.setValue(4);//7
        variationalSolver->f_computeHamiltonian.setValue(1);
        variationalSolver->f_saveEnergyInFile.setValue(0);

        CGLinearSolver::SPtr cgLinearSolver = addNew<CGLinearSolver> (root);
        cgLinearSolver->f_maxIter=3000;
        cgLinearSolver->f_tolerance =1e-12;
        cgLinearSolver->f_smallDenominatorThreshold=1e-12;

        // Set initial positions and velocities of fixed point and mass
        Coord xSun = Coord(0,0,0);
        //MechanicalObject3::DataTypes::set( xSun[0], 0., 0.,0.);
        Deriv vSun = Deriv(0,0,0);
        //MechanicalObject3::DataTypes::set( vSun[0], 0.,0.,0.);
        Coord xPlanet = Coord(x0_x,0,0);
        //MechanicalObject3::DataTypes::set( xPlanet[0], 1., 0.,0.);
        Deriv vPlanet = Deriv(0,v0_y,0);
        //MechanicalObject3::DataTypes::set( vPlanet[0], 0., 0.18, 0.);

        // Sun planet system
        root = this->createSunPlanetSystem(
                root,           // add sun planet system to the node containing solver
                M,              // sun mass
                m,              // planet mass
                g,              // gravity constant
                xSun,           // Initial position of sun
                vSun,           // Final velocity of sun
                xPlanet,        // Initial position of planet
                vPlanet);       // Initial velocity of planet

    }

    /// Check if planet positions are on the right trajectory
    bool comparePlanetPositionsAndTrajectory (double g, double v0, double r0,double m,double M, double nbIter, double toleranceTrajectory, double tolerancePosition, double toleranceEnergy)
    {
        // Set parameters
        double e = fabs(r0*v0*v0/(g*M) - 1); // eccentricity
        double a = r0*g*M/(2*g*M-r0*v0*v0);
        double period = 2*M_PI*a*a*sqrt(1-e*e)/(r0*v0);
        double h = period*1.0/nbIter;

        // totalEnergy
        totalEnergy = 0.5*m*v0*v0 - m*M*g/sqrt(r0*r0);

        int i = 0;

        // Init simulation
        sofa::simulation::getSimulation()->init(root.get());
        double time = root->getTime();

        // Get mechanical object
        typename MechanicalObject::SPtr dofs = root->get<MechanicalObject>(root->SearchDown);

        // Variables
        double angle, radius, truePosX, truePosY;

        // Animate
        do
        {
            // Record the planet position
            Coord p0=dofs.get()->read(sofa::core::ConstVecCoordId::position())->getValue()[1];

            // Compute angle
            angle = atan2(p0[1],p0[0]);

            // Compute radius
            radius = (1-e)*r0/(1-e*cos(angle));

            // Expected positions
            truePosX = radius*cos(angle);
            truePosY = radius*sin(angle);

            double absoluteError = sqrt((p0[0]-truePosX)*(p0[0]-truePosX)+(p0[1]-truePosY)*(p0[1]-truePosY));

            // Compare mass position to the theoretical position
            if( absoluteError > toleranceTrajectory )
            {
                ADD_FAILURE() << "Position of planet at time " << time << " is wrong: "  << std::endl
                    <<" expected Position is " << truePosX << " , " << truePosY << std::endl
                    <<" actual Position is   " << p0[0] << " , " << p0[1] << std::endl
                    << "absolute error     = " << absoluteError << std::endl;
                return false;
            }

            //Animate
            sofa::simulation::getSimulation()->animate(root.get(),h);
            time = root->getTime();

            // Check energy is constant when there is no damping
            if(fabs(variationalSolver->f_hamiltonianEnergy.getValue() -totalEnergy) > toleranceEnergy )
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
        while (i < nbIter);

        // Check period T
       if(i == nbIter) // normally true
       {
           double absoluteError = sqrt((r0-truePosX)*(r0-truePosX)+(0-truePosY)*(0-truePosY));

           // Compare mass position to the theoretical position
           if( absoluteError > tolerancePosition )
           {
               ADD_FAILURE() << "Position of planet at period " << time << " is wrong: "  << std::endl
                   <<" expected Position is " << truePosX << " , " << truePosY << std::endl
                   <<" actual Position is   " << r0 << " , " << 0 << std::endl
                   << "absolute error     = " << absoluteError << std::endl;
               return false;
           }
       }

        return true;
    }


};

// Define the list of DataTypes to instanciate
using testing::Types;
typedef Types<
    Vec3Types
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(VariationalSymplecticImplicitSolverNonLinearForceDynamic_test, DataTypes);

// Test case: h=0.001
TYPED_TEST( VariationalSymplecticImplicitSolverNonLinearForceDynamic_test , variationalSymplecticImplicitSolverGravitationForceTest)
{
   this->createScene(0.075, 1, 1, 1, 0.18); // g, M, m, x0, v0

   // Compute potential energy from force in variational solver
   this->comparePlanetPositionsAndTrajectory (0.075, 0.18, 1, 1, 1, 1700, 8e-5, 1.1e-3, 1e-14);
   //g, v0, r0, M, m, nbIter, toleranceTrajectory, tolerancePosition, toleranceEnergy

}

} // namespace sofa
