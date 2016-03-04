#include "Compliant_test.h"
#include "../numericalsolver/MinresSolver.h"
#include "../odesolver/CompliantImplicitSolver.h"

#include <SofaBaseMechanics/IdentityMapping.h>
#include <SofaExplicitOdeSolver/EulerSolver.h>
#include <SceneCreator/SceneCreator.h>

#include <SofaBaseMechanics/IdentityMapping.h>

namespace sofa
{

using namespace component;
using namespace modeling;
using helper::vector;


/**
 * @brief Integrate a one-dimensional damped oscillator in time, using an CompliantImplicitSolver.
 * The oscillator is composed of one fixed and one moving particle, connected by a spring with damping.
 * We check if the trajectory x(t) of the particle corresponds to the theory
 */
struct DampedOscillator_test : public CompliantSolver_test
{
    SReal mass;
    SReal stiffness;
    SReal damping;
    SReal x0;
    SReal v0;

    simulation::Node::SPtr node;
    MechanicalObject1::SPtr DOF;
    UniformCompliance1::SPtr compliance;

    /**
     * @brief Create the physical model
     *
     * @param m Mass of the particle
     * @param s Stiffness of the spring
     * @param d Damping ratio of the spring
     * @param xx0 initial position
     * @param vv0 initial velocity
     */
    void setup(
            SReal m, SReal s, SReal d,
            SReal xx0, SReal vv0
            )
    {
        mass = m;
        stiffness = s;
        damping = d;
        x0=xx0; v0=vv0;

        node = clearScene();
        node->setGravity( Vec3(0,0,0) );

        // The oscillator
        simulation::Node::SPtr oscillator = node->createChild("oscillator");

        DOF = addNew<MechanicalObject1>(oscillator,"DOF");
        DOF->resize(1);
        DOF->writePositions()[0]  = Vec1(x0);
        DOF->writeVelocities()[0] = Vec1(v0);

        UniformMass1::SPtr Mass = addNew<UniformMass1>(oscillator,"mass");
        Mass->mass.setValue( mass );

        compliance = addNew<UniformCompliance1>(oscillator,"compliance");
        compliance->isCompliance.setValue(false);
        compliance->compliance.setValue(1.0/stiffness);
        compliance->damping.setValue(damping);
    }


    /**
     * @brief Perform the time integration over several time steps and compare with the theoretical solution
     * @param endTime Simulation stops when time is higher than this
     * @param dt Time step
     * @param tolerance  Admissible absolute error
     * @param debug Print debug info
     */
    void testTimeIntegration( SReal endTime, SReal dt, SReal tolerance, bool debug )
    {

        node->setDt(dt);

        //**************************************************
        sofa::simulation::getSimulation()->init(node.get());
        //**************************************************

        if(debug) simulation::getSimulation()->exportXML ( simulation::getSimulation()->GetRoot().get(), "/tmp/oscilator.scn" );


        //**************************************************
        // Simulation loop
        SReal t=0;
        printInfo(debug,t);
        while( t<endTime )
        {
            simulation::getSimulation()->animate(node.get(),dt);
            t+=dt ;
            printInfo(debug,t);

            SReal x = DOF->readPositions()[0][0];
            ASSERT_TRUE( fabs(x-theoreticalPosition(t)) < tolerance );
        }
        //**************************************************


    }

    /**
     * @brief Theoretical position of the oscillator as a function of time
     *  As given in http://www2.ulg.ac.be/mathgen/cours/meca/Ref/Oscillateurs.pdf
     * @param t Time
     * @param debug Print debug info
     * @return Theoretical position of the oscillator
     */
    SReal theoreticalPosition( SReal t )
    {
        SReal k = stiffness;
        SReal c = damping;
        SReal m = mass;
        SReal w0 = sqrt( k/m );
        SReal e = c/( 2*sqrt(k*m) );
        SReal epsilon = 100 * std::numeric_limits<SReal>::epsilon();
        SReal x;

        if( e<1-epsilon ) // damped oscillations
        {
            SReal w = w0 * sqrt(1-e*e);
            x = exp( -e*w0*t ) * ( ((v0+e*w0*x0)/w)*sin(w*t) + x0*cos(w*t) );
        }
        else if (e<1+epsilon) // critical damping
        {
            x = exp(-w0*t) * ( x0 + (v0+w0*x0)*t );
        }
        else // over-critical damping
        {
            SReal z1 = ( -e-sqrt(e*e-1) ) * w0;
            SReal z2 = ( -e+sqrt(e*e-1) ) * w0;
            x = (v0-z2*x0)/(z1-z2)*exp(z1*t) + (v0-z1*x0)/(z2-z1)*exp(z2*t);
        }

        return x;
    }

    void printInfo( bool debug , SReal t)
    {
        if( !debug )
            return;

        MechanicalObject1::ReadVecCoord X = DOF->readPositions();
        MechanicalObject1::ReadVecDeriv V = DOF->readVelocities();

        std::cout<<"computed x= "<< X[0][0] << " , v= " << V[0][0] << " , t= "<< t << std::endl;
        std::cout<<"   exact x= "<< theoreticalPosition(t) << std::endl;
    }


};

//=================
// do run the tests
//=================

// explicit, first-degree precision
TEST_F(DampedOscillator_test, explicitEuler )
{
    // === Physical parameters and initial conditions
    setup( 1.0, 1.0, 0.0, 1.0, 0.0 );// mass, stiffness, damping, x0, v0

    // === Numerical integrator
    odesolver::EulerSolver::SPtr eulerSolver = addNew<odesolver::EulerSolver>(node);
    eulerSolver->symplectic.setValue(false);

    // === Run the test
    SReal moreThanOneCycle = 7 * sqrt( 1.0/1.0 ); // 2*M_PI*sqrt(m/k)
    testTimeIntegration( moreThanOneCycle, 0.01, 0.04, false );      // dt, numIter, tolerance, debug info

}

// explicit, second-degree precision
TEST_F(DampedOscillator_test, symplecticEuler )
{
    // === Physical parameters and initial conditions
    setup( 1.0, 1.0, 0.0, 1.0, 0.0 );// mass, stiffness, damping, x0, v0

    // === Numerical integrator
    odesolver::EulerSolver::SPtr eulerSolver = addNew<odesolver::EulerSolver>(node);
    eulerSolver->symplectic.setValue(true);

    // === Run the test
    SReal moreThanOneCycle = 7 * sqrt( 1.0/1.0 ); // 2*M_PI*sqrt(m/k)
    testTimeIntegration( moreThanOneCycle, 0.01, 0.006, false );      // dt, numIter, tolerance, debug info

}


// implicit, first-degree precision
TEST_F(DampedOscillator_test, stiffness_first_degree )
{
    // === Physical parameters and initial conditions
    setup( 1.0, 1.0, 1.0, 1.0, 0.0 );// mass, stiffness, damping, x0, v0

    // === Numerical integrators
    odesolver::CompliantImplicitSolver::SPtr complianceSolver = addNew<odesolver::CompliantImplicitSolver>(node);
    complianceSolver->debug.setValue( false );
    complianceSolver->alpha.setValue(1.0);
    complianceSolver->beta.setValue(1.0);
    //
    linearsolver::LDLTSolver::SPtr linearSolver = addNew<linearsolver::LDLTSolver>(node);
    linearSolver->debug.setValue(false);
    component::linearsolver::LDLTResponse::SPtr response = addNew<component::linearsolver::LDLTResponse>(node);


    // === Run the test
    SReal moreThanOneCycle = 7 * sqrt( 1.0/1.0 ); // 2*M_PI*sqrt(m/k)
    testTimeIntegration( moreThanOneCycle, 0.01, 0.04, false );      // dt, numIter, tolerance, debug info

}

// implicit, second-degree precision
TEST_F(DampedOscillator_test, stiffness_second_degree )
{
    // === Physical parameters and initial conditions
    setup( 1.0, 1.0, 1.0, 1.0, 0.0 );// mass, stiffness, damping, x0, v0

    // === Numerical integrators
    odesolver::CompliantImplicitSolver::SPtr complianceSolver = addNew<odesolver::CompliantImplicitSolver>(node);
    complianceSolver->debug.setValue( false );
    complianceSolver->alpha.setValue(0.5);
    complianceSolver->beta.setValue(1.0);
    //
    linearsolver::LDLTSolver::SPtr linearSolver = addNew<linearsolver::LDLTSolver>(node);
    linearSolver->debug.setValue(false);
    component::linearsolver::LDLTResponse::SPtr response = addNew<component::linearsolver::LDLTResponse>(node);


    // === Run the test
    SReal moreThanOneCycle = 7 * sqrt( 1.0/1.0 ); // 2*M_PI*sqrt(m/k)
    testTimeIntegration( moreThanOneCycle, 0.01, 0.003, false );      // dt, numIter, tolerance, debug info

}

// implicit, second-degree precision, using compliance rather than forces
TEST_F(DampedOscillator_test, compliance_second_degree )
{
    // === Physical parameters and initial conditions
//    setup( 1.0, 1.0, 1.0, 1.0, 0.0 );// mass, stiffness, damping, x0, v0
//    compliance->isCompliance.setValue(true);

    // TODO CLEAN THIS

    // for a constraint the scene is different
    // a dof cannot be both independent and constrained


    mass = 1;
    stiffness = 1;
    damping = 1;
    x0=1; v0=0;

    node = clearScene();
    node->setGravity( Vec3(0,0,0) );

    // The oscillator
    simulation::Node::SPtr oscillator = node->createChild("oscillator");

    DOF = addNew<MechanicalObject1>(oscillator,"DOF");
    DOF->resize(1);
    DOF->writePositions()[0]  = Vec1(x0);
    DOF->writeVelocities()[0] = Vec1(v0);

    UniformMass1::SPtr Mass = addNew<UniformMass1>(oscillator,"mass");
    Mass->mass.setValue( mass );

    simulation::Node::SPtr constraint = oscillator->createChild( "constraint" );

    MechanicalObject1::SPtr constraintDOF = addNew<MechanicalObject1>(constraint,"DOF");
    constraintDOF->resize(1);

    typedef component::mapping::IdentityMapping<defaulttype::StdVectorTypes<defaulttype::Vec<1, SReal>, defaulttype::Vec<1, SReal>, SReal>,defaulttype::StdVectorTypes<defaulttype::Vec<1, SReal>, defaulttype::Vec<1, SReal>, SReal> > IdentityMapping11;
    IdentityMapping11::SPtr mapping = addNew< IdentityMapping11 >(constraint,"mapping");
    mapping->setModels(DOF.get(), constraintDOF.get());
    compliance = addNew<UniformCompliance1>(constraint,"compliance");
    compliance->isCompliance.setValue(true);
    compliance->compliance.setValue(1.0/stiffness);
    compliance->damping.setValue(damping);




    // === Numerical integrators
    odesolver::CompliantImplicitSolver::SPtr complianceSolver = addNew<odesolver::CompliantImplicitSolver>(node);
    complianceSolver->debug.setValue( false );
    complianceSolver->alpha.setValue(0.5);
    complianceSolver->beta.setValue(1.0);
//    complianceSolver->stabilization.setValue( false ); // what about regularization?
    //
    linearsolver::LDLTSolver::SPtr linearSolver = addNew<linearsolver::LDLTSolver>(node);
    linearSolver->debug.setValue(false);
    component::linearsolver::LDLTResponse::SPtr response = addNew<component::linearsolver::LDLTResponse>(node);


    // === Run the test
    SReal moreThanOneCycle = 7 * sqrt( 1.0/1.0 ); // 2*M_PI*sqrt(m/k)
    testTimeIntegration( moreThanOneCycle, 0.01, 0.003, false );      // dt, numIter, tolerance, debug info

}

}// sofa



