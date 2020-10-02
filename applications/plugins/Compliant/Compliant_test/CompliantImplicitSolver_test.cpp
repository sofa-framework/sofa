#include "Compliant_test.h"
#include "../numericalsolver/MinresSolver.h"
#include "../odesolver/CompliantImplicitSolver.h"

#include <SofaBoundaryCondition/FixedConstraint.h>
#include <SofaExplicitOdeSolver/EulerSolver.h>
#include <SceneCreator/SceneCreator.h>

#include <SofaTest/TestMessageHandler.h>


using namespace sofa::modeling;
using namespace sofa::component;
using namespace sofa::simulation;

namespace sofa
{

struct CompliantImplicitSolver_test : public CompliantSolver_test
{

    /** @defgroup CompliantImplicitSolver_Unit_Tests CompliantImplicitSolver basic tests.
     *
     * The scene is composed of two particles connected by a spring. One particle is fixed, while the other has an initial velocity.
     * The solver is set to backward Euler: alpha = beta = 1.
     */
    ///@{

    /** Test in the linear case, with velocity parallel to the spring.
      Convergence should occur in one iteration.
      Integrate using backward Euler (alpha=1, beta=1).
      Post-condition: an explicit Euler step with step -dt brings the system back to the original state.
      */
    void testLinearOneFixedOneStiffnessSpringV100(bool debug)
    {
        SReal dt=0.1; // currently, this must be set in the root node AND passed to the animate function
        Node::SPtr root = clearScene();
        root->setGravity( Vec3(0,0,0) );
        root->setDt(dt);

        // The solver
        using odesolver::CompliantImplicitSolver;
        CompliantImplicitSolver::SPtr complianceSolver = addNew<CompliantImplicitSolver>(root);
        complianceSolver->debug.setValue(debug);
        complianceSolver->alpha.setValue(1.0);
        complianceSolver->beta.setValue(1.0);
        SReal precision = 1.0e-6;

        linearsolver::LDLTSolver::SPtr linearSolver = addNew<linearsolver::LDLTSolver>(root);
        linearSolver->debug.setValue(debug);
        linearsolver::LDLTResponse::SPtr response = addNew<linearsolver::LDLTResponse>(root);
        (void) response;

        // The string
        ParticleString  string1( root, Vec3(0,0,0), Vec3(1,0,0), 2, 1.0*2 ); // two particles
        string1.compliance->isCompliance.setValue(false); // handle it as a stiffness
        string1.compliance->compliance.setValue(1.0e-3);

        FixedConstraint3::SPtr fixed = addNew<FixedConstraint3>(string1.string_node,"fixedConstraint");
        fixed->addConstraint(0);      // attach first particle

        // velocity parallel to the spring
        {
        MechanicalObject3::WriteVecCoord v = string1.DOF->writeVelocities();
        v[1] = Vec3(1,0,0);
        }


        //**************************************************
        sofa::simulation::getSimulation()->init(root.get());
        //**************************************************

        // initial state
        Vector x0 = getVector( core::VecId::position() );
        Vector v0 = getVector( core::VecId::velocity() );

        //**************************************************
        sofa::simulation::getSimulation()->animate(root.get(),dt);
        //**************************************************

        Vector x1 = getVector( core::VecId::position() );
        Vector v1 = getVector( core::VecId::velocity() );

        // An explicit integration step using the opposite dt should bring us back to the initial state
        string1.compliance->isCompliance.setValue(false); string1.compliance->reinit(); // switch the spring as stiffness
        core::MechanicalParams mparams;
        simulation::common::MechanicalOperations mop (&mparams,root->getContext());
        mop.computeForce( 0+dt, core::VecId::force(), core::VecId::position(), core::VecId::velocity(), false );
        Vector f1 = modeling::getVector( core::VecId::force() );
        Vector v2 = v1 - f1 * dt;
        Vector x2 = x1 - v1 * dt;

        if( debug ){
            std::cerr<<"CompliantImplicitSolver_test::testLinearOneFixedOneStiffnessSpringV100, time step : " << dt << std::endl;
            std::cerr<<"CompliantImplicitSolver_test::testLinearOneFixedOneStiffnessSpringV100, initial positions : " << x0.transpose() << std::endl;
            std::cerr<<"CompliantImplicitSolver_test::testLinearOneFixedOneStiffnessSpringV100, initial velocities: " << v0.transpose() << std::endl;
            std::cerr<<"CompliantImplicitSolver_test::testLinearOneFixedOneStiffnessSpringV100, new positions : " << x1.transpose() << std::endl;
            std::cerr<<"CompliantImplicitSolver_test::testLinearOneFixedOneStiffnessSpringV100, new velocities: " << v1.transpose() << std::endl;
            std::cerr<<"CompliantImplicitSolver_test::testLinearOneFixedOneStiffnessSpringV100, new forces: " << f1.transpose() << std::endl;
            std::cerr<<"CompliantImplicitSolver_test::testLinearOneFixedOneStiffnessSpringV100, new positions  after backward integration: " << x2.transpose() << std::endl;
            std::cerr<<"CompliantImplicitSolver_test::testLinearOneFixedOneStiffnessSpringV100, new velocities after backward integration: " << v2.transpose() << std::endl;
        }

        ASSERT_TRUE( (x2-x0).lpNorm<Eigen::Infinity>() < precision );
        ASSERT_TRUE( (v2-v0).lpNorm<Eigen::Infinity>() < precision );
    }

    /// Same as @testLinearOneFixedOneStiffnessSpringV100, with a compliant spring instead of a stiff spring
    void testLinearOneFixedOneComplianceSpringV100( bool debug )
    {
        SReal dt=0.1; // currently, this must be set in the root node AND passed to the animate function
        Node::SPtr root = clearScene();
        root->setGravity( Vec3(0,0,0) );
        root->setDt(dt);

        // The solver
        using odesolver::CompliantImplicitSolver;
        CompliantImplicitSolver::SPtr complianceSolver = addNew<CompliantImplicitSolver>(root);
        complianceSolver->debug.setValue( debug );
        complianceSolver->alpha.setValue(1.0);
        complianceSolver->beta.setValue(1.0);
        SReal precision = 1.0e-6;

        linearsolver::LDLTSolver::SPtr linearSolver = addNew<linearsolver::LDLTSolver>(root);
        linearSolver->debug.setValue(debug);
        linearsolver::LDLTResponse::SPtr response = addNew<linearsolver::LDLTResponse>(root);
        (void) response;

        // The string
        ParticleString  string1( root, Vec3(0,0,0), Vec3(1,0,0), 2, 1.0*2 ); // two particles
        string1.compliance->isCompliance.setValue(true);
        string1.compliance->compliance.setValue(1.0e-3);

        FixedConstraint3::SPtr fixed = modeling::addNew<FixedConstraint3>(string1.string_node,"fixedConstraint");
        fixed->addConstraint(0);      // attach first particle

        // velocity parallel to the spring
        {
        MechanicalObject3::WriteVecCoord v = string1.DOF->writeVelocities();
        v[1] = Vec3(1,0,0);
        }


        //**************************************************
        sofa::simulation::getSimulation()->init(root.get());
        //**************************************************

        // initial state
        Vector x0 = modeling::getVector( core::VecId::position() );
        Vector v0 = modeling::getVector( core::VecId::velocity() );

        //**************************************************
        sofa::simulation::getSimulation()->animate(root.get(),dt);
        //**************************************************

        Vector x1 = modeling::getVector( core::VecId::position() );
        Vector v1 = modeling::getVector( core::VecId::velocity() );

        // We check the explicit step backward without a solver, because it would not accumulate compliance forces
        string1.compliance->isCompliance.setValue(false); string1.compliance->reinit(); // switch the spring as stiffness
        core::MechanicalParams mparams;
        simulation::common::MechanicalOperations mop (&mparams,root->getContext());
        mop.computeForce( 0+dt, core::VecId::force(), core::VecId::position(), core::VecId::velocity(), false );
        Vector f1 = modeling::getVector( core::VecId::force() );
//        cerr<<"test, f1 = " << f1.transpose() << endl;

        // backward step
        Vector v2 = v1 - f1 * dt;
        Vector x2 = x1 - v1 * dt;

        if( debug ){
        std::cerr<<"CompliantImplicitSolver_test, initial positions : " << x0.transpose() << std::endl;
        std::cerr<<"CompliantImplicitSolver_test, initial velocities: " << v0.transpose() << std::endl;
        std::cerr<<"CompliantImplicitSolver_test, new positions : " << x1.transpose() << std::endl;
        std::cerr<<"CompliantImplicitSolver_test, new velocities: " << v1.transpose() << std::endl;
        std::cerr<<"CompliantImplicitSolver_test, new forces: " << f1.transpose() << std::endl;
        std::cerr<<"CompliantImplicitSolver_test, new positions  after backward integration: " << x2.transpose() << std::endl;
        std::cerr<<"CompliantImplicitSolver_test, new velocities after backward integration: " << v2.transpose() << std::endl;
        }

        ASSERT_TRUE( (x2-x0).lpNorm<Eigen::Infinity>() < precision );
        ASSERT_TRUE( (v2-v0).lpNorm<Eigen::Infinity>() < precision );
    }

    /// One stiffness spring, initially extended
    void testLinearOneFixedOneStiffnessSpringX200( bool debug )
    {
        SReal dt=0.1;   // currently, this must be set in the root node AND passed to the animate function
        Node::SPtr root = clearScene();
        root->setGravity( Vec3(0,0,0) );
        root->setDt(dt);

        // The solver
        typedef odesolver::CompliantImplicitSolver OdeSolver;
        OdeSolver::SPtr odeSolver = addNew<OdeSolver>(root);
        odeSolver->debug.setValue(debug);
        odeSolver->alpha.setValue(1.0);
        odeSolver->beta.setValue(1.0);
        SReal precision = 1.0e-6;

        linearsolver::LDLTSolver::SPtr linearSolver = addNew<linearsolver::LDLTSolver>(root);
        linearSolver->debug.setValue(debug);
        linearsolver::LDLTResponse::SPtr response = addNew<linearsolver::LDLTResponse>(root);
        (void) response;

        // The string
        ParticleString  string1( root, Vec3(0,0,0), Vec3(1,0,0), 2, 1.0*2 ); // two particles
        string1.compliance->isCompliance.setValue(false);
        string1.compliance->compliance.setValue(1.0e-3);

        FixedConstraint3::SPtr fixed = modeling::addNew<FixedConstraint3>(string1.string_node,"fixedConstraint");
        fixed->addConstraint(0);      // attach first particle

        {
        MechanicalObject3::WriteVecCoord x = string1.DOF->writePositions();
        x[1] = Vec3(2,0,0);
        }


        //**************************************************
        sofa::simulation::getSimulation()->init(root.get());
        //**************************************************

        // initial state
        Vector x0 = modeling::getVector( core::VecId::position() );
        Vector v0 = modeling::getVector( core::VecId::velocity() );

        //**************************************************
        sofa::simulation::getSimulation()->animate(root.get(),dt);
        //**************************************************

        Vector x1 = modeling::getVector( core::VecId::position() );
        Vector v1 = modeling::getVector( core::VecId::velocity() );

        // We check the explicit step backward without a solver, because it would not accumulate compliance forces
        core::MechanicalParams mparams;
        simulation::common::MechanicalOperations mop (&mparams,root->getContext());
        mop.computeForce( 0+dt, core::VecId::force(), core::VecId::position(), core::VecId::velocity(), false );
        Vector f1 = modeling::getVector( core::VecId::force() );

        // backward step
        Vector v2 = v1 - f1 * dt;
        Vector x2 = x1 - v1 * dt;

//        cerr<<"CompliantImplicitSolver_test, initial positions : " << x0.transpose() << endl;
//        cerr<<"CompliantImplicitSolver_test, initial velocities: " << v0.transpose() << endl;
//        cerr<<"CompliantImplicitSolver_test, new positions     : " << x1.transpose() << endl;
//        cerr<<"CompliantImplicitSolver_test, new velocities    : " << v1.transpose() << endl;
//        cerr<<"CompliantImplicitSolver_test, new forces        : " << f1.transpose() << endl;
//        cerr<<"CompliantImplicitSolver_test, new positions  after backward integration: " << x2.transpose() << endl;
//        cerr<<"CompliantImplicitSolver_test, new velocities after backward integration: " << v2.transpose() << endl;

        // check that the implicit integration satisfies the implicit integration equation
        ASSERT_TRUE( (x2-x0).lpNorm<Eigen::Infinity>() < precision );
        ASSERT_TRUE( (v2-v0).lpNorm<Eigen::Infinity>() < precision );

        // The particle is initially in (2,0,0) and the closest rest configuration is (1,0,0)
        // The solution should therefore be inbetween.
        ASSERT_TRUE( x1(3)>1.0 ); // the spring should not get inversed


    }

    /// One compliant spring, initially extended
    void testLinearOneFixedOneComplianceSpringX200( bool debug )
    {
        SReal dt=1;
        Node::SPtr root = clearScene();
        root->setGravity( Vec3(0,0,0) );
        root->setDt(dt);

        // The solver
        typedef odesolver::CompliantImplicitSolver OdeSolver;
        OdeSolver::SPtr odeSolver = addNew<OdeSolver>(root);
        odeSolver->debug.setValue(debug);
        odeSolver->alpha.setValue(1.0);
        odeSolver->beta.setValue(1.0);
        SReal precision = 1.0e-6;

        linearsolver::LDLTSolver::SPtr linearSolver = addNew<linearsolver::LDLTSolver>(root);
        linearSolver->debug.setValue(debug);
        linearsolver::LDLTResponse::SPtr response = addNew<linearsolver::LDLTResponse>(root);
        (void) response;

        // The string
        ParticleString  string1( root, Vec3(0,0,0), Vec3(1,0,0), 2, 1.0*2 ); // two particles
        string1.compliance->isCompliance.setValue(true);
        string1.compliance->compliance.setValue(1.0e-3);

        FixedConstraint3::SPtr fixed = modeling::addNew<FixedConstraint3>(string1.string_node,"fixedConstraint");
        fixed->addConstraint(0);      // attach first particle

        {
        MechanicalObject3::WriteVecCoord x = string1.DOF->writePositions();
        x[1] = Vec3(2,0,0);
        }


        //**************************************************
        sofa::simulation::getSimulation()->init(root.get());
        //**************************************************

        // initial state
        Vector x0 = modeling::getVector( core::VecId::position() );
        Vector v0 = modeling::getVector( core::VecId::velocity() );

        //**************************************************
        sofa::simulation::getSimulation()->animate(root.get(),dt);
        //**************************************************

        // state at the end of the step
        Vector x1 = modeling::getVector( core::VecId::position() );
        Vector v1 = modeling::getVector( core::VecId::velocity() );



        // ******* testing lambda propagation *******
        // ie checking force at the end of the time step
        // 1- no lambda propagation -> force must be null
        Vector f1 = modeling::getVector( core::VecId::force() );
        ASSERT_TRUE( f1.sum() == 0 );
        // 2- with constraint force export (after cleaning) -> force must be NOT null
        odeSolver->constraint_forces.beginWriteOnly()->setSelectedItem(3); odeSolver->constraint_forces.endEdit();
        {
        MechanicalObject3::WriteVecCoord x = string1.DOF->writePositions();
        x[1] = Vec3(2,0,0);
        MechanicalObject3::WriteVecCoord v = string1.DOF->writeVelocities();
        v.clear();
        }
        sofa::simulation::getSimulation()->init(root.get());
        sofa::simulation::getSimulation()->animate(root.get(),dt);
        f1 = modeling::getVector( core::VecId::force() );
        ASSERT_TRUE( f1.sum() != 0 );
        // **************


        // We check the explicit step backward without a solver, because it would not accumulate compliance forces
        string1.compliance->isCompliance.setValue(false); string1.compliance->reinit(); // switch the spring as stiffness
        core::MechanicalParams mparams;
        simulation::common::MechanicalOperations mop (&mparams,root->getContext());
        mop.computeForce( 0+dt, core::VecId::force(), core::VecId::position(), core::VecId::velocity(), false );
        f1 = modeling::getVector( core::VecId::force() );

        // backward step
        Vector v2 = v1 - f1 * dt;
        Vector x2 = x1 - v1 * dt;

//        cerr<<"CompliantImplicitSolver_test, initial positions : " << x0.transpose() << endl;
//        cerr<<"CompliantImplicitSolver_test, initial velocities: " << v0.transpose() << endl;
//        cerr<<"CompliantImplicitSolver_test, new positions     : " << x1.transpose() << endl;
//        cerr<<"CompliantImplicitSolver_test, new velocities    : " << v1.transpose() << endl;
//        cerr<<"CompliantImplicitSolver_test, new forces        : " << f1.transpose() << endl;
//        cerr<<"CompliantImplicitSolver_test, new positions  after backward integration: " << x2.transpose() << endl;
//        cerr<<"CompliantImplicitSolver_test, new velocities after backward integration: " << v2.transpose() << endl;

        // check that the implicit integration satisfies the implicit integration equation
        ASSERT_TRUE( (x2-x0).lpNorm<Eigen::Infinity>() < precision );
        ASSERT_TRUE( (v2-v0).lpNorm<Eigen::Infinity>() < precision );

        // The particle is initially in (2,0,0) and the closest rest configuration is (1,0,0)
        // The solution should therefore be inbetween.
        ASSERT_TRUE( x1(3)>1.0 ); // the spring should not get inversed


    }



    /// stiffness & compliance on an empty mstate
    /// only ensuring that Assembly is not crashing
    void testEmptyMState( bool debug )
    {
        Node::SPtr root = clearScene();

        // The solver
        typedef odesolver::CompliantImplicitSolver OdeSolver;
        OdeSolver::SPtr odeSolver = addNew<OdeSolver>(root);
        odeSolver->debug.setValue(debug);
        odeSolver->alpha.setValue(1.0);
        odeSolver->beta.setValue(1.0);

        linearsolver::LDLTSolver::SPtr linearSolver = addNew<linearsolver::LDLTSolver>(root);
        linearSolver->debug.setValue(debug);
        linearsolver::LDLTResponse::SPtr response = addNew<linearsolver::LDLTResponse>(root);
        (void) response;


        MechanicalObject3::SPtr DOF = addNew<MechanicalObject3>(root);
        DOF->resize(1);
        UniformMass3::SPtr mass = addNew<UniformMass3>(root);
        mass->d_totalMass.setValue(1);


        Node::SPtr mappedComplianceNode = root->createChild("mappedComplianceNode");
        MechanicalObject1::SPtr mappedComplianceDOF = addNew<MechanicalObject1>(mappedComplianceNode);
        mappedComplianceDOF->resize(0);

        DistanceFromTargetMapping31::SPtr mappingCompliance = addNew<DistanceFromTargetMapping31>(mappedComplianceNode);
        mappingCompliance->setFrom( DOF.get() );
        mappingCompliance->setTo( mappedComplianceDOF.get() );

        UniformCompliance1::SPtr compliance = addNew<UniformCompliance1>(mappedComplianceNode);
        compliance->isCompliance = true;
        compliance->compliance = 1e-5;


        Node::SPtr mappedStiffnessNode = root->createChild("mappedStiffnessNode");
        MechanicalObject1::SPtr mappedStiffnessDOF = addNew<MechanicalObject1>(mappedStiffnessNode);
        mappedStiffnessDOF->resize(0);

        DistanceFromTargetMapping31::SPtr mappingStiffness = addNew<DistanceFromTargetMapping31>(mappedStiffnessNode);
        mappingStiffness->setFrom( DOF.get() );
        mappingStiffness->setTo( mappedStiffnessDOF.get() );

        UniformCompliance1::SPtr stiffness = addNew<UniformCompliance1>(mappedStiffnessNode);
        stiffness->isCompliance = false;
        stiffness->compliance = 1e-5;


        sofa::simulation::getSimulation()->init(root.get());
        sofa::simulation::getSimulation()->animate(root.get(),1);

    }

};

//=================
// do run the tests
//=================
// simple linear cases
TEST_F(CompliantImplicitSolver_test, OneFixedOneComplianceSpringV100 ){
    EXPECT_MSG_NOEMIT(Error) ;
    testLinearOneFixedOneComplianceSpringV100(false);
}

TEST_F(CompliantImplicitSolver_test, OneFixedOneStiffnessSpringV100  ){
    EXPECT_MSG_NOEMIT(Error) ;
    testLinearOneFixedOneStiffnessSpringV100(false);
}

TEST_F(CompliantImplicitSolver_test, OneFixedOneStiffnessSpringX200  ){
    EXPECT_MSG_NOEMIT(Error) ;
    testLinearOneFixedOneStiffnessSpringX200(false);
}

TEST_F(CompliantImplicitSolver_test, OneFixedOneComplianceSpringX200 ){
    EXPECT_MSG_NOEMIT(Error) ;
    testLinearOneFixedOneComplianceSpringX200(false);
}

TEST_F(CompliantImplicitSolver_test, EmptyMState                     ){
    EXPECT_MSG_NOEMIT(Error) ;
    testEmptyMState(false);
}

}// sofa



