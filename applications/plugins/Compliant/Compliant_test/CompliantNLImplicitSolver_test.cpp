#include <plugins/Compliant/numericalsolver/MinresSolver.h>
#include <plugins/Compliant/numericalsolver/LDLTSolver.h>
#include <plugins/Compliant/Compliant_test/Compliant_test.h>
#include <plugins/Compliant/odesolver/CompliantNLImplicitSolver.h>
#include <SofaExplicitOdeSolver/EulerSolver.h>
#include <plugins/SceneCreator/SceneCreator.h>

using namespace sofa::modeling;
using namespace sofa::component;
using namespace sofa;


struct CompliantNLImplicitSolver_test : public sofa::CompliantSolver_test
{

    /** @defgroup CompliantNLImplicitSolver_Unit_Tests CompliantNLImplicitSolver basic tests.
     *
     * In the most basic tests, the scene is composed of two particles connected by a spring. One particle is fixed, while the other has an initial velocity.
     * The solver is set to backward Euler: alpha = beta = 1.
     */
    ///@{

    /** Test in the linear case, with velocity parallel to the spring.
      Convergence should occur in one iteration.
      Integrate using backward Euler (alpha=1, beta=1).
      Post-condition: an explicit Euler step with step -dt brings the system back to the original state.
      */
    void testLinearOneFixedOneSpringV100( bool compliant, bool debug=false )
    {
        SReal dt=0.1;
        Node::SPtr root = clearScene();
        root->setDt(dt);
        root->setGravity( Vec3(0,0,0) );

        // The solver
        using odesolver::CompliantNLImplicitSolver;
        CompliantNLImplicitSolver::SPtr complianceSolver = addNew<CompliantNLImplicitSolver>(getRoot());
        complianceSolver->iterations.setValue(1);
        complianceSolver->debug.setValue(debug);
        complianceSolver->alpha.setValue(1.0);
        complianceSolver->beta.setValue(1.0);
        SReal precision = 1.0e-6;
        complianceSolver->precision.setValue(precision);
        complianceSolver->stabilization.beginEdit()->setSelectedItem(CompliantNLImplicitSolver::NO_STABILIZATION); complianceSolver->stabilization.endEdit();

        linearsolver::LDLTSolver::SPtr linearSolver = addNew<linearsolver::LDLTSolver>(getRoot());
        linearSolver->debug.setValue(debug);

        // The string
        ParticleString  string1( root, Vec3(0,0,0), Vec3(1,0,0), 2, 1.0*2 ); // two particles
        string1.compliance->isCompliance.setValue(compliant);
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

         // An integration step using the opposite dt should bring us back to the initial state
        core::MechanicalParams mparams;
        simulation::common::MechanicalOperations mop (&mparams,getRoot()->getContext());
        mop.computeForce( 0+dt, core::VecId::force(), core::VecId::position(), core::VecId::velocity(), false );
        Vector f1 = modeling::getVector( core::VecId::force() );

        // backward step
        Vector v2 = v1 - f1 * dt;
        Vector x2 = x1 - v1 * dt;

//                cerr<<"CompliantNLImplicitSolver_test, initial positions : " << x0.transpose() << endl;
//                cerr<<"CompliantNLImplicitSolver_test, initial velocities: " << v0.transpose() << endl;
//                cerr<<"CompliantNLImplicitSolver_test, new positions :     " << x1.transpose() << endl;
//                cerr<<"CompliantNLImplicitSolver_test, new velocities:     " << v1.transpose() << endl;
//                cerr<<"CompliantNLImplicitSolver_test, new forces:         " << f1.transpose() << endl;
//                cerr<<"CompliantNLImplicitSolver_test, new positions  after backward integration: " << x2.transpose() << endl;
//                cerr<<"CompliantNLImplicitSolver_test, new velocities after backward integration: " << v2.transpose() << endl;

        SReal maxPosError = (x2-x0).lpNorm<Eigen::Infinity>();
        SReal maxVelError = (v2-v0).lpNorm<Eigen::Infinity>();
        ASSERT_TRUE( maxPosError < precision );
        ASSERT_TRUE( maxVelError < precision );
    }


    /// One compliant spring, initially extended
    void testLinearOneFixedOneSpringX200( bool compliant, bool debug=false )
    {
        SReal dt=0.1;
        Node::SPtr root = clearScene();
        root->setGravity( Vec3(0,0,0) );
        root->setDt(dt);

        // The solver
        typedef odesolver::CompliantNLImplicitSolver OdeSolver;
        OdeSolver::SPtr odeSolver = addNew<OdeSolver>(root);
        odeSolver->iterations.setValue(1);
        odeSolver->debug.setValue(debug);
        odeSolver->alpha.setValue(1.0);
        odeSolver->beta.setValue(1.0);
        SReal precision = 1.0e-6;
        odeSolver->precision.setValue(precision);
        odeSolver->stabilization.beginEdit()->setSelectedItem(OdeSolver::NO_STABILIZATION); odeSolver->stabilization.endEdit();

        linearsolver::LDLTSolver::SPtr linearSolver = addNew<linearsolver::LDLTSolver>(root);
        linearSolver->debug.setValue(debug);

        // The string
        ParticleString  string1( root, Vec3(0,0,0), Vec3(1,0,0), 2, 1.0*2 ); // two particles
        string1.compliance->isCompliance.setValue(compliant);
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
        simulation::common::MechanicalOperations mop (&mparams,getRoot()->getContext());
        mop.computeForce( 0+dt, core::VecId::force(), core::VecId::position(), core::VecId::velocity(), false );
        Vector f1 = modeling::getVector( core::VecId::force() );

        // backward step
        Vector v2 = v1 - f1 * dt;
        Vector x2 = x1 - v1 * dt;

        if( debug ){
            cerr<<"CompliantNLImplicitSolver_test, initial positions : " << x0.transpose() << endl;
            cerr<<"CompliantNLImplicitSolver_test, initial velocities: " << v0.transpose() << endl;
            cerr<<"CompliantNLImplicitSolver_test, new positions     : " << x1.transpose() << endl;
            cerr<<"CompliantNLImplicitSolver_test, new velocities    : " << v1.transpose() << endl;
            cerr<<"CompliantNLImplicitSolver_test, new forces        : " << f1.transpose() << endl;
            cerr<<"CompliantNLImplicitSolver_test, new positions  after backward integration: " << x2.transpose() << endl;
            cerr<<"CompliantNLImplicitSolver_test, new velocities after backward integration: " << v2.transpose() << endl;
        }

        ASSERT_TRUE( (x2-x0).lpNorm<Eigen::Infinity>() < precision );
        ASSERT_TRUE( x1(3)>1.0 ); // the spring should not get inversed
        ASSERT_TRUE( (v2-v0).lpNorm<Eigen::Infinity>() < precision );
    }


    /// One  spring, orthogonal to gravity. Converges in 3 iterations for g=100, dt=4.1
    void testLinearOneFixedOneSpringGravity( bool isCompliance, bool debug=false, unsigned iterations=3, SReal precision=1e-6 )
    {
        SReal dt=4.1;
        Node::SPtr root = clearScene();
        root->setGravity( Vec3(0,100,0) );
        root->setDt(dt);

        // The solver
        using odesolver::CompliantNLImplicitSolver;
        CompliantNLImplicitSolver::SPtr complianceSolver = addNew<CompliantNLImplicitSolver>(root);
        complianceSolver->iterations.setValue(iterations);
        complianceSolver->debug.setValue(debug);
        complianceSolver->alpha.setValue(1.0);
        complianceSolver->beta.setValue(1.0);

        complianceSolver->precision.setValue(precision);
        complianceSolver->neglecting_compliance_forces_in_geometric_stiffness.setValue(false);
        complianceSolver->newtonStepLength.setValue(isCompliance?0.1:1);
        complianceSolver->stabilization.beginEdit()->setSelectedItem(CompliantNLImplicitSolver::NO_STABILIZATION); complianceSolver->stabilization.endEdit();

        linearsolver::LDLTSolver::SPtr linearSolver = addNew<linearsolver::LDLTSolver>(root);
//        linearSolver->debug.setValue(debug);

        // The string
        ParticleString  string1( root, Vec3(0,0,0), Vec3(1,0,0), 2, 1.0*2 ); // two particles
        string1.compliance->isCompliance.setValue(isCompliance);
        string1.compliance->compliance.setValue(1.0e-3);

        FixedConstraint3::SPtr fixed = modeling::addNew<FixedConstraint3>(string1.string_node,"fixedConstraint");
        fixed->addConstraint(0);      // attach first particle

        {
            MechanicalObject3::WriteVecCoord x = string1.DOF->writePositions();
            x[1] = Vec3(1,0,0);
        }


//        sofa::simulation::getSimulation()->exportXML( sofa::simulation::getSimulation()->GetRoot().get(), "/tmp/test.scn" );


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
        simulation::common::MechanicalOperations mop (&mparams,getRoot()->getContext());
        mop.computeForce( 0+dt, core::VecId::force(), core::VecId::position(), core::VecId::velocity(), false );
        Vector f1 = modeling::getVector( core::VecId::force() );

        // backward step
        Vector v2 = v1 - f1 * dt;
        Vector x2 = x1 - v1 * dt;

        if( debug ){
            cerr<<"CompliantNLImplicitSolver_test, initial positions : " << x0.transpose() << endl;
            cerr<<"CompliantNLImplicitSolver_test, initial velocities: " << v0.transpose() << endl;
            cerr<<"CompliantNLImplicitSolver_test, new positions : " << x1.transpose() << endl;
            cerr<<"CompliantNLImplicitSolver_test, new velocities: " << v1.transpose() << endl;
            cerr<<"CompliantNLImplicitSolver_test, new forces: " << f1.transpose() << endl;
            cerr<<"CompliantNLImplicitSolver_test, new positions  after backward integration: " << x2.transpose() << endl;
            cerr<<"CompliantNLImplicitSolver_test, new velocities after backward integration: " << v2.transpose() << endl;
        }

        ASSERT_TRUE( (x2-x0).lpNorm<Eigen::Infinity>() < precision );
        ASSERT_TRUE( (v2-v0).lpNorm<Eigen::Infinity>() < precision );
    }


    /// Two springs, 3 particles including 2 fixed, forming a triangle.
    ///  start configuration:
    ///   X
    ///     \.
    ///      \.
    ///   X---o
    /// solution:
    ///   X
    ///     \.
    ///      o
    ///     /
    ///   X
    /// no gravity
    void testNonlinear( bool compliance, bool debug=false )
    {
        SReal dt=1.0;
        Node::SPtr root = clearScene();
        root->setDt(dt);
        root->setGravity( Vec3(0,0,0) );

        // The solver
        odesolver::CompliantNLImplicitSolver::SPtr complianceSolver = addNew<odesolver::CompliantNLImplicitSolver>(root);
        complianceSolver->alpha.setValue(1.0);
        complianceSolver->beta.setValue(1.0);
        complianceSolver->iterations.setValue(10);
        complianceSolver->debug.setValue( debug );
        SReal precision = 1.0e-6;
        complianceSolver->precision.setValue(precision);
        complianceSolver->stabilization.beginEdit()->setSelectedItem(odesolver::CompliantNLImplicitSolver::NO_STABILIZATION); complianceSolver->stabilization.endEdit();

        linearsolver::LDLTSolver::SPtr linearSolver = addNew<linearsolver::LDLTSolver>(root);
//        linearSolver->debug.setValue( debug );

        // The string
        int nump = 3;
        ParticleString  string1( root, Vec3(0,0,0), Vec3(nump-1,0,0), nump, 1.0*nump );
        string1.compliance->isCompliance.setValue(compliance); // handle it in the constraints or as stiffness
        string1.compliance->compliance.setValue(1.0e-3);

        FixedConstraint3::SPtr fixed = modeling::addNew<FixedConstraint3>(string1.string_node,"fixedConstraint");
        fixed->addConstraint(0);      // attach first particle
        fixed->addConstraint(2);      // attach last particle

        // setup positions and velocities
        {
            MechanicalObject3::WriteVecCoord x= string1.DOF->writePositions();
            x[1] = Vec3(1,0,0);
            x[2] = Vec3(0,1,0);
            MechanicalObject3::WriteVecCoord v = string1.DOF->writeVelocities();
            v[1] = Vec3(0,0,0);
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

        Vector x1 = modeling::getVector( core::VecId::position() );
        Vector v1 = modeling::getVector( core::VecId::velocity() );

        // We check the explicit step backward without a solver, because it would not accumulate compliance forces
        core::MechanicalParams mparams;
        simulation::common::MechanicalOperations mop (&mparams,getRoot()->getContext());
        mop.computeForce( 0+dt, core::VecId::force(), core::VecId::position(), core::VecId::velocity(), false );
        Vector f1 = modeling::getVector( core::VecId::force() );

        // backward step
        Vector v2 = v1 - f1 * dt;
        Vector x2 = x1 - v1 * dt;

        if( debug ){
            cerr<<"CompliantNLImplicitSolver_test, initial positions : " << x0.transpose() << endl;
            cerr<<"CompliantNLImplicitSolver_test, initial velocities: " << v0.transpose() << endl;
            cerr<<"CompliantNLImplicitSolver_test, new positions : " << x1.transpose() << endl;
            cerr<<"CompliantNLImplicitSolver_test, new velocities: " << v1.transpose() << endl;
            cerr<<"CompliantNLImplicitSolver_test, new forces: " << f1.transpose() << endl;
            cerr<<"CompliantNLImplicitSolver_test, new positions  after backward integration: " << x2.transpose() << endl;
            cerr<<"CompliantNLImplicitSolver_test, new velocities after backward integration: " << v2.transpose() << endl;
        }

        ASSERT_TRUE( (x2-x0).lpNorm<Eigen::Infinity>() < precision );
        ASSERT_TRUE( (v2-v0).lpNorm<Eigen::Infinity>() < precision*10 );

    }




};

//=================
// do run the tests
//=================
// simple linear cases
TEST_F(CompliantNLImplicitSolver_test, OneFixedOneStiffnessSpringV100 ){     testLinearOneFixedOneSpringV100(false);  }
TEST_F(CompliantNLImplicitSolver_test, OneFixedOneComplianceSpringV100 ){    testLinearOneFixedOneSpringV100(true);  }
TEST_F(CompliantNLImplicitSolver_test, OneFixedOneStiffnessSpringX200 ){     testLinearOneFixedOneSpringX200(false);  }
TEST_F(CompliantNLImplicitSolver_test, OneFixedOneComplianceSpringX200 ){    testLinearOneFixedOneSpringX200(true);  }

//// simple nonlinear cases
TEST_F(CompliantNLImplicitSolver_test, OneFixedOneSpringGravityStiffness ) { testLinearOneFixedOneSpringGravity(false);  }
TEST_F(CompliantNLImplicitSolver_test, OneFixedOneSpringGravityCompliance ){ testLinearOneFixedOneSpringGravity(true);  }
TEST_F(CompliantNLImplicitSolver_test, NonlinearStiffness ){                 testNonlinear(false,true); }
TEST_F(CompliantNLImplicitSolver_test, NonlinearCompliance ){                testNonlinear(true,true); }




