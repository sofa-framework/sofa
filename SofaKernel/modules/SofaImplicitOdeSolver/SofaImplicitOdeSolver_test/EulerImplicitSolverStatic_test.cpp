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
#include <SofaTest/Sofa_test.h>
#include <SceneCreator/SceneCreator.h>
#include <SceneCreator/SceneUtils.h>
#include <SofaImplicitOdeSolver/EulerImplicitSolver.h>
#include <SofaBaseLinearSolver/CGLinearSolver.h>
#include <SofaBaseMechanics/UniformMass.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaBoundaryCondition/FixedConstraint.h>
#include <SofaDeformable/StiffSpringForceField.h>

#include <sofa/simulation/Simulation.h>

#include <SofaTest/TestMessageHandler.h>


namespace sofa {

using namespace modeling;
using namespace defaulttype;
using core::objectmodel::New;

using sofa::component::mass::UniformMass;
using sofa::component::container::MechanicalObject;
using sofa::component::interactionforcefield::StiffSpringForceField;
using sofa::component::projectiveconstraintset::FixedConstraint;
using sofa::component::odesolver::EulerImplicitSolver;
typedef component::linearsolver::CGLinearSolver<component::linearsolver::GraphScatteredMatrix, component::linearsolver::GraphScatteredVector> CGLinearSolver;


/** Test convergence to a static solution.
 * Mass-spring string composed of two particles in gravity, one is fixed.
 * Francois Faure, 2013.
 */
struct EulerImplicit_test_2_particles_to_equilibrium : public Sofa_test<>
{
    EulerImplicit_test_2_particles_to_equilibrium()
    {
        EXPECT_MSG_NOEMIT(Error) ;
        //*******
        simulation::Node::SPtr root = modeling::initSofa();
        //*******
        // begin create scene under the root node

        EulerImplicitSolver::SPtr eulerSolver = addNew<EulerImplicitSolver>(root);
        CGLinearSolver::SPtr linearSolver = addNew<CGLinearSolver>(root);
        linearSolver->f_maxIter.setValue(25);
        linearSolver->f_tolerance.setValue(1e-5);
        linearSolver->f_smallDenominatorThreshold.setValue(1e-5);

        simulation::Node::SPtr string = massSpringString(
                    root, // attached to root node
                    0,1,0,     // first particle position
                    0,0,0,     // last  particle position
                    2,      // number of particles
                    2.0,    // total mass
                    1000.0, // stiffness
                    0.1     // damping ratio
                    );
        FixedConstraint<Vec3Types>::SPtr fixed = modeling::addNew<FixedConstraint<Vec3Types> >(string,"fixedConstraint");
        fixed->addConstraint(0);      // attach first particle

        Vec3d expected(0,-0.00981,0); // expected position of second particle after relaxation

        // end create scene
        //*********
        initScene(root);
        //*********
        // run simulation

        Vector x0, x1, v0, v1;
        x0 = getVector( core::VecId::position() ); //cerr<<"EulerImplicit_test, initial positions : " << x0.transpose() << endl;
        v0 = getVector( core::VecId::velocity() );

        Real dx, dv;
        unsigned n=0;
        const unsigned nMax=100;
        const double  precision = 1.e-4;
        do {
            sofa::simulation::getSimulation()->animate(root.get(),1.0);

            x1 = getVector( core::VecId::position() ); //cerr<<"EulerImplicit_test, new positions : " << x1.transpose() << endl;
            v1 = getVector( core::VecId::velocity() );

            dx = (x0-x1).lpNorm<Eigen::Infinity>();
            dv = (v0-v1).lpNorm<Eigen::Infinity>();
            x0 = x1;
            v0 = v1;
            n++;

        } while(
                (dx>1.e-4 || dv>1.e-4) // not converged
                && n<nMax );           // give up if not converging

        // end simulation
        // test convergence
        if( n==nMax )
            ADD_FAILURE() << "Solver test has not converged in " << nMax << " iterations, precision = " << precision << std::endl
                          <<" previous x = " << x0.transpose() << std::endl
                          <<" current x  = " << x1.transpose() << std::endl
                          <<" previous v = " << v0.transpose() << std::endl
                          <<" current v  = " << v1.transpose() << std::endl;

        // test position of the second particle
        Vec3d actual( x0[3],x0[4],x0[5]); // position of second particle after relaxation
        if( vectorMaxDiff(expected,actual)>precision )
            ADD_FAILURE() << "Solver test has not converged to the expected position" <<
                             " expected: " << expected << std::endl <<
                             " actual " << actual << std::endl;

    }
};

/**
 * @brief The EulerImplicit_test_2_particles_in_different_nodes_to_equilibrium class is used to test if the solver works well for two particles in different nodes.
 */
struct EulerImplicit_test_2_particles_in_different_nodes_to_equilibrium  : public Sofa_test<>
{

    EulerImplicit_test_2_particles_in_different_nodes_to_equilibrium()
    {
        //*******
        simulation::Node::SPtr root = modeling::initSofa();
        //*******
        // create scene
        root->setGravity(Vec3(0,0,0));

        EulerImplicitSolver::SPtr eulerSolver = addNew<EulerImplicitSolver> (root );
        CGLinearSolver::SPtr linearSolver = addNew<CGLinearSolver> (root );
        linearSolver->f_maxIter.setValue(25);
        linearSolver->f_tolerance.setValue(1e-5);
        linearSolver->f_smallDenominatorThreshold.setValue(1e-5);


        MechanicalObject<Vec3Types>::SPtr DOF = addNew<MechanicalObject<Vec3Types> >(root,"DOF");

        UniformMass<Vec3Types, SReal>::SPtr mass = addNew<UniformMass<Vec3Types, SReal> >(root,"mass");
        mass->d_mass.setValue( 1. );


        // create a child node with its own DOF
        simulation::Node::SPtr child = root->createChild("childNode");
        MechanicalObject<Vec3Types>::SPtr childDof = addNew<MechanicalObject<Vec3Types> >(child);
        UniformMass<Vec3Types, SReal>::SPtr childMass = addNew<UniformMass<Vec3Types, SReal> >(child,"childMass");
        childMass->d_mass.setValue( 1. );

        // attach a spring
        StiffSpringForceField<Vec3Types>::SPtr spring = New<StiffSpringForceField<Vec3Types> >(DOF.get(), childDof.get());
        root->addObject(spring);
        spring->addSpring(0,0,  1000. ,0.1, 1.);

        // set position and velocity vectors, using DataTypes::set to cope with tests in dimension 2
        MechanicalObject<Vec3Types>::VecCoord xp(1),xc(1);
        MechanicalObject<Vec3Types>::DataTypes::set( xp[0], 0., 2.,0.);
        MechanicalObject<Vec3Types>::DataTypes::set( xc[0], 0.,-1.,0.);
        MechanicalObject<Vec3Types>::VecDeriv vp(1),vc(1);
        MechanicalObject<Vec3Types>::DataTypes::set( vp[0], 0.,0.,0.);
        MechanicalObject<Vec3Types>::DataTypes::set( vc[0], 0.,0.,0.);
        // copy the position and velocities to the scene graph
        DOF->resize(1);
        childDof->resize(1);
        MechanicalObject<Vec3Types>::WriteVecCoord xdof = DOF->writePositions(), xchildDof = childDof->writePositions();
        copyToData( xdof, xp );
        copyToData( xchildDof, xc );
        MechanicalObject<Vec3Types>::WriteVecDeriv vdof = DOF->writeVelocities(), vchildDof = childDof->writeVelocities();
        copyToData( vdof, vp );
        copyToData( vchildDof, vc );

        Vec3d expected(0,0,0); // expected position of second particle after relaxation

        // end create scene
        //*********
        initScene(root);
        //*********
        // run simulation

        Vector x0, x1, v0, v1;
        x0 = getVector( core::VecId::position() ); //cerr<<"EulerImplicit_test, initial positions : " << x0.transpose() << endl;
        v0 = getVector( core::VecId::velocity() );

        SReal dx, dv;
        unsigned n=0;
        const unsigned nMax=100;
        const double  precision = 1.e-4;
        do {
            sofa::simulation::getSimulation()->animate(root.get(),1.0);

            x1 = getVector( core::VecId::position() ); //cerr<<"EulerImplicit_test, new positions : " << x1.transpose() << endl;
            v1 = getVector( core::VecId::velocity() );

            dx = (x0-x1).lpNorm<Eigen::Infinity>();
            dv = (v0-v1).lpNorm<Eigen::Infinity>();
            x0 = x1;
            v0 = v1;
            n++;

        } while(
                (dx>1.e-4 || dv>1.e-4) // not converged
                && n<nMax );           // give up if not converging

        // end simulation
        // test convergence
        if( n==nMax )
            ADD_FAILURE() << "Solver test has not converged in " << nMax << " iterations, precision = " << precision << std::endl
                          <<" previous x = " << x0.transpose() << std::endl
                          <<" current x  = " << x1.transpose() << std::endl
                          <<" previous v = " << v0.transpose() << std::endl
                          <<" current v  = " << v1.transpose() << std::endl;

        // test position of the second particle
        Vec3d actual( x0[3],x0[4],x0[5]); // position of second particle after relaxation
        if( vectorMaxDiff(expected,actual)>precision )
            ADD_FAILURE() << "Solver test has not converged to the expected position" <<
                             " expected: " << expected << std::endl <<
                             " actual " << actual << std::endl;

    }

};

TEST_F( EulerImplicit_test_2_particles_to_equilibrium,  ){}
TEST_F( EulerImplicit_test_2_particles_in_different_nodes_to_equilibrium,  ){}

}// namespace sofa







