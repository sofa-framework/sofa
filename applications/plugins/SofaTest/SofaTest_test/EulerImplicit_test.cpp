/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/


#include "Solver_test.h"
#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>
#include <sofa/simulation/common/Simulation.h>

namespace sofa {

using namespace modeling;
typedef component::odesolver::EulerImplicitSolver EulerImplicitSolver;
typedef component::linearsolver::CGLinearSolver<component::linearsolver::GraphScatteredMatrix, component::linearsolver::GraphScatteredVector> CGLinearSolver;


/** Test convergence to a static solution.
 * Mass-spring string composed of two particles in gravity, one is fixed.
 * Francois Faure, 2013.
 */
struct EulerImplicit_test_2_particles_to_equilibrium : public Solver_test
{
    EulerImplicit_test_2_particles_to_equilibrium()
    {
        //*******
        initSofa();
        //*******
        // begin create scene under the root node

        EulerImplicitSolver::SPtr eulerSolver = addNew<EulerImplicitSolver> (getRoot() );
        CGLinearSolver::SPtr linearSolver = addNew<CGLinearSolver>   (getRoot() );

        Node::SPtr string = massSpringString(
                    getRoot(), // attached to root node
                    0,1,0,     // first particle position
                    0,0,0,     // last  particle position
                    2,      // number of particles
                    2.0,    // total mass
                    1000.0, // stiffness
                    0.1     // damping ratio
                    );
        FixedConstraint3::SPtr fixed = modeling::addNew<FixedConstraint3>(string,"fixedConstraint");
        fixed->addConstraint(0);      // attach first particle

        Vec3d expected(0,-0.00981,0); // expected position of second particle after relaxation

        // end create scene
        //*********
        initScene();
        //*********
        // run simulation

        Vector x0, x1, v0, v1;
        x0 = getVector( core::VecId::position() ); cerr<<"EulerImplicit_test, new positions : " << x0.transpose() << endl;
        v0 = getVector( core::VecId::velocity() );

        Real dx, dv;
        unsigned n=0;
        const unsigned nMax=100;
        const double  precision = 1.e-4;
        do {
            sofa::simulation::getSimulation()->animate(getRoot().get(),1.0);

            x1 = getVector( core::VecId::position() ); cerr<<"EulerImplicit_test, new positions : " << x1.transpose() << endl;
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
            ADD_FAILURE() << "Solver test has not converged in " << nMax << " iterations, precision = " << precision << endl
                          <<" previous x = " << x0.transpose() << endl
                          <<" current x  = " << x1.transpose() << endl
                          <<" previous v = " << v0.transpose() << endl
                          <<" current v  = " << v1.transpose() << endl;

        // test position of the second particle
        Vec3d actual( x0[3],x0[4],x0[5]); // position of second particle after relaxation
        if( vectorCompare(expected,actual)>precision )
            ADD_FAILURE() << "Solver test has not converged to the expected position" <<
                             " expected: " << expected << endl <<
                             " actual " << actual << endl;

    }

};

TEST_F( EulerImplicit_test_2_particles_to_equilibrium,  ){}

}// namespace sofa







