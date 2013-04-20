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

namespace sofa {

struct EulerImplicit_test_2particles : public Simulation_test
{
    EulerImplicit_test_2particles()
    {
        //*******
        initSofa();
        //*******
        // begin create scene under the root node

        modeling::addNew<odesolver::EulerImplicitSolver>(getRoot(),"odesolver" );
        modeling::addNew<CGLinearSolver>(getRoot(),"linearsolver");

        Node::SPtr string = massSpringString(
                    getRoot(),
                    0,0,0, // first endpoint
                    1,0,0, // second endpoint
                    2,     // number of particles
                    2.0,    // total mass
                    1000.0,   // stiffness
                    0.1    // damping ratio
                    );
        FixedConstraint3::SPtr fixed = modeling::addNew<FixedConstraint3>(string,"fixedConstraint");
        fixed->addConstraint(0);

        // end create scene
        //*********
        initScene();
        //*********

        FullVector x0, x1, v0, v1;
        getAssembledPositionVector(&x0);
//        cerr<<"My_test, initial positions : " << x0 << endl;
        getAssembledVelocityVector(&v0);
//        cerr<<"My_test, initial velocities: " << v0 << endl;

        Real dx, dv;
        unsigned n=0;
        const unsigned nMax=100;
        const double  precision = 1.e-4;
        do {
            sofa::simulation::getSimulation()->animate(getRoot().get(),1.0);

            getAssembledPositionVector(&x1);
//            cerr<<"My_test, new positions : " << x1 << endl;
            getAssembledVelocityVector(&v1);
//            cerr<<"My_test, new velocities: " << v1 << endl;

            dx = this->vectorCompare(x0,x1);
            dv = this->vectorCompare(v0,v1);
            x0 = x1;
            v0 = v1;
            n++;

        } while( (dx>1.e-4 || dv>1.e-4) && n<nMax );

        // test convergence
        if( n==nMax )
            ADD_FAILURE() << "Solver test has not converged in " << nMax << " iterations, precision = " << precision << endl
                          <<" previous x = " << x0 << endl
                          <<" current x  = " << x1 << endl
                          <<" previous v = " << v0 << endl
                          <<" current v  = " << v1 << endl;

        // test position of the second particle
        Vec3d expected(0,-1.00981,0), actual( x0[3],x0[4],x0[5]);
        if( vectorCompare(expected,actual)>precision )
            ADD_FAILURE() << "Solver test has not converged to the expected position" <<
                             " expected: " << expected << endl <<
                             " actual " << actual << endl;

    }

};

TEST_F( EulerImplicit_test_2particles, myEulerImplicit_test_2particles ){}

}// namespace sofa







