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

#include <sofa/simulation/graph/SimpleApi.h>

#include <sofa/simulation/Simulation.h>
#include <sofa/component/odesolver/testing/EigenTestUtilities.h>


namespace sofa {

using namespace type;
using namespace testing;
using namespace defaulttype;
using core::objectmodel::New;


/// Create a stiff string
Node::SPtr massSpringString(Node::SPtr parent,
    double x0, double y0, double z0, // start point,
    double x1, double y1, double z1, // end point
    unsigned numParticles,
    double totalMass,
    double stiffnessValue,
    double dampingRatio)
{
    static unsigned numObject = 1;
    std::ostringstream oss;
    oss << "string_" << numObject++;

    Vec3d startPoint(x0, y0, z0), endPoint(x1, y1, z1);
    SReal totalLength = (endPoint - startPoint).norm();

    std::stringstream positions;
    std::stringstream springs;
    for (unsigned i = 0; i < numParticles; i++)
    {
        double alpha = (double)i / (numParticles - 1);
        Vec3d currpos = startPoint * (1 - alpha) + endPoint * alpha;
        positions << simpleapi::str(currpos) << " ";

        if (i > 0)
        {
            springs << simpleapi::str(i - 1) << " " << simpleapi::str(i) << " " << simpleapi::str(stiffnessValue)
                << " " << simpleapi::str(dampingRatio) << " " << simpleapi::str(totalLength / (numParticles - 1));
        }
    }

    Node::SPtr node = simpleapi::createChild(parent, oss.str());
    simpleapi::createObject(node, "MechanicalObject", {
                                {"name", oss.str() + "_DOF"},
                                {"size", simpleapi::str(numParticles)},
                                {"position", positions.str()}
        });

    simpleapi::createObject(node, "UniformMass", {
                                {"name",oss.str() + "_mass"},
                                {"vertexMass", simpleapi::str(totalMass / numParticles)} });

    simpleapi::createObject(node, "StiffSpringForceField", {
                                {"name", oss.str() + "_spring"},
                                {"spring", springs.str()}
        });

    return node;
}

/** Test convergence to a static solution.
 * Mass-spring string composed of two particles in gravity, one is fixed.
 * Francois Faure, 2013.
 */
struct EulerImplicit_test_2_particles_to_equilibrium : public BaseSimulationTest, NumericTest<SReal>
{
    EulerImplicit_test_2_particles_to_equilibrium()
    {
        EXPECT_MSG_NOEMIT(Error) ;
        //*******
        const auto simu = simpleapi::createSimulation();
        const simulation::Node::SPtr root = simpleapi::createRootNode(simu, "root");
        //*******
        // begin create scene under the root node
        sofa::simpleapi::importPlugin("Sofa.Component.ODESolver");
        sofa::simpleapi::importPlugin("Sofa.Component.LinearSolver.Iterative");
        sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");
        sofa::simpleapi::importPlugin("Sofa.Component.Mass");
        sofa::simpleapi::importPlugin("Sofa.Component.Constraint.Projective");
        sofa::simpleapi::importPlugin("Sofa.Component.SolidMechanics.Spring");

        // remove warnings
        simpleapi::createObject(root, "DefaultAnimationLoop", {});
        simpleapi::createObject(root, "DefaultVisualManagerLoop", {});

        simpleapi::createObject(root, "EulerImplicitSolver", {});
        simpleapi::createObject(root, "CGLinearSolver", {
            { "iterations", simpleapi::str(25)},
            { "tolerance", simpleapi::str(1e-5)},
            { "threshold", simpleapi::str(1e-5)},
        });

        const simulation::Node::SPtr string = massSpringString (
                    root, // attached to root node
                    0,1,0,     // first particle position
                    0,0,0,     // last  particle position
                    2,      // number of particles
                    2.0,    // total mass
                    1000.0, // stiffness
                    0.1     // damping ratio
                    );

        simpleapi::createObject(string, "FixedProjectiveConstraint", {
            { "indices", "0"}
        });

        Vec3d expected(0,-0.00981,0); // expected position of second particle after relaxation

        // end create scene
        //*********
        sofa::simulation::node::initRoot(root.get());
        //*********
        // run simulation

        Eigen::VectorXd x0, x1, v0, v1;
        x0 = component::odesolver::testing::getVector( root, core::VecId::position() ); //cerr<<"EulerImplicit_test, initial positions : " << x0.transpose() << endl;
        v0 = component::odesolver::testing::getVector( root, core::VecId::velocity() );

        Eigen::VectorXd::RealScalar dx, dv;
        unsigned n=0;
        const unsigned nMax=100;
        const double  precision = 1.e-4;
        do {
            sofa::simulation::node::animate(root.get(), 1_sreal);

            x1 = component::odesolver::testing::getVector( root, core::VecId::position() ); //cerr<<"EulerImplicit_test, new positions : " << x1.transpose() << endl;
            v1 = component::odesolver::testing::getVector( root, core::VecId::velocity() );

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
struct EulerImplicit_test_2_particles_in_different_nodes_to_equilibrium  : public BaseSimulationTest, NumericTest<SReal>
{

    EulerImplicit_test_2_particles_in_different_nodes_to_equilibrium()
    {
        //*******
        const auto simu = simpleapi::createSimulation();
        const simulation::Node::SPtr root = simpleapi::createRootNode(simu, "root");
        //*******
        // create scene
        root->setGravity(Vec3(0,0,0));

        sofa::simpleapi::importPlugin("Sofa.Component.ODESolver");
        sofa::simpleapi::importPlugin("Sofa.Component.LinearSolver.Iterative");
        sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");
        sofa::simpleapi::importPlugin("Sofa.Component.Mass");
        sofa::simpleapi::importPlugin("Sofa.Component.Constraint.Projective");
        sofa::simpleapi::importPlugin("Sofa.Component.SolidMechanics.Spring");
        // remove warnings
        simpleapi::createObject(root, "DefaultAnimationLoop", {});
        simpleapi::createObject(root, "DefaultVisualManagerLoop", {});

        simpleapi::createObject(root, "EulerImplicitSolver", {});
        simpleapi::createObject(root, "CGLinearSolver", {
            { "iterations", simpleapi::str(25)},
            { "tolerance", simpleapi::str(1e-5)},
            { "threshold", simpleapi::str(1e-5)},
            });

        simpleapi::createObject(root, "MechanicalObject", {
            {"name", "DOF"},
            {"position", simpleapi::str("0.0 2.0 0.0")},
            {"velocity", simpleapi::str("0.0 0.0 0.0")}
        });

        simpleapi::createObject(root, "UniformMass", {
            { "name","mass"},
            { "vertexMass", "1.0"}
        });

        // create a child node with its own DOF
        const simulation::Node::SPtr child = root->createChild("childNode");
        simpleapi::createObject(child, "MechanicalObject", {
            {"name", "childDof"},
            {"position", simpleapi::str("0.0 -1.0 0.0")},
            {"velocity", simpleapi::str("0.0 0.0 0.0")}
        });

        simpleapi::createObject(child, "UniformMass", {
            { "name","childMass"},
            { "vertexMass", simpleapi::str("1.0")}
        });

        // attach a spring
        std::ostringstream oss;
        oss << 0 << " " << 0 << " " << 1000.0 << " " << 0.1 << " " << 1.0f;
        simpleapi::createObject(root, "StiffSpringForceField", {
            {"spring", oss.str()},
            { "object1", "@/DOF"},
            { "object2", "@childNode/childDof"},
        });

        Vec3d expected(0,0,0); // expected position of second particle after relaxation

        // end create scene
        //*********
        sofa::simulation::node::initRoot(root.get());
        //*********
        // run simulation

        Eigen::VectorXd x0, x1, v0, v1;
        x0 = component::odesolver::testing::getVector(root, core::VecId::position() ); //cerr<<"EulerImplicit_test, initial positions : " << x0.transpose() << endl;
        v0 = component::odesolver::testing::getVector(root, core::VecId::velocity() );

        SReal dx, dv;
        unsigned n=0;
        const unsigned nMax=100;
        const double  precision = 1.e-4;
        do {
            sofa::simulation::node::animate(root.get(), 1_sreal);

            x1 = component::odesolver::testing::getVector(root, core::VecId::position() ); //cerr<<"EulerImplicit_test, new positions : " << x1.transpose() << endl;
            v1 = component::odesolver::testing::getVector(root, core::VecId::velocity() );

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

TEST_F( EulerImplicit_test_2_particles_to_equilibrium, check ){}
TEST_F( EulerImplicit_test_2_particles_in_different_nodes_to_equilibrium, check ){}

}// namespace sofa







