/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <SofaDeformable/StiffSpringForceField.h>
#include <SofaTest/ForceField_test.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <SofaImplicitOdeSolver/EulerImplicitSolver.h>
#include <SofaBaseLinearSolver/CGLinearSolver.h>

namespace sofa {

using namespace modeling;
typedef component::odesolver::EulerImplicitSolver EulerImplicitSolver;
typedef component::linearsolver::CGLinearSolver<component::linearsolver::GraphScatteredMatrix, component::linearsolver::GraphScatteredVector> CGLinearSolver;

/** Used to rotate points and vectors in space for tests
 *
 */
template <class DataTypes>
struct RigidTransform
{
    typedef typename DataTypes::Real Real;
    typedef typename defaulttype::StdRigidTypes<DataTypes::spatial_dimensions,Real>::Coord Transform;
    typedef typename DataTypes::Coord Point;
    typedef typename DataTypes::Deriv Vec;
    Transform transform;

    void randomize( Real amplitude )
    {
        transform = Transform::rand(amplitude);
    }

    void projectPoint( Point& p )
    {
        p = transform.projectPoint(p);
    }

    void projectVector( Vec& v )
    {
        v = transform.projectVector(v);
    }

};


/**  Test suite for StiffSpringForceField.
  *  The test cases are defined in the #Test_Cases member group.
  *
  * @author Francois Faure, 2014
  */
template <typename _StiffSpringForceField>
struct StiffSpringForceField_test : public ForceField_test<_StiffSpringForceField>
{

    typedef _StiffSpringForceField ForceType;
    typedef ForceField_test<_StiffSpringForceField> Inherit;
    typedef typename ForceType::DataTypes DataTypes;

    typedef typename ForceType::VecCoord VecCoord;
    typedef typename ForceType::VecDeriv VecDeriv;
    typedef typename ForceType::Coord Coord;
    typedef typename ForceType::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef defaulttype::Vec<3,Real> Vec3;

    typedef ForceType Spring;
    typedef component::container::MechanicalObject<DataTypes> DOF;


    /** @name Test_Cases
      For each of these cases, we check if the accurate forces are computed
      */
    ///@{

    /** Two particles
    */
    void test_2particles( Real stiffness, Real dampingRatio, Real restLength,
                          Vec3 x0, Vec3 v0,
                          Vec3 x1, Vec3 v1,
                          Vec3 f0)
    {
        // set position and velocity vectors, using DataTypes::set to cope with tests in dimension 2
        VecCoord x(2);
        DataTypes::set( x[0], x0[0],x0[1],x0[2]);
        DataTypes::set( x[1], x1[0],x1[1],x1[2]);
        VecDeriv v(2);
        DataTypes::set( v[0], v0[0],v0[1],v0[2]);
        DataTypes::set( v[1], v1[0],v1[1],v1[2]);
        VecDeriv f(2);
        DataTypes::set( f[0],  f0[0], f0[1], f0[2]);
        DataTypes::set( f[1], -f0[0],-f0[1],-f0[2]);

        // randomly rotate and translate the scene
        /// @todo functions to apply the transform to VecCoord and VecDeriv
        RigidTransform<DataTypes> transform;
        transform.randomize(1.1);
        transform.projectPoint( x[0]);
        transform.projectVector(v[0]);
        transform.projectPoint( x[1]);
        transform.projectVector(v[1]);
        transform.projectVector(f[0]);
        transform.projectVector(f[1]);


        // tune the force field
        this->force->addSpring(0,1,stiffness,dampingRatio,restLength);

        //        cout<<"test_2particles, x = " << x << endl;
        //        cout<<"                 v = " << v << endl;
        //        cout<<"                 f = " << f << endl;

        // and run the test
        this->run_test( x, v, f );
    }

    /** Two particles in different nodes. One node is the child of the other. The parent contains a solver.
    */
    void test_2particles_in_parent_and_child(
            Real stiffness, Real dampingRatio, Real restLength,
            Vec3 x0, Vec3 v0,
            Vec3 x1, Vec3 v1,
            Vec3 f0)
    {
        // create a child node with its own DOF
        simulation::Node::SPtr child = this->node->createChild("childNode");
        typename DOF::SPtr childDof = addNew<DOF>(child);

        // replace the spring with another one, between the parent and the child
        this->node->removeObject(this->force);
        typename Spring::SPtr spring = sofa::core::objectmodel::New<Spring>(this->dof.get(), childDof.get());
        this->node->addObject(spring);

        // set position and velocity vectors, using DataTypes::set to cope with tests in dimension 2
        VecCoord xp(1),xc(1);
        DataTypes::set( xp[0], x0[0],x0[1],x0[2]);
        DataTypes::set( xc[0], x1[0],x1[1],x1[2]);
        VecDeriv vp(1),vc(1);
        DataTypes::set( vp[0], v0[0],v0[1],v0[2]);
        DataTypes::set( vc[0], v1[0],v1[1],v1[2]);
        VecDeriv fp(1),fc(1);
        DataTypes::set( fp[0],  f0[0], f0[1], f0[2]);
        DataTypes::set( fc[0], -f0[0],-f0[1],-f0[2]);
        // copy the position and velocities to the scene graph
        this->dof->resize(1);
        childDof->resize(1);
        typename DOF::WriteVecCoord xdof = this->dof->writePositions(), xchildDof = childDof->writePositions();
        copyToData( xdof, xp );
        copyToData( xchildDof, xc );
        typename DOF::WriteVecDeriv vdof = this->dof->writeVelocities(), vchildDof = childDof->writeVelocities();
        copyToData( vdof, vp );
        copyToData( vchildDof, vc );

        // tune the force field
        spring->addSpring(0,0,stiffness,dampingRatio,restLength);


        // and run the test

        // init scene and compute force
        sofa::simulation::getSimulation()->init(this->node.get());
        core::MechanicalParams mparams;
        mparams.setKFactor(1.0);
        simulation::MechanicalComputeForceVisitor computeForce( &mparams, core::VecDerivId::force() );
        this->node->execute(computeForce);

        // check force
        typename DOF::ReadVecDeriv actualfp = this->dof->readForces();
        typename DOF::ReadVecDeriv actualfc = childDof->readForces();
        if(this->debug){
            std::cout << "run_test,          xp = " << xp << std::endl;
            std::cout << "                   xc = " << xc << std::endl;
            std::cout << "                   vp = " << vp << std::endl;
            std::cout << "                   vc = " << vc << std::endl;
            std::cout << "          expected fp = " << fp << std::endl;
            std::cout << "            actual fp = " << actualfp << std::endl;
            std::cout << "          expected fc = " << fc << std::endl;
            std::cout << "            actual fc = " << actualfc << std::endl;
        }
        ASSERT_TRUE( this->vectorMaxDiff(fp,actualfp)< this->errorMax*this->epsilon() );
        ASSERT_TRUE( this->vectorMaxDiff(fc,actualfc)< this->errorMax*this->epsilon() );
    }

    ///@}
};



// ========= Define the list of types to instanciate.
//using testing::Types;
typedef testing::Types<
component::interactionforcefield::StiffSpringForceField<defaulttype::Vec2Types>,  // 2D
component::interactionforcefield::StiffSpringForceField<defaulttype::Vec3Types>   // 3D
> TestTypes; // the types to instanciate.



// ========= Tests to run for each instanciated type
TYPED_TEST_CASE(StiffSpringForceField_test, TestTypes);

// first test case: extension, no velocity
TYPED_TEST( StiffSpringForceField_test , extension )
{
    this->debug = false;

    SReal
            k = 1.0,  // stiffness
            d = 0.1,  // damping ratio
            l0 = 1.0; // rest length

    typename TestFixture::Vec3
            x0(0,0,0), // position of the first particle
            v0(0,0,0), // velocity of the first particle
            x1(2,0,0), // position of the second particle
            v1(0,0,0), // velocity of the second particle
            f0(1,0,0); // expected force on the first particle


    // use the parent  class to automatically test the functions
    this->test_2particles(k,d,l0, x0,v0, x1,v1, f0);
}


// velocity, no extension
TYPED_TEST( StiffSpringForceField_test , viscosity )
{
    // initial velocities with viscosity create dissipative forces
    // that would break the potential energy (that is only valid for conservative forces)
    // so remove getPotentialEnery API test here
    this->flags &= ~TestFixture::TEST_POTENTIAL_ENERGY;

    this->debug = false;

    SReal
            k = 1.0,  // stiffness
            d = 0.1,  // damping ratio
            l0 = 1.0; // rest length

    typename TestFixture::Vec3
            x0( 0,0,0), // position of the first particle
            v0(-1,0,0), // velocity of the first particle
            x1( 1,0,0), // position of the second particle
            v1( 1,0,0), // velocity of the second particle
            f0(0.2,0,0); // expected force on the first particle

    this->test_2particles(k,d,l0, x0,v0, x1,v1, f0);
}

// extension, two particles in different nodes
TYPED_TEST( StiffSpringForceField_test , extension_in_parent_and_child )
{
    this->debug = false;

    SReal
            k = 1.0,  // stiffness
            d = 0.1,  // damping ratio
            l0 = 1.0; // rest length

    typename TestFixture::Vec3
            x0(0,0,0), // position of the first particle
            v0(0,0,0), // velocity of the first particle
            x1(2,0,0), // position of the second particle
            v1(0,0,0), // velocity of the second particle
            f0(1,0,0); // expected force on the first particle


    // use the parent  class to automatically test the functions
    this->test_2particles_in_parent_and_child(k,d,l0, x0,v0, x1,v1, f0);
}


} // namespace sofa
