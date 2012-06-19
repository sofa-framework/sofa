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
/** \file test suite file */
#include <sofa/simulation/bgl/BglSimulation.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/vector.h>

//#include <sofa/component/init.h>
//#ifdef SOFA_DEV
//#include <sofa/component/initDev.h>
//#endif
#include <sofa/component/mass/UniformMass.h>
#include <sofa/component/forcefield/ConstantForceField.h>
#include <sofa/component/mapping/SubsetMultiMapping.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/component/topology/EdgeSetTopologyContainer.h>

#include <plugins/Compliant/Compliant_lib/ComplianceSolver.h>
#include <plugins/Compliant/Compliant_lib/UniformCompliance.h>
#include <plugins/Flexible/deformationMapping/ExtensionMapping.h>
#include <plugins/Flexible/deformationMapping/DistanceMapping.h>

#include <Eigen/Dense>
using std::cout;

using namespace sofa;
using namespace sofa::component;
using sofa::helper::vector;

/** Test suite for class sofa::component::odesolver::ComplianceSolver.
The unit tests are defined in group  \ref ComplianceSolver_Unit_Tests
 */
class CompliantTestFixture
{
protected:
    typedef SReal Real;
    typedef odesolver::ComplianceSolver ComplianceSolver;
    typedef ComplianceSolver::SMatrix SMatrix;
    typedef topology::EdgeSetTopologyContainer EdgeSetTopologyContainer;
    typedef simulation::Node Node;
    typedef simulation::Simulation Simulation;
    typedef Eigen::MatrixXd DenseMatrix;
    typedef Eigen::VectorXd Vector;

    // Vec3
    typedef defaulttype::Vec<3,SReal> Vec3;
    typedef defaulttype::StdVectorTypes<Vec3,Vec3> Vec3Types;
    typedef container::MechanicalObject<Vec3Types> MechanicalObject3;
    typedef mass::UniformMass<Vec3Types,Real> UniformMass3;
    typedef forcefield::ConstantForceField<Vec3Types> ConstantForceField3;

    // Vec1
    typedef defaulttype::Vec<1,SReal> Vec1;
    typedef defaulttype::StdVectorTypes<Vec1,Vec1> Vec1Types;
    typedef container::MechanicalObject<Vec1Types> MechanicalObject1;
    typedef forcefield::UniformCompliance<Vec1Types> UniformCompliance1;

    // Vec3-Vec1
    typedef mapping::ExtensionMapping<MechanicalObject3::DataTypes, MechanicalObject1::DataTypes> ExtensionMapping31;
    typedef mapping::DistanceMapping<MechanicalObject3::DataTypes, MechanicalObject1::DataTypes> DistanceMapping31;



    /** Store the expected results of the different tests. */
    struct
    {
        DenseMatrix M,C,J,P;
        Vector f,phi,dv,lambda;
    } expected;
    ComplianceSolver::SPtr complianceSolver; ///< Solver used to perform the test simulation, and which contains the actual results, to be compared with the expected ones.
    Node::SPtr root;
    Simulation* simulation;



public:
    CompliantTestFixture()
    {
        sofa::simulation::setSimulation(new sofa::simulation::bgl::BglSimulation());
        simulation = sofa::simulation::getSimulation();
        root = simulation->createNewGraph("root");
    }

    ~CompliantTestFixture()
    {
        clear();
    }


    /** @defgroup ComplianceSolver_Unit_Tests ComplianceSolver Unit Tests
     These methods create a scene, run a short simulation, and save the expected matrix and vector values in the  expected member struct.
     Each test in performed by an external function which runs the test method, then compares the results with the expected results.
     */
    ///@{


    /** String of particles with unit mass, connected with rigid links, undergoing two opposed forces at the ends.
      The particles remain at equilibrium, and the internal force is equal to the external force.

      \param n the number of particles

      \post M is the identity of size 3*n
      \post P is the identity of size 3*n (no projection)
      \post J is a block-bidiagonal matrix with opposite unit vectors in each row
      \post C is null because rigidity
      \post f is null because final net force is null
      \post phi is null because a null because rigidity
      \post dv is null because equilibrium
      \post lambda intensities are equal to the intensity of the external force
      */
    void testRigidString( unsigned n )
    {
        clear();
        root->setGravity( Vec3(0,0,0) );

        // The solver
        complianceSolver = New<ComplianceSolver>();
        root->addObject( complianceSolver );
        complianceSolver->implicitVelocity.setValue(1.0);
        complianceSolver->implicitPosition.setValue(1.0);
        complianceSolver->f_rayleighMass.setValue(0.0);
        complianceSolver->f_rayleighStiffness.setValue(0.0);

        // The string
        simulation::Node::SPtr  string1 = createCompliantString( root, Vec3(0,0,0), Vec3(1,0,0), n, 1.0*n, 0., 0. );

        // Opposite forces applied to the ends
        ConstantForceField3::SPtr ff = New<ConstantForceField3>();
        string1->addObject(ff);
        vector<unsigned>* indices =  ff->points.beginEdit(); // not managed to create a WriteAccessor with a resize function for a ConstantForceField::SetIndex
        helper::WriteAccessor< Data<vector<Vec3> > > forces( ff->forces );
        (*indices).resize(2);
        forces.resize(2);
        // pull the left-hand particle to the left
        (*indices)[0]= 0; forces[0]= Vec3(-1,0,0);
        // pull the right-hand particle to the right
        (*indices)[1]= n-1; forces[1]= Vec3(1,0,0);
        ff->points.endEdit();


        sofa::simulation::getSimulation()->init(root.get());
        sofa::simulation::getSimulation()->animate(root.get(),1.0);

//        // actual results
//        cout<<"M = " << complianceSolver->M() << endl;
//        cout<<"J = " << complianceSolver->J() << endl;
//        cout<<"C = " << complianceSolver->C() << endl;
//        cout<<"P = " << complianceSolver->P() << endl;
//        cout<<"f = " << complianceSolver->f().transpose() << endl;
//        cout<<"phi = " << complianceSolver->phi().transpose() << endl;
//        cout<<"dv = " << complianceSolver->dv().transpose() << endl;
//        cout<<"lambda = " << complianceSolver->lambda().transpose() << endl;

        // Expected results
        expected.M = expected.P = DenseMatrix::Identity( 3*n, 3*n );
        expected.C = DenseMatrix::Zero(n-1,n-1); // null
        expected.f = Vector::Zero(3*n);   // final net force is null
        expected.phi = Vector::Zero(n-1); // null imposed constraint value
        expected.dv = Vector::Zero(3*n);  // equilibrium

        expected.J = DenseMatrix::Zero(n-1,3*n);
        expected.lambda.resize(n-1);
        for(unsigned i=0; i<n-1; i++)
        {
            expected.J(i,3* i    ) = -1;   // block-bidiagonal matrix with opposite unit vectors in each row
            expected.J(i,3*(i+1) ) =  1;
            expected.lambda(i) = -1;       // internal forces intensity = external force
        }
    }

    ///@}


protected:
    /// Helper method to create strings used in various tests.
    Node::SPtr createCompliantString(simulation::Node::SPtr parent, Vec3 startPoint, Vec3 endPoint, unsigned numParticles, double totalMass, double complianceValue=0, double dampingRatio=0 )
    {
        static unsigned numObject = 1;
        std::ostringstream oss;
        oss << "string_" << numObject++;
        SReal totalLength = (endPoint-startPoint).norm();

        //--------
        simulation::Node::SPtr  string_node = parent->createChild(oss.str());

        MechanicalObject3::SPtr DOF = New<MechanicalObject3>();
        string_node->addObject(DOF);
        DOF->setName(oss.str()+"_DOF");

        UniformMass3::SPtr mass = New<UniformMass3>();
        string_node->addObject(mass);
        mass->setName(oss.str()+"_mass");
        mass->mass.setValue( totalMass/numParticles );




        //--------
        simulation::Node::SPtr extension_node = string_node->createChild( oss.str()+"_ExtensionNode");

        MechanicalObject1::SPtr extensions = New<MechanicalObject1>();
        extension_node->addObject(extensions);

        EdgeSetTopologyContainer::SPtr edgeSet = New<EdgeSetTopologyContainer>();
        extension_node->addObject(edgeSet);

        ExtensionMapping31::SPtr extensionMapping = New<ExtensionMapping31>();
        extensionMapping->setModels(DOF.get(),extensions.get());
        extension_node->addObject( extensionMapping );
        extensionMapping->setName(oss.str()+"_ExtensionMapping");
        extensionMapping->setModels( DOF.get(), extensions.get() );

        UniformCompliance1::SPtr compliance = New<UniformCompliance1>();
        extension_node->addObject(compliance);
        compliance->setName(oss.str()+"_compliance");
        compliance->compliance.setValue(complianceValue);
        compliance->dampingRatio.setValue(dampingRatio);


        //--------
        // create the particles
        DOF->resize(numParticles);
        MechanicalObject3::WriteVecCoord x = DOF->writePositions();
        helper::vector<SReal> restLengths;
        for( unsigned i=0; i<numParticles; i++ )
        {
            double alpha = (double)i/(numParticles-1);
            x[i] = startPoint * (1-alpha)  +  endPoint * alpha;
            if(i>0)
            {
                edgeSet->addEdge(i-1,i);
                restLengths.push_back( totalLength/(numParticles-1) );
            }
        }
        extensionMapping->f_restLengths.setValue( restLengths );

        //    {
        //        //-------- fix a particle
        //        Node::SPtr fixNode = string_node->createChild("fixNode");
        //        MechanicalObject1::SPtr extensions = New<MechanicalObject1>();
        //        fixNode->addObject(extensions);

        //        DistanceMapping31::SPtr distanceMapping = New<DistanceMapping31>();
        //        distanceMapping->setModels(DOF.get(),extensions.get());
        //        fixNode->addObject( distanceMapping );
        //        distanceMapping->setName("fix_distanceMapping");
        //        distanceMapping->setModels( DOF.get(), extensions.get() );
        //        distanceMapping->createTarget( numParticles-1, endPoint, 0.0 );

        //        UniformCompliance1::SPtr compliance = New<UniformCompliance1>();
        //        fixNode->addObject(compliance);
        //        compliance->setName("fix_compliance");
        //        compliance->compliance.setValue(complianceValue);
        //        compliance->dampingRatio.setValue(dampingRatio);
        //    }

        return string_node;

    }

    /// remove all children nodes
    void clear()
    {
        if( root )
            simulation->unload( root );
        root = simulation->createNewGraph("");
    }

    /// Return an identity matrix, or if not square, a matrix with 1 on each entry of the main diagonal
    static SMatrix makeSparseIdentity( unsigned rows, unsigned cols )
    {
        SMatrix m(rows,cols);
        for(unsigned i=0; i<rows; i++ )
        {
            if(i<cols)
            {
                m.startVec(i);
                m.insertBack(i,i) = 1.0;
            }
        }
        m.finalize();
        return m;
    }

    /// Return an identity matrix, or if not square, a matrix with 1 on each entry of the main diagonal
    static DenseMatrix makeIdentity( unsigned rows, unsigned cols )
    {
        DenseMatrix m(rows,cols);
        for(unsigned i=0; i<rows; i++ )
        {
            m(i,i) = 1.0;
        }
        return m;
    }

    /// return true if the matrices have same size and all their entries are equal within the given tolerance
    static bool matricesAreEqual( const DenseMatrix m1, const SMatrix& sm2, double tolerance=std::numeric_limits<double>::epsilon() )
    {
        DenseMatrix m2 = sm2;
        if( m1.rows()!=m2.rows() || m1.cols()!=m2.cols() ) return false;

        DenseMatrix diff = m1 - m2;
        if( !(fabs(diff.maxCoeff()<tolerance && fabs(diff.minCoeff()<tolerance))) )
        {
            cout<<"matricesAreEqual, tolerance = "<< tolerance << ", difference = " << endl << diff << endl;
        }
        return fabs(diff.maxCoeff()<tolerance && fabs(diff.minCoeff()<tolerance));

    }

    /// return true if the matrices have same size and all their entries are equal within the given tolerance
    static bool matricesAreEqual( const SMatrix m1, const SMatrix& m2, double tolerance=std::numeric_limits<double>::epsilon() )
    {
        if( m1.rows()!=m2.rows() || m1.cols()!=m2.cols() ) return false;

        SMatrix diff = m1 - m2;
        for (int k=0; k<diff.outerSize(); ++k)
            for (SMatrix::InnerIterator it(diff,k); it; ++it)
            {
                if( fabs(it.value()) >tolerance )
                {
                    return false;
                }

            }
        return true;

    }

    /// return true if the matrices have same size and all their entries are equal within the given tolerance
    static bool vectorsAreEqual( const Vector& m1, const Vector& m2, double tolerance=std::numeric_limits<double>::epsilon() )
    {
        if( m1.size()!=m2.size() ) return false;
        for( unsigned i=0; i<m1.size(); i++ )
            if( fabs(m1(i)-m2(i))>tolerance  )
            {
//                cout<<"vectorsAreEqual, fabs(" << m1(i) << "-" << m2(i) << ")=" << m1(i)-m2(i) << " > " << tolerance << endl;
                return false;
            }
        return true;
    }



};

//int main(int argc, char** argv)
//{
//    cerr<<"Starting Compliant_test" << endl;
//    CompliantTestFixture fix;
//    fix.makeFirstTest();
//    return 0;
//}


/** \page Page_CompliantTestSuite Compliant plugin test suite
  This test suite uses the Boost Unit Testing Framework. http://www.boost.org/doc/libs/1_49_0/libs/test/doc/html/index.html
  A good introduction can be found in: http://www.ibm.com/developerworks/aix/library/au-ctools1_boost/

  The test suite is run from file Compliant_test.cpp .
  The main() function is actually in an external library. Installatations instructions can be found on the web, e.g.:
  - for linux: http://www.alittlemadness.com/2009/03/31/c-unit-testing-with-boosttest/
  - for windows: http://www.beroux.com/english/articles/boost_unit_testing/

 Currently all the tests are based on the unit test methods defined in class CompliantTestFixture.

  */
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE example
#include <boost/test/included/unit_test.hpp>

BOOST_FIXTURE_TEST_SUITE( ts1, CompliantTestFixture );

BOOST_AUTO_TEST_CASE( rigid_string )
{
    unsigned numParticles=3;
    BOOST_TEST_MESSAGE( "CompliantTestFixture: rigid string of " << numParticles << " particles");
    testRigidString(numParticles);
    BOOST_CHECK(matricesAreEqual( expected.M, complianceSolver->M() ));
    BOOST_CHECK(matricesAreEqual( expected.P, complianceSolver->P() ));
    BOOST_CHECK(matricesAreEqual( expected.J, complianceSolver->J() ));
    BOOST_CHECK(matricesAreEqual( expected.C, complianceSolver->C() ));
    BOOST_CHECK(vectorsAreEqual( expected.f, complianceSolver->f() ));
}

BOOST_AUTO_TEST_SUITE_END();




