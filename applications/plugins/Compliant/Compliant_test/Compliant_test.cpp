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
/** \file Compliant test suite main file */
// Francois Faure,

#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/vector.h>

#include <sofa/component/init.h>
#include <sofa/component/mass/UniformMass.h>
#include <sofa/component/forcefield/ConstantForceField.h>
#include <sofa/component/mapping/SubsetMultiMapping.h>
#include <sofa/component/mapping/RigidMapping.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/component/topology/EdgeSetTopologyContainer.h>
#include <sofa/component/projectiveconstraintset/FixedConstraint.h>

#include <plugins/Compliant/Compliant_lib/ComplianceSolver.h>
#include <plugins/Compliant/Compliant_lib/UniformCompliance.h>
#include <plugins/Flexible/deformationMapping/ExtensionMapping.h>
#include <plugins/Flexible/deformationMapping/DistanceMapping.h>

#include <sofa/helper/ArgumentParser.h>
#include <sofa/simulation/common/xml/initXml.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/helper/system/PluginManager.h>
#ifdef SOFA_HAVE_DAG
#include <sofa/simulation/graph/DAGSimulation.h>
#endif
#ifdef SOFA_HAVE_BGL
#include <sofa/simulation/bgl/BglSimulation.h>
#endif
#include <sofa/component/misc/ReadState.h>
#include <sofa/component/misc/CompareState.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glut.h>
#include <sofa/helper/system/atomic.h>


#include <Eigen/Dense>
using std::cout;

using namespace sofa;
using namespace sofa::component;
using sofa::helper::vector;

/** Test suite for matrix assembly of class sofa::component::odesolver::ComplianceSolver.
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
    typedef projectiveconstraintset::FixedConstraint<Vec3Types> FixedConstraint3;
    typedef mapping::SubsetMultiMapping<Vec3Types,Vec3Types> SubsetMultiMapping3_to_3;

    // Vec1
    typedef defaulttype::Vec<1,SReal> Vec1;
    typedef defaulttype::StdVectorTypes<Vec1,Vec1> Vec1Types;
    typedef container::MechanicalObject<Vec1Types> MechanicalObject1;
    typedef forcefield::UniformCompliance<Vec1Types> UniformCompliance1;

    // Vec3-Vec1
    typedef mapping::ExtensionMapping<MechanicalObject3::DataTypes, MechanicalObject1::DataTypes> ExtensionMapping31;
    typedef mapping::DistanceMapping<MechanicalObject3::DataTypes, MechanicalObject1::DataTypes> DistanceMapping31;

    // Rigid3
    typedef defaulttype::StdRigidTypes<3,Real> Rigid3Types;
    typedef Rigid3Types::Coord Rigid3Coord;
    typedef Rigid3Types::Deriv Rigid3Deriv;
    typedef container::MechanicalObject<Rigid3Types> MechanicalObjectRigid;
    typedef defaulttype::RigidMass<3,Real> Rigid3Mass;
    typedef mass::UniformMass<Rigid3Types,Rigid3Mass> UniformMassRigid;

    // Rigid3-Vec3
    typedef mapping::RigidMapping<Rigid3Types,Vec3Types> RigidMapping33;


protected:
    /** @name Helpers */
    ///@{

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
        extensions->setName(oss.str()+"_extensionsDOF");

        EdgeSetTopologyContainer::SPtr edgeSet = New<EdgeSetTopologyContainer>();
        extension_node->addObject(edgeSet);

        ExtensionMapping31::SPtr extensionMapping = New<ExtensionMapping31>();
        extensionMapping->setName(oss.str()+"_extensionsMapping");
        extensionMapping->setModels( DOF.get(), extensions.get() );
        extension_node->addObject( extensionMapping );

        UniformCompliance1::SPtr compliance = New<UniformCompliance1>();
        extension_node->addObject(compliance);
        compliance->setName(oss.str()+"_extensionsCompliance");
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
    static bool matricesAreEqual( const DenseMatrix m1, const SMatrix& sm2, double tolerance=100*std::numeric_limits<double>::epsilon() )
    {
        DenseMatrix m2 = sm2;
        if( m1.rows()!=m2.rows() || m1.cols()!=m2.cols() ) return false;

        DenseMatrix diff = m1 - m2;
        bool areEqual = abs(diff.maxCoeff()<tolerance && abs(diff.minCoeff()<tolerance));
        if( !areEqual )
        {
            cerr<<"matricesAreEqual, tolerance = "<< tolerance << ", difference = " << endl << diff << endl;
        }
        return areEqual;

    }

    /// return true if the matrices have same size and all their entries are equal within the given tolerance
    static bool matricesAreEqual( const SMatrix m1, const SMatrix& m2, double tolerance=100*std::numeric_limits<double>::epsilon() )
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
    static bool vectorsAreEqual( const Vector& m1, const Vector& m2, double tolerance=100*std::numeric_limits<double>::epsilon() )
    {
        if( m1.size()!=m2.size() )
        {
            cerr<<"vectorsAreEqual: sizes " << m1.size() << " != " << m2.size() << endl;
            return false;
        }

        Vector diff = m1-m2;
        bool areEqual = abs(diff.maxCoeff()<tolerance && abs(diff.minCoeff()<tolerance));
        if( !areEqual )
        {
            cerr<<"matricesAreEqual, tolerance = "<< tolerance << ", difference = " << endl << diff << endl;
        }
        return areEqual;
    }

    /// create a new component with the given name, and attach it to the given node
    template<class Component>
    static typename Component::SPtr addObject( std::string name, simulation::Node::SPtr parent )
    {
        typename Component::SPtr c = New<Component>();
        parent->addObject(c);
        c->setName(name);
        return c;
    }

    ///@}


    /** Expected results of the different tests. */
    struct
    {
        DenseMatrix M,C,J,P;
        Vector f,phi,dv,lambda;
    } expected;

    ComplianceSolver::SPtr complianceSolver; ///< Solver used to perform the test simulation, and which contains the actual results, to be compared with the expected ones.

    Node::SPtr root;                 ///< Root of the scene graph, created by the constructor an re-used in the tests
    Simulation* simulation;          ///< created by the constructor an re-used in the tests


    // ========================================
public:
    CompliantTestFixture()
    {
//        sofa::helper::BackTrace::autodump();
//        sofa::core::ExecParams::defaultInstance()->setAspectID(0);

        sofa::component::init();
//        sofa::simulation::xml::initXml();

//        std::vector<std::string> plugins;
//        plugins.push_back(std::string("Flexible"));
//        for (unsigned int i=0;i<plugins.size();i++)
//            sofa::helper::system::PluginManager::getInstance().loadPlugin(plugins[i]);

//        sofa::helper::system::PluginManager::getInstance().init();


//        sofa::gui::initMain();
//        if (int err = sofa::gui::GUIManager::Init("argv[0]","batch") )
//                cerr<<"sofa::gui::GUIManager::Init failed " << endl;

//        if (int err=sofa::gui::GUIManager::createGUI(NULL))
//            cerr<<"sofa::gui::GUIManager::createGUI failed " << endl;

        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());
//        sofa::simulation::setSimulation(simulation = new sofa::simulation::bgl::BglSimulation());
        root = simulation->createNewGraph("root");
        root->setName("Scene root");

    }

    ~CompliantTestFixture()
    {
        clear();
    }


    /** @defgroup ComplianceSolver_Unit_Tests ComplianceSolver Assembly Tests
     These methods create a scene, run a short simulation, and save the expected matrix and vector values in the  expected member struct.
     Each test in performed by an external function which runs the test method, then compares the results with the expected results.
     This tests suite is designed to test assembly, and involves only unit masses and perfectly hard constraints.

     The following notations are used in the documentation :
     - \f$ I_{n} \f$ is the identity matrix of size \f$ n \times n \f$
     - \f$ O_{n} \f$ is the null matrix of size \f$ n \times n \f$
     */
    ///@{

    /** An object with internal constraints.
      String of particles with unit mass, connected with rigid links, undergoing two opposed forces at the ends.
      The particles remain at equilibrium, and the internal force is equal to the external force.

      The equation structure is:
        \f[
        \left( \begin{array}{c|c}
        I_{3n} & -J^T \\ \hline
        J & O_{n-1}
        \end{array} \right)

        \left( \begin{array}{c}
        dv \\ \lambda
        \end{array} \right)
        =
        \left( \begin{array}{c}
        f_e \\ \phi
        \end{array} \right)
        \f]
        where Jacobian  J is a block-bidiagonal matrix with opposite unit vectors in each row.

      \param n the number of particles

      \post dv is null because equilibrium
      \post lambda intensities are all equal to the intensity of the external force
      */
    void testHardString( unsigned n )
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
        //        cout<<"dv = " << complianceSolver->getDv().transpose() << endl;
        //        cout<<"lambda = " << complianceSolver->getLambda().transpose() << endl;

        // Expected results
        expected.M = expected.P = DenseMatrix::Identity( 3*n, 3*n );
        expected.C = DenseMatrix::Zero(n-1,n-1); // null
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

    /** An object with internal constraints, subject to a projective constraint.
      String of particles with unit mass, connected with rigid links, attached at one end using a FixedConstraint (projective constraint) and undergoing gravity parallel to the string
      The particles remain at equilibrium, and the internal force is decreasing along with the weight carried by each link.

      The equation structure is:
        \f[
        \left( \begin{array}{c|c}
        I_{3n} & -J^T \\ \hline
        J & O_{n-1}
        \end{array} \right)

        \left( \begin{array}{c}
        dv \\ \lambda
        \end{array} \right)
        =
        \left( \begin{array}{c}
        f_e \\ \phi
        \end{array} \right)
        \f]

      \param n the number of particles

      \post \f$ P =  \left(\begin{array}{cc} O_3 \\ & I_{3(n-1)} \end{array}\right) \f$
      \post dv is null because equilibrium
      \post lambda intensities are equal to g*(n-1), g*(n-2) ... , g
      */
    void testAttachedHardString( unsigned n )
    {
        clear();
        SReal g=10;
        root->setGravity( Vec3(g,0,0) );

        // The solver
        complianceSolver = New<ComplianceSolver>();
        root->addObject( complianceSolver );
        complianceSolver->implicitVelocity.setValue(1.0);
        complianceSolver->implicitPosition.setValue(1.0);
        complianceSolver->f_rayleighMass.setValue(0.0);
        complianceSolver->f_rayleighStiffness.setValue(0.0);

        // The string
        simulation::Node::SPtr  string1 = createCompliantString( root, Vec3(0,0,0), Vec3(1,0,0), n, 1.0*n, 0., 0. );

        // attached
        FixedConstraint3::SPtr fixed1 = New<FixedConstraint3>();
        string1->addObject( fixed1 );



        sofa::simulation::getSimulation()->init(root.get());
        sofa::simulation::getSimulation()->animate(root.get(),1.0);

        //        cout<<"lambda = " << complianceSolver->lambda().transpose() << endl;
        //        cout<<"f = " << complianceSolver->f().transpose() << endl;

        // Expected results
        expected.M = expected.P = DenseMatrix::Identity( 3*n, 3*n );
        for(unsigned i=0; i<3; i++) expected.P(i,i)=0; // fixed point
        expected.C = DenseMatrix::Zero(n-1,n-1); // null
        expected.phi = Vector::Zero(n-1); // null imposed constraint value
        expected.dv = Vector::Zero(3*n);  // equilibrium

        expected.J = DenseMatrix::Zero(n-1,3*n);
        expected.lambda.resize(n-1);
        for(unsigned i=0; i<n-1; i++)
        {
            expected.J(i,3* i    ) = -1;   // block-bidiagonal matrix with opposite unit vectors in each row
            expected.J(i,3*(i+1) ) =  1;
            expected.lambda(i) = -g*(n-1-i);
        }
    }

    /** An object with internal constraints, subject to an additional constraint.
      String of particles with unit mass, connected with rigid links, attached at one end using a hard constraint and undergoing gravity parallel to the string
      The particles remain at equilibrium, and the internal force is decreasing along with the weight carried by each link.

      The equation structure is:
        \f[
        \left( \begin{array}{c|c|c}
        I_{3n} & -J_1^T  & -J_2^T\\ \hline
        J_1    & O_{n-1} & \\ \hline
        J_2    &         & O_1
        \end{array} \right)

        \left( \begin{array}{c}
        dv \\ \lambda
        \end{array} \right)
        =
        \left( \begin{array}{c}
        f_e \\ \phi
        \end{array} \right)
        \f]
        where Jacobian \f$ J_1 \f$  corresponds to the string internal constraints,
        and  Jacobian \f$J_2 \f$ corresponds to the attachment of the string to a fixed point.

      \param n the number of particles

      \post dv is null because equilibrium
      \post lambda intensities are decreasing within the string, then there is the weight of the string: -g*(n-1), ... , -g, g*n
      */
    void testConstrainedHardString( unsigned n )
    {
        clear();
        SReal g=10;
        root->setGravity( Vec3(g,0,0) );

        // The solver
        complianceSolver = New<ComplianceSolver>();
        root->addObject( complianceSolver );
        complianceSolver->implicitVelocity.setValue(1.0);
        complianceSolver->implicitPosition.setValue(1.0);
        complianceSolver->f_rayleighMass.setValue(0.0);
        complianceSolver->f_rayleighStiffness.setValue(0.0);

        // The string
        Vec3 startPoint(0,0,0);
        simulation::Node::SPtr  string1 = createCompliantString( root, startPoint, Vec3(1,0,0), n, 1.0*n, 0., 0. );

        //-------- fix a particle using a constraint: map the distance of the particle to its initial position, and constrain this to 0
        std::string nodeName = "fixDistance_";
        Node::SPtr fixNode = string1->createChild(nodeName + "Node");
        MechanicalObject1::SPtr extensions = New<MechanicalObject1>();
        extensions->setName(nodeName+"DOF");
        fixNode->addObject(extensions);

        DistanceMapping31::SPtr distanceMapping = New<DistanceMapping31>();
        MechanicalObject3* DOF = dynamic_cast<MechanicalObject3*>(string1->getMechanicalState());
        assert(DOF != NULL);
        distanceMapping->setModels(DOF,extensions.get());
        fixNode->addObject( distanceMapping );
        distanceMapping->setName(nodeName+"Mapping");
        distanceMapping->createTarget( 0, startPoint, 0.0 );

        UniformCompliance1::SPtr compliance = New<UniformCompliance1>();
        fixNode->addObject(compliance);
        compliance->setName(nodeName+"Compliance");
        compliance->compliance.setValue(0);
        compliance->dampingRatio.setValue(0);


        // =================  Expected results
        expected.M = expected.P = DenseMatrix::Identity( 3*n, 3*n );
        expected.C = DenseMatrix::Zero(n,n); // null
        expected.phi = Vector::Zero(n-1); // null imposed constraint value
        expected.dv = Vector::Zero(3*n);  // equilibrium

        expected.J = DenseMatrix::Zero(n,3*n);
        expected.lambda.resize(n);
        for(unsigned i=0; i<n-1; i++)
        {
            expected.J(i,3* i    ) = -1;   // block-bidiagonal matrix with opposite unit vectors in each row
            expected.J(i,3*(i+1) ) =  1;
            expected.lambda(i) = -g*(n-1-i);
        }
        expected.J(n-1,0    ) = 1;   // the constrained endpoint
        expected.lambda(n-1) = -g*n;
        //        cerr<<"expected C = " << endl << DenseMatrix(expected.C) << endl;
        //        cerr<<"expected dv = " << expected.dv.transpose() << endl;
        //        cerr<<"expected lambda = " << expected.lambda.transpose() << endl;


        //  ================= Run
        sofa::simulation::getSimulation()->init(root.get());
        sofa::simulation::getSimulation()->animate(root.get(),1.0);

        // actual results
        //        cerr<<"M = " << endl << DenseMatrix(complianceSolver->M()) << endl;
        //        cerr<<"result, J = " << endl << DenseMatrix(complianceSolver->J()) << endl;
        //        cerr<<"result, C = " << endl << DenseMatrix(complianceSolver->C()) << endl;
        //        cerr<<"P = " << endl << DenseMatrix(complianceSolver->P()) << endl;
        //        cerr<<"f = " << complianceSolver->f().transpose() << endl;
        //        cerr<<"phi = " << complianceSolver->phi().transpose() << endl;
        //        cerr<<"dv = " << complianceSolver->getDv().transpose() << endl;
//                cerr<<"lambda = " << complianceSolver->getLambda().transpose() << endl;
        //        cerr<<"lambda - expected.lambda = " << (complianceSolver->getLambda()-expected.lambda).transpose() << endl;

    }

    /** An object with internal constraints, connected using an additional constraint to a fixed point out of the scope of the solver.
      In practical examples, a point out of the scope of the solver may be controlled by the user using the mouse tracker.

      String of particles with unit mass, connected with rigid links, attached at one end using a hard constraint and undergoing gravity parallel to the string
      The particles remain at equilibrium, and the internal force is decreasing along with the weight carried by each link.

      The equation structure is:
        \f[
        \left( \begin{array}{c|c|c}
        I_{3n} & -J_1^T  & -J_2^T\\ \hline
        J_1    & O_{n-1} & \\ \hline
        J_2    &         & O_1
        \end{array} \right)

        \left( \begin{array}{c}
        dv \\ \lambda
        \end{array} \right)
        =
        \left( \begin{array}{c}
        f_e \\ \phi
        \end{array} \right)
        \f]
        where Jacobian \f$ J_1 \f$  corresponds to the string internal constraints,
        and  Jacobian \f$J_2 \f$ corresponds to the attachment of the string to a fixed point.

      \param n the number of particles

      \post dv is null because equilibrium
      \post lambda intensities are decreasing within the string, then there is the total weight of the string: -g*(n-1),... , -g, g*n
      */
    void testExternallyConstrainedHardString( unsigned n )
    {
        clear();
        SReal g=10;
        root->setGravity( Vec3(g,0,0) );

        // ======== object out of scope
        Node::SPtr  outOfScope = root->createChild("outOfScope");
        MechanicalObject3::SPtr outOfScopeDOF = addObject<MechanicalObject3>("outOfScopeDOF",outOfScope);
        outOfScopeDOF->resize(1);
        MechanicalObject3::WriteVecCoord x = outOfScopeDOF->writePositions();
        x[0] = Vec3(0,0,0);


        // ======== object controlled by the solver
        Node::SPtr  solverObject = root->createChild("solverObject");

        // The solver
        complianceSolver = addObject<ComplianceSolver>("complianceSolver",solverObject);
        complianceSolver->implicitVelocity.setValue(1.0);
        complianceSolver->implicitPosition.setValue(1.0);
        complianceSolver->f_rayleighMass.setValue(0.0);
        complianceSolver->f_rayleighStiffness.setValue(0.0);


        // ========  string
        Node::SPtr  string1 = createCompliantString( solverObject, Vec3(0,0,0), Vec3(1,0,0), n, 1.0*n, 0,0 );


        // .=======  Node with multiple parents to create an interaction using a MultiMapping
        Node::SPtr commonChild = string1->createChild("commonChild");
        outOfScope->addChild(commonChild);

        MechanicalObject3::SPtr mappedDOF = addObject<MechanicalObject3>("multiMappedDOF",commonChild); // to contain particles from the two strings

        SubsetMultiMapping3_to_3::SPtr multimapping = New<SubsetMultiMapping3_to_3>();
        multimapping->setName("ConnectionMultiMapping");
        multimapping->addInputModel( string1->getMechanicalState() );
        multimapping->addInputModel( outOfScope->getMechanicalState() );
        multimapping->addOutputModel( mappedDOF.get() );
        multimapping->addPoint( string1->getMechanicalState(), 0 ); // first particle of string1
        multimapping->addPoint( outOfScope->getMechanicalState(), 0 );   // with out of scope particle
        commonChild->addObject(multimapping);

        // ..========  Node to handle the extension of the interaction link
        Node::SPtr extension_node = commonChild->createChild("ConnectionExtensionNode");

        MechanicalObject1::SPtr extensions = New<MechanicalObject1>();
        extension_node->addObject(extensions);
        extensions->setName("extensionsDOF");

        EdgeSetTopologyContainer::SPtr edgeSet = New<EdgeSetTopologyContainer>();
        extension_node->addObject(edgeSet);
        edgeSet->addEdge(0,1);

        ExtensionMapping31::SPtr extensionMapping = New<ExtensionMapping31>();
        extensionMapping->setModels(mappedDOF.get(),extensions.get());
        extension_node->addObject( extensionMapping );
        extensionMapping->setName("ConnectionExtension_mapping");


        UniformCompliance1::SPtr compliance = New<UniformCompliance1>();
        extension_node->addObject(compliance);
        compliance->compliance.setName("connectionCompliance");
        compliance->compliance.setValue(0);
        compliance->dampingRatio.setValue(0);


        // =================  Expected results
        expected.M = expected.P = DenseMatrix::Identity( 3*n, 3*n );
        expected.C = DenseMatrix::Zero(n,n); // null
        expected.phi = Vector::Zero(n-1); // null imposed constraint value
        expected.dv = Vector::Zero(3*n);  // equilibrium

        expected.J = DenseMatrix::Zero(n,3*n);
        expected.lambda.resize(n);
        for(unsigned i=0; i<n-1; i++)
        {
            expected.J(i,3* i    ) = -1;   // block-bidiagonal matrix with opposite unit vectors in each row
            expected.J(i,3*(i+1) ) =  1;
            expected.lambda(i) = -g*(n-1-i);
        }
        expected.J(n-1,0      ) = -1;   // the constrained endpoint
        expected.lambda(n-1) = -g*n;
//                cerr<<"expected J = " << endl << DenseMatrix(expected.J) << endl;
        //        cerr<<"expected dv = " << expected.dv.transpose() << endl;
        //        cerr<<"expected lambda = " << expected.lambda.transpose() << endl;


        //  ================= Run
        sofa::simulation::getSimulation()->init(root.get());
        sofa::simulation::getSimulation()->animate(root.get(),1.0);

        // actual results
        //        cerr<<"M = " << endl << DenseMatrix(complianceSolver->M()) << endl;
//                cerr<<"result, J = " << endl << DenseMatrix(complianceSolver->J()) << endl;
        //        cerr<<"result, C = " << endl << DenseMatrix(complianceSolver->C()) << endl;
        //        cerr<<"P = " << endl << DenseMatrix(complianceSolver->P()) << endl;
        //        cerr<<"f = " << complianceSolver->f().transpose() << endl;
        //        cerr<<"phi = " << complianceSolver->phi().transpose() << endl;
        //        cerr<<"dv = " << complianceSolver->getDv().transpose() << endl;
//                cerr<<"lambda = " << complianceSolver->getLambda().transpose() << endl;
        //        cerr<<"lambda - expected.lambda = " << (complianceSolver->getLambda()-expected.lambda).transpose() << endl;

    }

    /** Two objects of the same type with internal constraints, connected by a constraint.
      String of particles with unit mass, connected with rigid links, attached at one end using a FixedConstraint (projective constraint) and undergoing gravity parallel to the string
      The string is composed of two strings connected by a constraint using MultiMapping.
      The particles remain at equilibrium, and the internal force is decreasing along with the weight carried by each link.

      The equation structure is:
        \f[
        \left( \begin{array}{c|c|c|c|c}
        I_{3n}  \\ \hline
         & I_{3n}  \\ \hline
        J_1 & &    O_{n-1}   \\ \hline
            & J_2  & &        O_{n-1}   \\ \hline
         \multicolumn{2}{c|}{J_3}    &  &    & O_1     \\
        \end{array} \right)

        \left( \begin{array}{c}
        dv \\ \lambda
        \end{array} \right)
        =
        \left( \begin{array}{c}
        f_e \\ \phi
        \end{array} \right)
        \f]
        where Jacobian \f$ J_1 \f$  corresponds to the internal constraints of the first string,
              Jacobian \f$ J_2 \f$  corresponds to the internal constraints of the second string,
        and  Jacobian \f$J_3 \f$ corresponds to the attachment of the strings together,
        and the (anti-)symmetric upper part is skipped for convenience.

      \param n the number of particles in each sub-string. The total number is 2*n.

      \post \f$ P= \left( \begin{array}{c|c|c} O_3 & & \\ \hline & I_{3(n-1)} & \\ \hline  & & I_{3(n-1)}   \end{array} \right)\f$
      \post dv is null because equilibrium
      \post lambda intensities are: (1) decreasing within first string, (2) decreasing within second string, (3) weight of the second string to connect it to the first string:  g*(2*n-1),... g*(n),   g*(*n-2),... g, g*(*n-1)
      */
    void testAttachedConnectedHardStrings( unsigned n )
    {
        clear();
        SReal g=10;
        root->setGravity( Vec3(g,0,0) );

        // The solver
        complianceSolver = New<ComplianceSolver>();
        root->addObject( complianceSolver );
        complianceSolver->implicitVelocity.setValue(1.0);
        complianceSolver->implicitPosition.setValue(1.0);
        complianceSolver->f_rayleighMass.setValue(0.0);
        complianceSolver->f_rayleighStiffness.setValue(0.0);

        // ========  strings
        Node::SPtr  string1 = createCompliantString( root, Vec3(0,0,0), Vec3(1,0,0), n, 1.0*n, 0,0 );
        Node::SPtr  string2 = createCompliantString( root, Vec3(2,0,0), Vec3(3,0,0), n, 1.0*n, 0,0 );

        // string1 is attached
        FixedConstraint3::SPtr fixed1 = New<FixedConstraint3>();
        string1->addObject( fixed1 );


        // ========  Node with multiple parents to create an interaction using a MultiMapping
        Node::SPtr commonChild = string1->createChild("commonChild");
        string2->addChild(commonChild);

        MechanicalObject3::SPtr mappedDOF = New<MechanicalObject3>(); // to contain particles from the two strings
        commonChild->addObject(mappedDOF);
        mappedDOF->setName("multiMappedDOF");

        SubsetMultiMapping3_to_3::SPtr multimapping = New<SubsetMultiMapping3_to_3>();
        multimapping->setName("ConnectionMultiMapping");
        multimapping->addInputModel( string1->getMechanicalState() );
        multimapping->addInputModel( string2->getMechanicalState() );
        multimapping->addOutputModel( mappedDOF.get() );
        multimapping->addPoint( string1->getMechanicalState(), n-1 ); // last particle of string1
        multimapping->addPoint( string2->getMechanicalState(), 0 );   // with first of string2
        commonChild->addObject(multimapping);

        //  ========  Node to handle the extension of the interaction link
        Node::SPtr extension_node = commonChild->createChild("ConnectionExtensionNode");

        MechanicalObject1::SPtr extensions = New<MechanicalObject1>();
        extension_node->addObject(extensions);
        extensions->setName("extensionsDOF");

        EdgeSetTopologyContainer::SPtr edgeSet = New<EdgeSetTopologyContainer>();
        extension_node->addObject(edgeSet);
        edgeSet->addEdge(0,1);

        ExtensionMapping31::SPtr extensionMapping = New<ExtensionMapping31>();
        extensionMapping->setModels(mappedDOF.get(),extensions.get());
        extension_node->addObject( extensionMapping );
        extensionMapping->setName("ConnectionExtension_mapping");


        UniformCompliance1::SPtr compliance = New<UniformCompliance1>();
        extension_node->addObject(compliance);
        compliance->compliance.setName("connectionCompliance");
        compliance->compliance.setValue(0);
        compliance->dampingRatio.setValue(0);


        // Expected results
        expected.M = expected.P = DenseMatrix::Identity( 6*n, 6*n );
        for(unsigned i=0; i<3; i++) expected.P(i,i)=0; // fixed point
        expected.C = DenseMatrix::Zero(2*n-1,2*n-1); // null
        expected.phi = Vector::Zero(2*n-1); // null imposed constraint value
        expected.dv = Vector::Zero(6*n);  // equilibrium

        expected.J = DenseMatrix::Zero(2*n-1,6*n);
        expected.lambda.resize(2*n-1);
        // first string
        for(unsigned i=0; i<n-1; i++)
        {
            expected.J(i,3* i    ) = -1;   // block-bidiagonal matrix with opposite unit vectors in each row
            expected.J(i,3*(i+1) ) =  1;
            expected.lambda(i) = -g*(2*n-1-i);
        }
        // second string
        for(unsigned i=0; i<n-1; i++)
        {
            expected.J((n-1)+i,3*(n+i  ) ) = -1;   // block-bidiagonal matrix with opposite unit vectors in each row
            expected.J((n-1)+i,3*(n+i+1) ) =  1;
            expected.lambda((n-1)+i) = -g*(n-1-i);
        }
        // the connection constraint between the two strings is the last constraint
        expected.J(2*n-2,3*(n-1) ) = -1;   // last particle of the first string
        expected.J(2*n-2,3*(n  ) ) =  1;   // with first particle of the second string
        expected.lambda(2*n-2) = -g*(n);     // weight of the second string
//        cerr<<"expected J = " << endl << DenseMatrix(expected.J) << endl;
//        cerr<<"expected lambda = " << expected.lambda.transpose() << endl;


        sofa::simulation::getSimulation()->init(root.get());
        sofa::simulation::getSimulation()->animate(root.get(),1.0);

        // actual results
//        cerr<<"M = " << endl << DenseMatrix(complianceSolver->M()) << endl;
//        cerr<<"J = " << endl << DenseMatrix(complianceSolver->J()) << endl;
//        cerr<<"C = " << endl << DenseMatrix(complianceSolver->C()) << endl;
//        cerr<<"P = " << endl << DenseMatrix(complianceSolver->P()) << endl;
//        cerr<<"f = " << endl << complianceSolver->f().transpose() << endl;
//        cerr<<"phi = " << complianceSolver->phi().transpose() << endl;
//        cerr<<"actual dv = " << complianceSolver->getDv().transpose() << endl;
//        cerr<<"actual lambda = " << complianceSolver->getLambda().transpose() << endl;

    }

    /** Two objects of different types, connected by a constraint.
      Rigid connected to particles.
      A hard string of particles is attached on the left, and connected to a rigid body on the right.
      The connection is created using a point mapped from the rigid body and constraint between this point and the last point of the string.

      The equation structure is:
        \f[
        \left( \begin{array}{c|c|c|c}
        I_{3n} \\ \hline
         & I_6  \\ \hline
        J_1 &     & O_{n-1} \\ \hline
            \multicolumn{2}{c|}{J_2} &        & O_3
        \end{array} \right)

        \left( \begin{array}{c}
        dv \\ \lambda
        \end{array} \right)
        =
        \left( \begin{array}{c}
        f_e \\ \phi
        \end{array} \right)
        \f]
        where Jacobian \f$ J_1 \f$  corresponds to the internal constraints of the first string,
              Jacobian \f$ J_2 \f$  corresponds to the attachment of the string to the rigid object,
        and the (anti-)symmetric upper part is skipped for convenience.

      \post \f$ P= \left( \begin{array}{c|c|c} O_3 & & \\ \hline & I_{3(n-1)} & \\ \hline  & & I_{6}   \end{array} \right)\f$
      \post dv is null because equilibrium
      \post lambda intensities are equal to g*(n-1), g*(n-2) ... , g
    */
    void testRigidConnectedToString( unsigned n )
    {
        clear();
        SReal g=10;
        root->setGravity( Vec3(g,0,0) );

        // The solver
        complianceSolver = New<ComplianceSolver>();
        root->addObject( complianceSolver );
        complianceSolver->implicitVelocity.setValue(1.0);
        complianceSolver->implicitPosition.setValue(1.0);
        complianceSolver->f_rayleighMass.setValue(0.0);
        complianceSolver->f_rayleighStiffness.setValue(0.0);

        // ========= The rigid object
        simulation::Node::SPtr rigid = root->createChild("rigid");
        MechanicalObjectRigid::SPtr rigidDOF = addObject<MechanicalObjectRigid>("rigidDOF",rigid);
        rigidDOF->resize(1);
        MechanicalObjectRigid::WriteVecCoord x = rigidDOF->writePositions();
        x[0].getCenter() = Vec3(n,0,0);
        UniformMassRigid::SPtr rigidMass = addObject<UniformMassRigid>("rigidMass",rigid);

        // .========= Particle attached to the rigid object
        simulation::Node::SPtr particleOnRigid = rigid->createChild("particleOnRigid");
        MechanicalObject3::SPtr particleOnRigidDOF = addObject<MechanicalObject3>("particleOnRigidDOF",particleOnRigid);
        particleOnRigidDOF->resize(1);
        RigidMapping33::SPtr particleOnRigidMapping = addObject<RigidMapping33>("particleOnRigidMapping",particleOnRigid);
        particleOnRigidMapping->setModels(rigidDOF.get(),particleOnRigidDOF.get());

        // ========= The string
        simulation::Node::SPtr  string1 = createCompliantString( root, Vec3(0,0,0), Vec3(1,0,0), n, 1.0*n, 0., 0. );

        // Fix the first particle of the string
        FixedConstraint3::SPtr fixed1 = New<FixedConstraint3>();
        string1->addObject( fixed1 );

        // ..======== Mixed subset containing the last particle of the string and the particle attached to the rigid object
        simulation::Node::SPtr pointPair = particleOnRigid->createChild("pointPair");
        string1->addChild(pointPair); // two parents: particleOnRigid and string1

        MechanicalObject3::SPtr pointPairDOF = addObject<MechanicalObject3>("pointPairDOF",pointPair);
        SubsetMultiMapping3_to_3::SPtr pointPairMapping = addObject<SubsetMultiMapping3_to_3>("pointPairMapping",pointPair);
        pointPairMapping->addInputModel(string1->mechanicalState);
        pointPairMapping->addInputModel(particleOnRigid->mechanicalState);
        pointPairMapping->addOutputModel(pointPair->mechanicalState);
        pointPairMapping->addPoint(string1->mechanicalState,n-1 );     // last particle
        pointPairMapping->addPoint(particleOnRigid->mechanicalState,0 );

        //  ...========  Distance between the particles in pointPair
        Node::SPtr extension = pointPair->createChild("extension");

        MechanicalObject1::SPtr extensionDOF = addObject<MechanicalObject1>("extensionDOF",extension);

        EdgeSetTopologyContainer::SPtr extensionEdgeSet = addObject<EdgeSetTopologyContainer>("extensionEdgeSet",extension);
        extensionEdgeSet->addEdge(0,1);

        ExtensionMapping31::SPtr extensionMapping = addObject<ExtensionMapping31>("extensionMapping",extension);
        extensionMapping->setModels(pointPairDOF.get(),extensionDOF.get());
//        helper::WriteAccessor< Data< vector< Real > > > restLengths( extensionMapping->f_restLengths );
//        restLengths.resize(1);
//        restLengths[0] = 1.0;

        UniformCompliance1::SPtr extensionCompliance = addObject<UniformCompliance1>("extensionCompliance",extension);
        extensionCompliance->compliance.setValue(0);
        extensionCompliance->dampingRatio.setValue(0);



        // ***** Expected results
        unsigned nM = 3*n+6;  // n particles + 1 rigid
        unsigned nC = n;      // n-1 in the string + 1 to connect the string to the rigid
        expected.M = expected.P = DenseMatrix::Identity( nM,nM );
        for(unsigned i=0; i<3; i++) expected.P(6+i,6+i)=0; // fixed point
        expected.dv = Vector::Zero(nM);        // equilibrium
        expected.C = DenseMatrix::Zero(nC,nC); // null compliance
        expected.phi = Vector::Zero(nC);       // null imposed constraint value

        expected.J = DenseMatrix::Zero(nC,nM);
        expected.lambda.resize(nC);
        // string
        for(unsigned i=0; i<nC-1; i++)
        {
            expected.J(i,6+3* i    ) = -1;   // block-bidiagonal matrix with opposite unit vectors in each row
            expected.J(i,6+3*(i+1) ) =  1;
            expected.lambda(i) = -g*(nC-i); // weight: nC-i-1 particle + the rigid
        }
        // the connection constraint between the two strings is the last constraint
        expected.J( nC-1, 6+3*(n-1) ) = -1;   // last particle of the first string
        expected.J( nC-1, 0 ) =  1;           // with the rigid translation
        expected.lambda(nC-1) = -g;           // weight of the rigid
//        cerr<<"expected J = " << endl << DenseMatrix(expected.J) << endl;
//        cerr<<"expected P = " << endl << DenseMatrix(expected.P) << endl;
//        cerr<<"expected lambda = " << expected.lambda.transpose() << endl;


        // ***** Perform simulation
        sofa::simulation::getSimulation()->init(root.get());
        sofa::simulation::getSimulation()->animate(root.get(),1.0);

//        // actual results
//        cerr<<"M = " << endl << DenseMatrix(complianceSolver->M()) << endl;
//        cerr<<"J = " << endl << DenseMatrix(complianceSolver->J()) << endl;
//        cerr<<"C = " << endl << DenseMatrix(complianceSolver->C()) << endl;
//        cerr<<"P = " << endl << DenseMatrix(complianceSolver->P()) << endl;
//        cerr<<"f = " << endl << complianceSolver->f().transpose() << endl;
//        cerr<<"phi = " << complianceSolver->phi().transpose() << endl;
//        cerr<<"actual dv = " << complianceSolver->getDv().transpose() << endl;
//        cerr<<"actual lambda = " << complianceSolver->getLambda().transpose() << endl;
    }

    ///@}



};



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
#ifdef WIN32
#define BOOST_TEST_INCLUDED
#endif
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_log.hpp>
#if BOOST_WORKAROUND(  __GNUC__, < 3 )
#include <boost/test/output_test_stream.hpp>
typedef boost::test_tools::output_test_stream onullstream_type;
#else
#include <boost/test/utils/nullstream.hpp>
typedef boost::onullstream onullstream_type;
#endif
namespace ut = boost::unit_test;


BOOST_FIXTURE_TEST_SUITE( ts1, CompliantTestFixture );

BOOST_AUTO_TEST_CASE( test_CompliantSolver_assembly )
{
    ut::unit_test_log.set_stream( std::cerr );




    unsigned numParticles=3;
    BOOST_TEST_MESSAGE( "CompliantTestFixture: hard string of " << numParticles << " particles");
    testHardString(numParticles);
    BOOST_CHECK(matricesAreEqual( expected.M, complianceSolver->M() ));
    BOOST_CHECK(matricesAreEqual( expected.P, complianceSolver->P() ));
    BOOST_CHECK(matricesAreEqual( expected.J, complianceSolver->J() ));
    BOOST_CHECK(matricesAreEqual( expected.C, complianceSolver->C() ));
    BOOST_CHECK(vectorsAreEqual( expected.dv, complianceSolver->getDv() ));
    BOOST_CHECK(vectorsAreEqual( expected.lambda, complianceSolver->getLambda() ));

    BOOST_TEST_MESSAGE( "CompliantTestFixture: hard string of " << numParticles << " particles attached using a projective constraint (FixedConstraint)");
    testAttachedHardString(numParticles);
    BOOST_CHECK(matricesAreEqual( expected.M, complianceSolver->M() ));
    BOOST_CHECK(matricesAreEqual( expected.P, complianceSolver->P() ));
    BOOST_CHECK(matricesAreEqual( expected.J, complianceSolver->J() ));
    BOOST_CHECK(matricesAreEqual( expected.C, complianceSolver->C() ));
    BOOST_CHECK(vectorsAreEqual( expected.dv, complianceSolver->getDv() ));
    BOOST_CHECK(vectorsAreEqual( expected.lambda, complianceSolver->getLambda() ));

    numParticles=4;
    BOOST_TEST_MESSAGE( "CompliantTestFixture: hard string of " << numParticles << " particles attached using a distance constraint");
    testConstrainedHardString(numParticles);
    BOOST_CHECK(matricesAreEqual( expected.M, complianceSolver->M() ));
    BOOST_CHECK(matricesAreEqual( expected.P, complianceSolver->P() ));
    BOOST_CHECK(matricesAreEqual( expected.J, complianceSolver->J() ));
    BOOST_CHECK(matricesAreEqual( expected.C, complianceSolver->C() ));
    BOOST_CHECK(vectorsAreEqual( expected.dv, complianceSolver->getDv() ));
    BOOST_CHECK(vectorsAreEqual( expected.lambda, complianceSolver->getLambda() ));

    numParticles=2;
    BOOST_TEST_MESSAGE( "CompliantTestFixture: hard string of " << numParticles << " particles attached using a constraint with an out-of-scope particle");
    testExternallyConstrainedHardString(numParticles);
    BOOST_CHECK(matricesAreEqual( expected.M, complianceSolver->M() ));
    BOOST_CHECK(matricesAreEqual( expected.P, complianceSolver->P() ));
    BOOST_CHECK(matricesAreEqual( expected.J, complianceSolver->J() ));
    BOOST_CHECK(matricesAreEqual( expected.C, complianceSolver->C() ));
    BOOST_CHECK(vectorsAreEqual( expected.dv, complianceSolver->getDv() ));
    BOOST_CHECK(vectorsAreEqual( expected.lambda, complianceSolver->getLambda() ));

    numParticles=3;
    BOOST_TEST_MESSAGE( "CompliantTestFixture: hard strings of " << numParticles << " particles connected using a MultiMapping");
    testAttachedConnectedHardStrings(numParticles);
    BOOST_CHECK(matricesAreEqual( expected.M, complianceSolver->M() ));
    BOOST_CHECK(matricesAreEqual( expected.P, complianceSolver->P() ));
    BOOST_CHECK(matricesAreEqual( expected.J, complianceSolver->J() ));
    BOOST_CHECK(matricesAreEqual( expected.C, complianceSolver->C() ));
    BOOST_CHECK(vectorsAreEqual( expected.dv, complianceSolver->getDv() ));
    BOOST_CHECK(vectorsAreEqual( expected.lambda, complianceSolver->getLambda() ));

    numParticles=2;
    BOOST_TEST_MESSAGE( "CompliantTestFixture: hard string of " << numParticles << " particles connected to a rigid");
    testRigidConnectedToString(numParticles);
    BOOST_CHECK(matricesAreEqual( expected.M, complianceSolver->M() ));
    BOOST_CHECK(matricesAreEqual( expected.P, complianceSolver->P() ));
    BOOST_CHECK(matricesAreEqual( expected.J, complianceSolver->J() ));
    BOOST_CHECK(matricesAreEqual( expected.C, complianceSolver->C() ));
    BOOST_CHECK(vectorsAreEqual( expected.dv, complianceSolver->getDv() ));
    BOOST_CHECK(vectorsAreEqual( expected.lambda, complianceSolver->getLambda() ));
}

BOOST_AUTO_TEST_SUITE_END();




