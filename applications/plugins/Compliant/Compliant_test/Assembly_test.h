#include "Compliant_test.h"

namespace sofa
{
using namespace modeling;

struct Assembly_test : public CompliantSolver_test
{
    typedef odesolver::AssembledSolver OdeSolver;
    typedef linearsolver::LDLTSolver LinearSolver;
    OdeSolver::SPtr complianceSolver; ///< Solver used to perform the test simulation, and which contains the actual results, to be compared with the expected ones.
    LinearSolver::SPtr linearSolver; ///< Auxiliary linear equation solver used by the ode solver


    /** Expected results of the different tests. */
    struct
    {
        DenseMatrix M,C,J,P;
        Vector f,phi,dv,lambda;
    } expected;


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
        clearScene();
        Node::SPtr root = getRoot();
        getRoot()->setGravity( Vec3(0,0,0) );

        // The solver
        complianceSolver = addNew<OdeSolver>(getRoot());
//        root->addObject( complianceSolver );
        complianceSolver->storeDynamicsSolution(true);
        linearSolver = addNew<LinearSolver>(getRoot());
//        root->addObject( linearSolver);
        complianceSolver->alpha.setValue(1.0);
        complianceSolver->beta.setValue(1.0);

        // The string
        simulation::Node::SPtr  string1 = createCompliantString( root, Vec3(0,0,0), Vec3(1,0,0), n, 1.0*n, 0. );

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

        // actual results
        //        cout<<"M = " << complianceSolver->M() << endl;
        //        cout<<"J = " << complianceSolver->J() << endl;
        //        cout<<"C = " << complianceSolver->C() << endl;
        //        cout<<"P = " << complianceSolver->P() << endl;
        //        cout<<"f = " << complianceSolver->getF().transpose() << endl;
        //        cout<<"phi = " << complianceSolver->getPhi().transpose() << endl;
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
        clearScene();
        SReal g=10;
        Node::SPtr root = getRoot();
        root->setGravity( Vec3(g,0,0) );

        // The solver
        complianceSolver = addNew<OdeSolver>(root);
        complianceSolver->storeDynamicsSolution(true);
        complianceSolver->alpha.setValue(1.0);
        complianceSolver->beta.setValue(1.0);
        //        complianceSolver->debug.setValue(true);
        linearSolver = addNew<LinearSolver>(root);
        //        linearSolver->debug.setValue(true);

        // The string
        simulation::Node::SPtr  string1 = createCompliantString( root, Vec3(0,0,0), Vec3(1,0,0), n, 1.0*n, 0. );

        // attached
        FixedConstraint3::SPtr fixed1 = addNew<FixedConstraint3>(string1);



        sofa::simulation::getSimulation()->init(root.get());
        sofa::simulation::getSimulation()->animate(root.get(),1.0);

        //        cerr<<"lambda = " << complianceSolver->getLambda().transpose() << endl;
        //        cerr<<"f = " << complianceSolver->getF().transpose() << endl;

        // Expected results
        expected.M = expected.P = DenseMatrix::Identity( 3*n, 3*n );
        for(unsigned i=0; i<3; i++)
            expected.P(i,i)=0; // fixed point
        expected.M = expected.P.transpose() * expected.M * expected.P;
        expected.C = DenseMatrix::Zero(n-1,n-1); // null compliance
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
        clearScene();
        SReal g=10;
        Node::SPtr root = getRoot();
        root->setGravity( Vec3(g,0,0) );

        // The solver
        complianceSolver = addNew<OdeSolver>(root);
        complianceSolver->storeDynamicsSolution(true);
        complianceSolver->alpha.setValue(1.0);
        complianceSolver->beta.setValue(1.0);

        linearSolver = addNew<LinearSolver>(root);

        // The string
        Vec3 startPoint(0,0,0);
        simulation::Node::SPtr  string1 = createCompliantString( root, startPoint, Vec3(1,0,0), n, 1.0*n, 0. );

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
        //        cerr<<"f = " << complianceSolver->getF().transpose() << endl;
        //        cerr<<"phi = " << complianceSolver->getPhi().transpose() << endl;
        //        cerr<<"dv = " << complianceSolver->getDv().transpose() << endl;
        //        cerr<<"lambda = " << complianceSolver->getLambda().transpose() << endl;
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
        Node::SPtr root = clearScene();
        SReal g=10;
        root->setGravity( Vec3(g,0,0) );

        // ======== object out of scope
        Node::SPtr  outOfScope = root->createChild("outOfScope");
        MechanicalObject3::SPtr outOfScopeDOF = addNew<MechanicalObject3>(outOfScope,"outOfScopeDOF");
        outOfScopeDOF->resize(1);
        MechanicalObject3::WriteVecCoord x = outOfScopeDOF->writePositions();
        x[0] = Vec3(0,0,0);


        // ======== object controlled by the solver
        Node::SPtr  solverObject = root->createChild("solverObject");

        // The solver
        complianceSolver = addNew<OdeSolver>(solverObject);
        complianceSolver->storeDynamicsSolution(true);
        linearSolver = addNew<LinearSolver>(solverObject);
        complianceSolver->alpha.setValue(1.0);
        complianceSolver->beta.setValue(1.0);


        // ========  string
        Node::SPtr  string1 = createCompliantString( solverObject, Vec3(0,0,0), Vec3(1,0,0), n, 1.0*n, 0 );


        // .=======  Node with multiple parents to create an interaction using a MultiMapping
        Node::SPtr commonChild = string1->createChild("commonChild");
        outOfScope->addChild(commonChild);

        MechanicalObject3::SPtr mappedDOF = addNew<MechanicalObject3>(commonChild); // to contain particles from the two strings

        SubsetMultiMapping3_to_3::SPtr multimapping = New<SubsetMultiMapping3_to_3>();
        multimapping->setName("ConnectionMultiMapping");
        multimapping->addInputModel( string1->getMechanicalState() );
        multimapping->addInputModel( outOfScope->getMechanicalState() );
        multimapping->addOutputModel( mappedDOF.get() );
        multimapping->addPoint( string1->getMechanicalState(), 0 );      // first particle of string1
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
        //        cerr<<"expected J = " << endl << DenseMatrix(expected.J) << endl;
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
        //        cerr<<"f = " << complianceSolver->getF().transpose() << endl;
        //        cerr<<"phi = " << complianceSolver->getPhi().transpose() << endl;
        //        cerr<<"dv = " << complianceSolver->getDv().transpose() << endl;
        //        cerr<<"lambda = " << complianceSolver->getLambda().transpose() << endl;
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
        Node::SPtr root = clearScene();
        SReal g=10;
        root->setGravity( Vec3(g,0,0) );

        // The solver
        complianceSolver = New<OdeSolver>();
        root->addObject( complianceSolver );
        complianceSolver->storeDynamicsSolution(true);
        linearSolver = New<LinearSolver>();
        root->addObject( linearSolver);
        complianceSolver->alpha.setValue(1.0);
        complianceSolver->beta.setValue(1.0);

        // ========  strings
        Node::SPtr  string1 = createCompliantString( root, Vec3(0,0,0), Vec3(1,0,0), n, 1.0*n, 0 );
        Node::SPtr  string2 = createCompliantString( root, Vec3(2,0,0), Vec3(3,0,0), n, 1.0*n, 0 );

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


        // Expected results
        expected.M = expected.P = DenseMatrix::Identity( 6*n, 6*n );
        for(unsigned i=0; i<3; i++)
            expected.P(i,i)=0; // fixed point
        expected.M = expected.P.transpose() * expected.M * expected.P;
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
        //        for( unsigned i=0; i<multimapping->getJs()->size(); i++ ){
        //            cerr<<"multimapping Jacobian " << i << ": " << endl << *(*multimapping->getJs())[i] << endl;
        //        }
        sofa::simulation::getSimulation()->animate(root.get(),1.0);

        // actual results
        //        cerr<<"M = " << endl << DenseMatrix(complianceSolver->M()) << endl;
        //        cerr<<"J = " << endl << DenseMatrix(complianceSolver->J()) << endl;
        //        cerr<<"C = " << endl << DenseMatrix(complianceSolver->C()) << endl;
        //        cerr<<"P = " << endl << DenseMatrix(complianceSolver->P()) << endl;
        //        cerr<<"f = " << endl << complianceSolver->getF().transpose() << endl;
        //        cerr<<"phi = " << complianceSolver->getPhi().transpose() << endl;
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
        Node::SPtr root = clearScene();
        SReal g=10;
        root->setGravity( Vec3(g,0,0) );

        // The solver
        complianceSolver = New<OdeSolver>();
        root->addObject( complianceSolver );
        complianceSolver->storeDynamicsSolution(true);
        linearSolver = New<LinearSolver>();
        root->addObject( linearSolver);
        complianceSolver->alpha.setValue(1.0);
        complianceSolver->beta.setValue(1.0);

        // ========= The rigid object
        simulation::Node::SPtr rigid = root->createChild("rigid");
        MechanicalObjectRigid3::SPtr rigidDOF = addNew<MechanicalObjectRigid3>(rigid);
        rigidDOF->resize(1);
        MechanicalObjectRigid3::WriteVecCoord x = rigidDOF->writePositions();
        x[0].getCenter() = Vec3(n,0,0);
        UniformMassRigid3::SPtr rigidMass = addNew<UniformMassRigid3>(rigid);

        // .========= Particle attached to the rigid object
        simulation::Node::SPtr particleOnRigid = rigid->createChild("particleOnRigid");
        MechanicalObject3::SPtr particleOnRigidDOF = addNew<MechanicalObject3>(particleOnRigid);
        particleOnRigidDOF->resize(1);
        RigidMappingRigid3d_to_3d::SPtr particleOnRigidMapping = addNew<RigidMappingRigid3d_to_3d>(particleOnRigid);
        particleOnRigidMapping->setModels(rigidDOF.get(),particleOnRigidDOF.get());

        // ========= The string
        simulation::Node::SPtr  string1 = createCompliantString( root, Vec3(0,0,0), Vec3(1,0,0), n, 1.0*n, 0. );

        // Fix the first particle of the string
        FixedConstraint3::SPtr fixed1 = New<FixedConstraint3>();
        string1->addObject( fixed1 );

        // ..======== Mixed subset containing the last particle of the string and the particle attached to the rigid object
        simulation::Node::SPtr pointPair = particleOnRigid->createChild("pointPair");
        string1->addChild(pointPair); // two parents: particleOnRigid and string1

        MechanicalObject3::SPtr pointPairDOF = addNew<MechanicalObject3>(pointPair);
        SubsetMultiMapping3_to_3::SPtr pointPairMapping = addNew<SubsetMultiMapping3_to_3>(pointPair);
        pointPairMapping->addInputModel(string1->mechanicalState);
        pointPairMapping->addInputModel(particleOnRigid->mechanicalState);
        pointPairMapping->addOutputModel(pointPair->mechanicalState);
        pointPairMapping->addPoint(string1->mechanicalState,n-1 );     // last particle
        pointPairMapping->addPoint(particleOnRigid->mechanicalState,0 );

        //  ...========  Distance between the particles in pointPair
        Node::SPtr extension = pointPair->createChild("extension");

        MechanicalObject1::SPtr extensionDOF = addNew<MechanicalObject1>(extension);

        EdgeSetTopologyContainer::SPtr extensionEdgeSet = addNew<EdgeSetTopologyContainer>(extension);
        extensionEdgeSet->addEdge(0,1);

        ExtensionMapping31::SPtr extensionMapping = addNew<ExtensionMapping31>(extension);
        extensionMapping->setModels(pointPairDOF.get(),extensionDOF.get());
        //        helper::WriteAccessor< Data< vector< Real > > > restLengths( extensionMapping->f_restLengths );
        //        restLengths.resize(1);
        //        restLengths[0] = 1.0;

        UniformCompliance1::SPtr extensionCompliance = addNew<UniformCompliance1>(extension);
        extensionCompliance->compliance.setValue(0);



        // ***** Expected results
        unsigned nM = 3*n+6;  // n particles + 1 rigid
        unsigned nC = n;      // n-1 in the string + 1 to connect the string to the rigid
        expected.M = expected.P = DenseMatrix::Identity( nM,nM );
        for(unsigned i=0; i<3; i++)
            expected.P(6+i,6+i)=0; // fixed point
        expected.M = expected.P.transpose() * expected.M * expected.P;
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

        // actual results
        //        cerr<<"M = " << endl << DenseMatrix(complianceSolver->M()) << endl;
        //        cerr<<"J = " << endl << DenseMatrix(complianceSolver->J()) << endl;
        //        cerr<<"C = " << endl << DenseMatrix(complianceSolver->C()) << endl;
        //        cerr<<"P = " << endl << DenseMatrix(complianceSolver->P()) << endl;
        //        cerr<<"f = " << endl << complianceSolver->getF().transpose() << endl;
        //        cerr<<"phi = " << complianceSolver->getPhi().transpose() << endl;
        //        cerr<<"actual dv = " << complianceSolver->getDv().transpose() << endl;
        //        cerr<<"actual lambda = " << complianceSolver->getLambda().transpose() << endl;
    }

    ///@}



};



} // sofa



