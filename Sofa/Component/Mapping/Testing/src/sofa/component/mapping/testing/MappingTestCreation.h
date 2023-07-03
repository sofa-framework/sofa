/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest;

#include <sofa/testing/NumericTest.h>
using sofa::testing::NumericTest;

#include <sofa/simulation/VectorOperations.h>

#include <sofa/linearalgebra/FullVector.h>
#include <sofa/linearalgebra/EigenSparseMatrix.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/core/Mapping.h>

#include <sofa/helper/logging/Messaging.h>

namespace sofa::mapping_test {


/** @brief Base class for the Mapping tests, with helpers to automatically test applyJ, applyJT, applyDJT and getJs using finite differences.

  Specific test cases can be created using a derived class instantiated on the mapping class to test,
  and calling function runTest( const InVecCoord& parentInit,
                  const OutVecCoord& childInit,
                  const InVecCoord parentNew,
                  const OutVecCoord expectedChildNew);


  This function compares the actual output positions with the expected ones, then automatically tests the methods related to
  the Jacobian using finite differences.
  - A small change of the input positions dxIn is randomly chosen and added to the current position. The same is set as velocity.
  - mapping->apply is called, and the difference dXout between the new output positions and the previous positions is computed
  - to validate mapping->applyJ, dXin is converted to input velocity vIn and mapping->applyJ is called. dXout and the output velocity vOut must be the same (up to linear approximations errors, thus we apply a very small change of position).
  - to validate mapping->getJs, we use it to get the Jacobian, then we check that J.vIn = vOut
  - to validate mapping->applyJT, we apply it after setting the child force fc=vOut, then we check that parent force fp = J^T.fc
  - to validate mapping->applyDJT, we set the child force, and we compare the parent force before and after a small displacement

  The magnitude of the small random changes applied in finite differences is between deltaRange.first*epsilon and deltaRange.second*epsilon,
  and a failure is issued if the error is greater than errorMax*epsilon,
  where epsilon=std::numeric_limits<Real>::epsilon() is 1.19209e-07 for float and 2.22045e-16 for double.

  @author François Faure @date 2013
  */

template< class _Mapping>
struct Mapping_test: public BaseSimulationTest, NumericTest<typename _Mapping::In::Real>
{
    typedef _Mapping Mapping;
    typedef typename Mapping::In In;
    typedef component::statecontainer::MechanicalObject<In> InDOFs;
    typedef typename InDOFs::Real  Real;
    typedef typename InDOFs::Coord  InCoord;
    typedef typename InDOFs::Deriv  InDeriv;
    typedef typename InDOFs::VecCoord  InVecCoord;
    typedef typename InDOFs::VecDeriv  InVecDeriv;
    typedef typename InDOFs::ReadVecCoord  ReadInVecCoord;
    typedef typename InDOFs::WriteVecCoord WriteInVecCoord;
    typedef typename InDOFs::ReadVecDeriv  ReadInVecDeriv;
    typedef typename InDOFs::WriteVecDeriv WriteInVecDeriv;
    typedef typename InDOFs::MatrixDeriv  InMatrixDeriv;
    typedef Data<InVecCoord> InDataVecCoord;
    typedef Data<InVecDeriv> InDataVecDeriv;
    typedef Data<InMatrixDeriv> InDataMatrixDeriv;

    typedef typename Mapping::Out Out;
    typedef component::statecontainer::MechanicalObject<Out> OutDOFs;
    typedef typename OutDOFs::Coord     OutCoord;
    typedef typename OutDOFs::Deriv     OutDeriv;
    typedef typename OutDOFs::VecCoord  OutVecCoord;
    typedef typename OutDOFs::VecDeriv  OutVecDeriv;
    typedef typename OutDOFs::ReadVecCoord  ReadOutVecCoord;
    typedef typename OutDOFs::WriteVecCoord WriteOutVecCoord;
    typedef typename OutDOFs::ReadVecDeriv  ReadOutVecDeriv;
    typedef typename OutDOFs::WriteVecDeriv WriteOutVecDeriv;
    typedef typename OutDOFs::MatrixDeriv  OutMatrixDeriv;
    typedef Data<OutVecCoord> OutDataVecCoord;
    typedef Data<OutVecDeriv> OutDataVecDeriv;
    typedef Data<OutMatrixDeriv> OutDataMatrixDeriv;

    typedef linearalgebra::EigenSparseMatrix<In,Out> EigenSparseMatrix;

    core::Mapping<In,Out>* mapping; ///< the mapping to be tested
    typename InDOFs::SPtr  inDofs;  ///< mapping input
    typename OutDOFs::SPtr outDofs; ///< mapping output
    simulation::Node::SPtr root;         ///< Root of the scene graph, created by the constructor an re-used in the tests
    simulation::Simulation* simulation;  ///< created by the constructor an re-used in the tests
    std::pair<Real,Real> deltaRange; ///< The minimum and maximum magnitudes of the change of each scalar value of the small displacement is perturbation * numeric_limits<Real>::epsilon. This epsilon is 1.19209e-07 for float and 2.22045e-16 for double.
    Real errorMax;     ///< The test is successfull if the (infinite norm of the) difference is less than  errorMax * numeric_limits<Real>::epsilon
    Real errorFactorDJ;     ///< The test for geometric stiffness is successfull if the (infinite norm of the) difference is less than  errorFactorDJ * errorMax * numeric_limits<Real>::epsilon


    static const unsigned char TEST_getJs = 1; ///< testing getJs used in assembly API
    static const unsigned char TEST_getK = 2; ///< testing getK used in assembly API
    static const unsigned char TEST_applyJT_matrix = 4; ///< testing applyJT on matrices
    static const unsigned char TEST_applyDJT = 8; ///< testing applyDJT 
    static const unsigned char TEST_ASSEMBLY_API = TEST_getJs | TEST_getK; ///< testing functions used in assembly API getJS getKS
    static const unsigned char TEST_GEOMETRIC_STIFFNESS = TEST_applyDJT | TEST_getK; ///< testing functions used in assembly API getJS getKS
    unsigned char flags; ///< testing options. (all by default). To be used with precaution. Please implement the missing API in the mapping rather than not testing it.


    Mapping_test():deltaRange(1,1000),errorMax(10),errorFactorDJ(1),flags(TEST_ASSEMBLY_API | TEST_GEOMETRIC_STIFFNESS)
    {
        simulation = sofa::simulation::getSimulation();

        /// Parent node
        root = simulation->createNewGraph("root");

        inDofs = core::objectmodel::New<InDOFs>();
        root->addObject(inDofs);

        /// Child node
        simulation::Node::SPtr childNode = root->createChild("childNode");
        outDofs = core::objectmodel::New<OutDOFs>();
        childNode->addObject(outDofs);
        auto mappingSptr = core::objectmodel::New<Mapping>();
        mapping = mappingSptr.get();
        childNode->addObject(mapping);

        mapping->setModels(inDofs.get(),outDofs.get());
    }

    Mapping_test(std::string fileName)
        : simulation(sofa::simulation::getSimulation()),
          deltaRange(1, 1000),
          errorMax(100),
          errorFactorDJ(1),
          flags(TEST_ASSEMBLY_API|TEST_GEOMETRIC_STIFFNESS)
    {
        assert(simulation);

        /// Load the scene
        root = simulation->createNewGraph("root");
        root = sofa::simulation::node::load(fileName.c_str(), false);

        // InDofs
        inDofs = root->get<InDOFs>(root->SearchDown);

        // Get child nodes
        const simulation::Node::SPtr patchNode = root->getChild("Patch");
        simulation::Node::SPtr elasticityNode = patchNode->getChild("Elasticity");

        // Add OutDofs
        outDofs = core::objectmodel::New<OutDOFs>();
        elasticityNode->addObject(outDofs);

        // Add mapping to the scene
        auto mappingSptr = core::objectmodel::New<Mapping>();
        mapping = mappingSptr.get();
        elasticityNode->addObject(mapping);
        mapping->setModels(inDofs.get(),outDofs.get());
        
    }


    /** Returns OutCoord substraction a-b */
    virtual OutDeriv difference( const OutCoord& a, const OutCoord& b )
    {
        return Out::coordDifference(a,b);
    }

    virtual OutVecDeriv difference( const OutVecDeriv& a, const OutVecDeriv& b )
    {
        if( a.size()!=b.size() ){
            ADD_FAILURE() << "OutVecDeriv have different sizes";
            return OutVecDeriv();
        }

        OutVecDeriv c(a.size());
        for (size_t i=0; i<a.size() ; ++i)
        {
            c[i] = a[i]-b[i];
        }
        return c;
    }

    /** Possible child force pre-treatment, does nothing by default
      */
    virtual OutVecDeriv preTreatment( const OutVecDeriv& f ) { return f; }


    /** Test the mapping using the given values and small changes.
     * Return true in case of success, if all errors are below maxError*epsilon.
     * The mapping is initialized using the two first parameters,
     * then a new parent position is applied,
     * and the new child position is compared with the expected one.
     * Additionally, the Jacobian-related methods are tested using finite differences.
     *
     * The initialization values can used when the mapping is an embedding, e.g. to attach a mesh to a rigid object we compute the local coordinates of the vertices based on their world coordinates and the frame coordinates.
     * In other cases, such as mapping from pairs of points to distances, no initialization values are necessary, an one can use the same values as for testing, i.e. runTest( xp, expected_xc, xp, expected_xc).
     *
     *\param parentInit initial parent position
     *\param childInit initial child position
     *\param parentNew new parent position
     *\param expectedChildNew expected position of the child corresponding to the new parent position
     */
    virtual bool runTest( const InVecCoord& parentInit,
                          const OutVecCoord& childInit,
                          const InVecCoord& parentNew,
                          const OutVecCoord& expectedChildNew)
    {
        if( deltaRange.second / errorMax <= sofa::testing::g_minDeltaErrorRatio )
            ADD_FAILURE() << "The comparison threshold is too large for the finite difference delta";

        if( !(flags & TEST_getJs) )          msg_warning("MappingTest") << "getJs is not tested";
        if( !(flags & TEST_getK) )           msg_warning("MappingTest") << "getK is not tested";
        if( !(flags & TEST_applyJT_matrix) ) msg_warning("MappingTest") << "applyJT on matrices is not tested";
        if( !(flags & TEST_applyDJT) )       msg_warning("MappingTest") << "applyDJT is not tested";


        const Real errorThreshold = this->epsilon()*errorMax;

        typedef linearalgebra::EigenSparseMatrix<In,Out> EigenSparseMatrix;
        core::MechanicalParams mparams;
        mparams.setKFactor(1.0);
        mparams.setSupportOnlySymmetricMatrix(false);
        inDofs->resize(parentInit.size());
        WriteInVecCoord xin = inDofs->writePositions();
        sofa::testing::copyToData(xin,parentInit); // xin = parentInit

        outDofs->resize(childInit.size());
        WriteOutVecCoord xout = outDofs->writePositions();
        sofa::testing::copyToData(xout,childInit);

        /// Init based on parentInit
        sofa::simulation::node::initRoot(root.get());

        /// Updated to parentNew
        sofa::testing::copyToData(xin,parentNew);
        mapping->apply(&mparams, core::VecCoordId::position(), core::VecCoordId::position());
        mapping->applyJ(&mparams, core::VecDerivId::velocity(), core::VecDerivId::velocity());

        /// test apply: check if the child positions are the expected ones
        bool succeed=true;

        if (expectedChildNew.size() != xout.size()) {
            ADD_FAILURE() << "Size of output dofs is wrong: " << xout.size() << " expected: " << expectedChildNew.size();
            succeed = false;
        }
        for( unsigned i=0; i<xout.size(); i++ )
        {
            if( !this->isSmall( difference(xout[i],expectedChildNew[i]).norm(), errorMax ) ) {
                ADD_FAILURE() << "Position of mapped particle " << i << " is wrong: \n" << xout[i] <<"\nexpected: \n" << expectedChildNew[i]
                              <<  "\ndifference should be less than " << errorThreshold << " (" << difference(xout[i],expectedChildNew[i]).norm() << ")" << std::endl;
                succeed = false;
            }
        }

        /// test applyJ and everything related to Jacobians
        size_t Np = inDofs->getSize(), Nc=outDofs->getSize();

        InVecCoord xp(Np),xp1(Np);
        InVecDeriv vp(Np),fp(Np),dfp(Np),fp2(Np);
        OutVecCoord xc(Nc),xc1(Nc);
        OutVecDeriv vc(Nc),fc(Nc);

        // get position data
        sofa::testing::copyFromData( xp, inDofs->readPositions() );
        sofa::testing::copyFromData( xc, outDofs->readPositions() ); // positions and have already been propagated

        // set random child forces and propagate them to the parent
        for( unsigned i=0; i<Nc; i++ ){
            fc[i] = Out::randomDeriv( 0.1, 1.0 );
        }
        fp2.fill( InDeriv() );
        WriteInVecDeriv fin = inDofs->writeForces();
        sofa::testing::copyToData( fin, fp2 );  // reset parent forces before accumulating child forces

        WriteOutVecDeriv fout = outDofs->writeForces();
        sofa::testing::copyToData( fout, fc );
        mapping->applyJT( &mparams, core::VecDerivId::force(), core::VecDerivId::force() );
        sofa::testing::copyFromData( fp, inDofs->readForces() );

        // set small parent velocities and use them to update the child
        for( unsigned i=0; i<Np; i++ ){
            vp[i] = In::randomDeriv(this->epsilon() * deltaRange.first, this->epsilon() * deltaRange.second );
        }

        for( unsigned i=0; i<Np; i++ ){             // and small displacements
            xp1[i] = xp[i] + vp[i];
        }

        // propagate small velocity
        WriteInVecDeriv vin = inDofs->writeVelocities();
        sofa::testing::copyToData( vin, vp );
        mapping->applyJ( &mparams, core::VecDerivId::velocity(), core::VecDerivId::velocity() );
        ReadOutVecDeriv vout = outDofs->readVelocities();
        sofa::testing::copyFromData( vc, vout);

        // apply geometric stiffness
        inDofs->vRealloc( &mparams, core::VecDerivId::dx() ); // dx is not allocated by default
        WriteInVecDeriv dxin = inDofs->writeDx();
        sofa::testing::copyToData( dxin, vp );
        dfp.fill( InDeriv() );
        sofa::testing::copyToData( fin, dfp );
        mapping->updateK( &mparams, core::ConstVecDerivId::force() ); // updating stiffness matrix for the current state and force
        mapping->applyDJT( &mparams, core::VecDerivId::force(), core::VecDerivId::force() );
        sofa::testing::copyFromData( dfp, inDofs->readForces() ); // fp + df due to geometric stiffness

        // Jacobian will be obsolete after applying new positions
        if( flags & TEST_getJs )
        {
            EigenSparseMatrix* J = this->getMatrix<EigenSparseMatrix>(mapping->getJs());
            //        cout<<"J = "<< endl << *J << endl;
            OutVecDeriv Jv(Nc);
            J->mult(Jv,vp);

            // ================ test applyJT()
            InVecDeriv jfc( (long)Np,InDeriv());
            J->addMultTranspose(jfc,fc);
            if( this->vectorMaxDiff(jfc,fp)>errorThreshold ){
                succeed = false;
                ADD_FAILURE() << "applyJT test failed, difference should be less than " << errorThreshold << " (" << this->vectorMaxDiff(jfc,fp) << ")" << std::endl
                              << "jfc = " << jfc << std::endl<<" fp = " << fp << std::endl;
            }
            // ================ test getJs()
            // check that J.vp = vc
            if(this->vectorMaxDiff(Jv,vc)>errorThreshold ){
                succeed = false;
                std::cout<<"vp = " << vp << std::endl;
                std::cout<<"Jvp = " << Jv << std::endl;
                std::cout<<"vc  = " << vc << std::endl;
                ADD_FAILURE() << "getJs() test failed"<<std::endl<<"vp = " << vp << std::endl<<"Jvp = " << Jv << std::endl <<"vc  = " << vc << std::endl;
            }
        }

        if( flags & TEST_applyJT_matrix )
        {
            // TODO test applyJT on matrices
            // basic idea build a out random matrix  e.g. with helper::drand(100.0)
            // call applyJT on both this matrice and on all its lines (oe cols ?) one by one
            // then compare results

//            OutMatrixDeriv outMatrices(  ); // how to build that, what size?
//            /*WriteInMatrixDeriv min = */inDofs->write( MatrixDerivId::constraintJacobian() );
//            WriteOutMatrixDeriv mout = outDofs->write( MatrixDerivId::constraintJacobian() );
//            copyToData(mout,outMatrices);

//            mapping->applyJt(  ConstraintParams*, MatrixDerivId::constraintJacobian(), MatrixDerivId::constraintJacobian() );


        }

        // compute parent forces from pre-treated child forces (in most cases, the pre-treatment does nothing)
        // the pre-treatement can be useful to be able to compute 2 comparable results of applyJT with a small displacement to test applyDJT
        fp.fill( InDeriv() );
        sofa::testing::copyToData( fin, fp );  // reset parent forces before accumulating child forces
        sofa::testing::copyToData( fout, preTreatment(fc) );
        mapping->applyJT( &mparams, core::VecDerivId::force(), core::VecDerivId::force() );
        sofa::testing::copyFromData( fp, inDofs->readForces() );



        ///////////////////////////////


        // propagate small displacement
        WriteInVecCoord pin (inDofs->writePositions());
        sofa::testing::copyToData( pin, xp1 );

        mapping->apply ( &mparams, core::VecCoordId::position(), core::VecCoordId::position() );
        ReadOutVecCoord pout = outDofs->readPositions();
        sofa::testing::copyFromData( xc1, pout );

        // ================ test applyJ: compute the difference between propagated displacements and velocities
        OutVecDeriv dxc(Nc);
        for(unsigned i=0; i<Nc; i++ ){
            dxc[i] = difference( xc1[i], xc[i] );
        }

        if(this->vectorMaxAbs(difference(dxc,vc))>errorThreshold ){
            succeed = false;
            ADD_FAILURE() << "applyJ test failed: the difference between child position change and child velocity (dt=1) "<< this->vectorMaxAbs(difference(dxc,vc))<<" should be less than  " << errorThreshold << std::endl
                          << "position change = " << dxc << std::endl
                          << "velocity        = " << vc << std::endl;
        }



        // update parent force based on the same child forces
        fp2.fill( InDeriv() );
        sofa::testing::copyToData( fin, fp2 );  // reset parent forces before accumulating child forces
        sofa::testing::copyToData( fout, preTreatment(fc) );
        mapping->applyJT( &mparams, core::VecDerivId::force(), core::VecDerivId::force() );
        sofa::testing::copyFromData( fp2, inDofs->readForces() );

        InVecDeriv fp12(Np);
        for(unsigned i=0; i<Np; i++){
            fp12[i] = fp2[i]-fp[i];       // fp2 - fp
        }


        // ================ test applyDJT()
        if( flags & TEST_applyDJT )
        {
            if(this->vectorMaxDiff(dfp,fp12)>errorThreshold*errorFactorDJ ){
                succeed = false;
                ADD_FAILURE() << "applyDJT test failed" << std::endl
                    << "dfp    = " << dfp << std::endl
                    << "fp2-fp = " << fp12 << std::endl;
            }
        }


        // ================ test getK()
        if( flags & TEST_getK )
        {
            InVecDeriv Kv(Np);

            const linearalgebra::BaseMatrix* bk = mapping->getK();

            // K can be null or empty for linear mappings
            // still performing the test with a null Kv vector to check if the mapping is really linear

            if( bk != nullptr ){

                typedef linearalgebra::EigenSparseMatrix<In,In> EigenSparseKMatrix;
                const EigenSparseKMatrix* K = dynamic_cast<const EigenSparseKMatrix*>(bk);
                if( K == nullptr ){
                    succeed = false;
                    ADD_FAILURE() << "getK returns a matrix of non-EigenSparseMatrix type";
                    // TODO perform a slow conversion with a big warning rather than a failure?
                }

                if( K->compressedMatrix.nonZeros() ) K->mult(Kv,vp);
            }

            // check that K.vp = dfp
            if(this->vectorMaxDiff(Kv,fp12)>errorThreshold*errorFactorDJ ){
                succeed = false;
                ADD_FAILURE() << "K test failed, difference should be less than " << errorThreshold*errorFactorDJ  << std::endl
                              << "Kv    = " << Kv << std::endl
                              << "dfp = " << fp12 << std::endl;
            }
        }

        if(!succeed)
        { ADD_FAILURE() << "Failed Seed number = " << this->seed << std::endl;}
        return succeed;
    }


    /** Test the mapping using the given values and small changes.
     * Return true in case of success, if all errors are below maxError*epsilon.
     * The mapping is initialized using the first parameter,
     * @warning this version supposes the mapping initialization does not depends on child positions
     * otherwise, use the runTest functions with 4 parameters
     * the child position is computed from parent position and compared with the expected one.
     * Additionally, the Jacobian-related methods are tested using finite differences.
     *
     *\param parent parent position
     *\param expectedChild expected position of the child corresponding to the parent position
     */
    virtual bool runTest( const InVecCoord& parent,
                          const OutVecCoord expectedChild)
    {
        OutVecCoord childInit( expectedChild.size() ); // start with null child
        return runTest( parent, childInit, parent, expectedChild );
    }

    ~Mapping_test() override
    {
        if (root!=nullptr)
            sofa::simulation::node::unload(root);
    }

protected:

    /// Get one EigenSparseMatrix out of a list. Error if not one single matrix in the list.
    template<class EigenSparseMatrixType>
    static EigenSparseMatrixType* getMatrix(const type::vector<sofa::linearalgebra::BaseMatrix*>* matrices)
    {
        if( !matrices ){
            ADD_FAILURE()<< "Matrix list is nullptr (API for assembly is not implemented)";
        }
        if( matrices->size() != 1 ){
            ADD_FAILURE()<< "Matrix list should have size == 1 in simple mappings";
        }
        EigenSparseMatrixType* ei = dynamic_cast<EigenSparseMatrixType*>((*matrices)[0] );
        if( ei == nullptr ){
            ADD_FAILURE() << "getJs returns a matrix of non-EigenSparseMatrix type";
            // TODO perform a slow conversion with a big warning rather than a failure?
        }
        return ei;
    }



};

} // namespace sofa::mapping_test
