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

/* Francois Faure, 2013 */
#ifndef SOFA_STANDARDTEST_Mapping_test_H
#define SOFA_STANDARDTEST_Mapping_test_H

#include "Sofa_test.h"
#include <SofaComponentMain/init.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/simulation/common/VectorOperations.h>
#include <SofaBaseLinearSolver/FullVector.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <plugins/SceneCreator/SceneCreator.h>

namespace sofa {

using std::cout;
using std::endl;


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

  The magnitude of the small random changes applied in finite differences is between 0 and deltaMax*epsilon,
  and a failure is issued if the error is greater than errorMax*epsilon,
  where epsilon=std::numeric_limits<Real>::epsilon() is 1.19209e-07 for float and 2.22045e-16 for double.

  @author François Faure @date 2013
  */

template< class _Mapping>
struct Mapping_test: public Sofa_test<typename _Mapping::Real>
{
    typedef _Mapping Mapping;
    typedef typename Mapping::In In;
    typedef component::container::MechanicalObject<In> InDOFs;
    typedef typename InDOFs::Real  Real;
    typedef typename InDOFs::Deriv  InDeriv;
    typedef typename InDOFs::VecCoord  InVecCoord;
    typedef typename InDOFs::VecDeriv  InVecDeriv;
    typedef typename InDOFs::ReadVecCoord  ReadInVecCoord;
    typedef typename InDOFs::WriteVecCoord WriteInVecCoord;
    typedef typename InDOFs::ReadVecDeriv  ReadInVecDeriv;
    typedef typename InDOFs::WriteVecDeriv WriteInVecDeriv;
    typedef Data<InVecCoord> InDataVecCoord;
    typedef Data<InVecDeriv> InDataVecDeriv;

    typedef typename Mapping::Out Out;
    typedef component::container::MechanicalObject<Out> OutDOFs;
    typedef typename OutDOFs::Coord     OutCoord;
    typedef typename OutDOFs::Deriv     OutDeriv;
    typedef typename OutDOFs::VecCoord  OutVecCoord;
    typedef typename OutDOFs::VecDeriv  OutVecDeriv;
    typedef typename OutDOFs::ReadVecCoord  ReadOutVecCoord;
    typedef typename OutDOFs::WriteVecCoord WriteOutVecCoord;
    typedef typename OutDOFs::ReadVecDeriv  ReadOutVecDeriv;
    typedef typename OutDOFs::WriteVecDeriv WriteOutVecDeriv;
    typedef Data<OutVecCoord> OutDataVecCoord;
    typedef Data<OutVecDeriv> OutDataVecDeriv;

    typedef component::linearsolver::EigenSparseMatrix<In,Out> EigenSparseMatrix;


    core::Mapping<In,Out>* mapping; ///< the mapping to be tested
    typename InDOFs::SPtr  inDofs;  ///< mapping input
    typename OutDOFs::SPtr outDofs; ///< mapping output
    simulation::Node::SPtr root;         ///< Root of the scene graph, created by the constructor an re-used in the tests
    simulation::Simulation* simulation;  ///< created by the constructor an re-used in the tests
    Real deltaMax; ///< The maximum magnitude of the change of each scalar value of the small displacement is perturbation * numeric_limits<Real>::epsilon. This epsilon is 1.19209e-07 for float and 2.22045e-16 for double.
    Real errorMax;     ///< The test is successfull if the (infinite norm of the) difference is less than  maxError * numeric_limits<Real>::epsilon


    Mapping_test():deltaMax(1000),errorMax(10)
    {
        sofa::component::init();
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());

        /// Parent node
        root = simulation->createNewGraph("root");
        inDofs = modeling::addNew<InDOFs>(root);

        /// Child node
        simulation::Node::SPtr childNode = root->createChild("childNode");
        outDofs = modeling::addNew<OutDOFs>(childNode);
        mapping = modeling::addNew<Mapping>(root).get();
        mapping->setModels(inDofs.get(),outDofs.get());
    }

    Mapping_test(std::string fileName):deltaMax(1000),errorMax(100)
    {
        sofa::component::init();
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());

        /// Load the scene
        root = simulation->createNewGraph("root");
        root = sofa::core::objectmodel::SPtr_dynamic_cast<sofa::simulation::Node>( sofa::simulation::getSimulation()->load(fileName.c_str()));

        // InDofs
        inDofs = root->get<InDOFs>(root->SearchDown);

        // Get child nodes
        simulation::Node::SPtr patchNode = root->getChild("Patch");
        simulation::Node::SPtr elasticityNode = patchNode->getChild("Elasticity");

        // Add OutDofs
        outDofs = modeling::addNew<OutDOFs>(elasticityNode);

        // Add mapping to the scene
        mapping = modeling::addNew<Mapping>(elasticityNode).get();
        mapping->setModels(inDofs.get(),outDofs.get());
        
    }


    /** Returns OutCoord substraction a-b (should return a OutDeriv, but???)
      */
    virtual OutDeriv difference( const OutCoord& a, const OutCoord& b )
    {
        return Out::coordDifference(a,b);
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
                          const InVecCoord parentNew,
                          const OutVecCoord expectedChildNew)
    {
        typedef component::linearsolver::EigenSparseMatrix<In,Out> EigenSparseMatrix;
        core::MechanicalParams mparams;
        mparams.setKFactor(1.0);
        mparams.setSymmetricMatrix(false);

        inDofs->resize(parentInit.size());
        WriteInVecCoord xin = inDofs->writePositions();
        copyToData(xin,parentInit); // xin = parentInit

        outDofs->resize(childInit.size());
        WriteOutVecCoord xout = outDofs->writePositions();
        copyToData(xout,childInit);

        /// Init based on parentInit
        sofa::simulation::getSimulation()->init(root.get());

        /// Updated to parentNew
        copyToData(xin,parentNew);
        mapping->apply(&mparams, core::VecCoordId::position(), core::VecCoordId::position());
        mapping->applyJ(&mparams, core::VecDerivId::velocity(), core::VecDerivId::velocity());

        /// test apply: check if the child positions are the expected ones
        bool succeed=true;
        for( unsigned i=0; i<xout.size(); i++ )
        {
            if( !this->isSmall( difference(xout[i],expectedChildNew[i]).norm(), errorMax ) ) {
                ADD_FAILURE() << "Position of mapped particle " << i << " is wrong: \n" << xout[i] <<"\nexpected: \n" << expectedChildNew[i];
                succeed = false;
            }
        }

        /// test applyJ and everything related to Jacobians
        const unsigned Np=inDofs->getSize(), Nc=outDofs->getSize();

        InVecCoord xp(Np),xp1(Np);
        InVecDeriv vp(Np),fp(Np),dfp(Np),fp2(Np);
        OutVecCoord xc(Nc),xc1(Nc);
        OutVecDeriv vc(Nc),fc(Nc);

        // get position data
        copyFromData( xp,inDofs->readPositions() );
        copyFromData( xc,  outDofs->readPositions() ); // positions and have already been propagated
        //          cout<<"parent positions xp = "<< xp << endl;
        //          cout<<"child  positions xc = "<< xc << endl;

        // set random child forces and propagate them to the parent
        for( unsigned i=0; i<Nc; i++ ){
            fc[i] = Out::randomDeriv( 1.0 , BaseSofa_test::seed  );
        }
        fp2.fill( InDeriv() );
        WriteInVecDeriv fin = inDofs->writeForces();
        copyToData( fin, fp2 );  // reset parent forces before accumulating child forces
        //        cout<<"random child forces  fc = "<<fc<<endl;
        WriteOutVecDeriv fout = outDofs->writeForces();
        copyToData( fout, fc );
        mapping->applyJT( &mparams, core::VecDerivId::force(), core::VecDerivId::force() );
        copyFromData( fp, inDofs->readForces() );
        //          cout<<"parent forces fp = "<<fp<<endl;

        // set small parent velocities and use them to update the child
        for( unsigned i=0; i<Np; i++ ){
            vp[i] = In::randomDeriv( this->epsilon() * deltaMax , BaseSofa_test::seed);
        }
//        cout<<"parent velocities vp = " << vp << endl;
        for( unsigned i=0; i<Np; i++ ){             // and small displacements
            xp1[i] = xp[i] + vp[i];
        }
        //          cout<<"new parent positions xp1 = " << xp1 << endl;

        // propagate small velocity
        WriteInVecDeriv vin = inDofs->writeVelocities();
        copyToData( vin, vp );
        mapping->applyJ( &mparams, core::VecDerivId::velocity(), core::VecDerivId::velocity() );
        WriteOutVecDeriv vout = outDofs->writeVelocities();
        copyFromData( vc, vout);
        //          cout<<"child velocity vc = " << vc << endl;


        // apply geometric stiffness
        inDofs->vRealloc( &mparams, core::VecDerivId::dx() ); // dx is not allocated by default
        WriteInVecDeriv dxin = inDofs->writeDx();
        copyToData( dxin, vp );
        dfp.fill( InDeriv() );
        copyToData( fin, dfp );
        mapping->applyDJT( &mparams, core::VecDerivId::force(), core::VecDerivId::force() );
        copyFromData( dfp, inDofs->readForces() ); // fp + df due to geometric stiffness
        //        cout<<"dfp = " << dfp << endl;

        // Jacobian will be obsolete after applying new positions
        if( mapping->getClassName() != "RigidRigidMapping" ) // Because RigidRigidMapping: getJs not implemented
        {
            EigenSparseMatrix* J = this->getMatrix(mapping->getJs());
            //        cout<<"J = "<< endl << *J << endl;
            OutVecDeriv Jv(Nc);
            J->mult(Jv,vp);

            // ================ test applyJT()
            InVecDeriv jfc( (long)Np,InDeriv());
            J->addMultTranspose(jfc,fc);
            if( this->vectorMaxDiff(jfc,fp)>this->epsilon()*errorMax ){
                succeed = false;
                ADD_FAILURE() << "applyJT test failed"<<endl<<"jfc = " << jfc << endl<<" fp = " << fp << endl;
            }
            // ================ test getJs()
            // check that J.vp = vc
            if( this->vectorMaxDiff(Jv,vc)>this->epsilon()*errorMax ){
                succeed = false;
                cout<<"vp = " << vp << endl;
                cout<<"Jvp = " << Jv << endl;
                cout<<"vc  = " << vc << endl;
                ADD_FAILURE() << "getJs() test failed"<<endl<<"vp = " << vp << endl<<"Jvp = " << Jv << endl <<"vc  = " << vc << endl;
            }
        }

        // compute parent forces from pre-treated child forces (in most cases, the pre-treatment does nothing)
        // the pre-treatement can be useful to be able to compute 2 comparable results of applyJT with a small displacement to test applyDJT
        fp.fill( InDeriv() );
        copyToData( fin, fp );  // reset parent forces before accumulating child forces
        copyToData( fout, preTreatment(fc) );
        mapping->applyJT( &mparams, core::VecDerivId::force(), core::VecDerivId::force() );
        copyFromData( fp, inDofs->readForces() );



        ///////////////////////////////


        // propagate small displacement
        WriteInVecCoord pin (inDofs->writePositions());
        copyToData( pin, xp1 );
        //            cout<<"new parent positions xp1 = " << xp1 << endl;
        mapping->apply ( &mparams, core::VecCoordId::position(), core::VecCoordId::position() );
        WriteOutVecCoord pout = outDofs->writePositions();
        copyFromData( xc1, pout );
        //            cout<<"old child positions xc = " << xc << endl;
        //            cout<<"new child positions xc1 = " << xc1 << endl;

        // ================ test applyJ: compute the difference between propagated displacements and velocities
        OutVecDeriv dxc(Nc);
        for(unsigned i=0; i<Nc; i++ ){
            dxc[i] = difference( xc1[i], xc[i] );
        }

        if( this->vectorMaxDiff(dxc,vc)>this->epsilon()*errorMax ){
            succeed = false;
            ADD_FAILURE() << "applyJ test failed: the difference between child position change and child velocity (dt=1) should be less than  " << this->epsilon()*errorMax  << endl
                          << "position change = " << dxc << endl
                          << "velocity        = " << vc << endl;
        }



        // update parent force based on the same child forces
        fp2.fill( InDeriv() );
        copyToData( fin, fp2 );  // reset parent forces before accumulating child forces
        copyToData( fout, preTreatment(fc) );
        mapping->applyJT( &mparams, core::VecDerivId::force(), core::VecDerivId::force() );
        copyFromData( fp2, inDofs->readForces() );
        //        cout<<"updated parent forces fp2 = "<< fp2 << endl;
        InVecDeriv fp12(Np);
        for(unsigned i=0; i<Np; i++){
            fp12[i] = fp2[i]-fp[i];       // fp2 - fp
        }
        //        cout<<"fp2 - fp = " << fp12 << endl;



        // ================ test applyDJT()
        if( this->vectorMaxDiff(dfp,fp12)>this->epsilon()*errorMax ){
            succeed = false;
            ADD_FAILURE() << "applyDJT test failed" << endl <<
                             "dfp    = " << dfp << endl <<
                             "fp2-fp = " << fp12 << endl;
        }
        //}

        if(!succeed)
        { ADD_FAILURE() << "Failed Seed number = " << BaseSofa_test::seed << std::endl;}
        return succeed;
    }

    virtual ~Mapping_test()
    {
        if (root!=NULL)
            sofa::simulation::getSimulation()->unload(root);
    }

    /// Get one EigenSparseMatrix out of a list. Error if not one single matrix in the list.
    static EigenSparseMatrix* getMatrix(const vector<sofa::defaulttype::BaseMatrix*>* matrices)
    {
        if( matrices->size() != 1 ){
            ADD_FAILURE()<< "Matrix list should have size == 1 in simple mappings";
        }
        EigenSparseMatrix* ei = dynamic_cast<EigenSparseMatrix*>((*matrices)[0] );
        if( ei == NULL ){
            ADD_FAILURE() << "getJs returns a matrix of non-EigenSparseMatrix type";
        }
        return ei;
    }



};

} // namespace sofa


#endif
