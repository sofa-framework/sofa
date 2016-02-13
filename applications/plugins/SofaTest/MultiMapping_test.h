/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

/* Francois Faure, 2013 */
#ifndef SOFA_STANDARDTEST_MultiMapping_test_H
#define SOFA_STANDARDTEST_MultiMapping_test_H

#include <sstream>

#include "Sofa_test.h"
#include <sofa/core/MechanicalParams.h>
#include <sofa/simulation/common/VectorOperations.h>
#include <SofaBaseLinearSolver/FullVector.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <SceneCreator/SceneCreator.h>
#include <sofa/helper/vector.h>
#include <sofa/core/MultiMapping.h>

namespace sofa {

typedef std::size_t Index;


/** @brief Base class for the MultiMapping tests, directly adapted from Mapping_test.
 * @sa Mapping_test

  @author François Faure @date 2014
  */

template< class _MultiMapping>
struct MultiMapping_test : public Sofa_test<typename _MultiMapping::Real>
{
    typedef _MultiMapping Mapping;
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


    core::MultiMapping<In,Out>* mapping; ///< the mapping to be tested
    helper::vector<InDOFs*>  inDofs;  ///< mapping input
    OutDOFs* outDofs; ///< mapping output
    simulation::Node* root;         ///< Root of the scene graph, created by the constructor an re-used in the tests
    simulation::Node::SPtr child; ///< Child node, created by setupScene
    helper::vector<simulation::Node::SPtr> parents; ///< Parent nodes, created by setupScene
    simulation::Simulation* simulation;  ///< created by the constructor an re-used in the tests
    std::pair<Real,Real> deltaRange; ///< The minimum and maximum magnitudes of the change of each scalar value of the small displacement is deltaRange * numeric_limits<Real>::epsilon. This epsilon is 1.19209e-07 for float and 2.22045e-16 for double.
    Real errorMax;     ///< The test is successfull if the (infinite norm of the) difference is less than  maxError * numeric_limits<Real>::epsilon


    MultiMapping_test():deltaRange(1,1000),errorMax(10)
    {
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());

    }

    /** Create scene with given number of parent states. Currently, only one child state is handled.
     * All the parents are set as child of the root node, while the child is in a child node.
    */
    void setupScene(int numParents)
    {
        root = simulation->createNewGraph("root").get();

        /// Child node
        child = root->createChild("childNode");
        outDofs = modeling::addNew<OutDOFs>(child).get();
        mapping = modeling::addNew<Mapping>(child).get();
        mapping->addOutputModel(outDofs);

        /// Parent states, added to specific parentNode{i} nodes. This is not a simulable scene.
        for( int i=0; i<numParents; i++ )
        {
            std::stringstream ss;
            ss << "parentNode" << i;
            parents.push_back(root->createChild(ss.str()));
            typename InDOFs::SPtr inDof = modeling::addNew<InDOFs>(parents[i]);
            mapping->addInputModel( inDof.get() );
            inDofs.push_back(inDof.get());
        }

    }



    /** Returns OutCoord substraction a-b (should return a OutDeriv, but???)
      */
    OutDeriv difference( const OutCoord& c1, const OutCoord& c2 )
    {
        return Out::coordDifference(c1,c2);
    }


    /** Test the mapping using the given values and small changes.
     * Return true in case of success, if all errors are below maxError*epsilon.
     * The parent position is applied,
     * the resulting child position is compared with the expected one.
     * Additionally, the Jacobian-related methods are tested using finite differences.
     *
     * The parent coordinates are transfered in the parent states, then the scene is initialized, then various mapping functions are applied.
     * The parent states are resized based on the size of the parentCoords vectors. The child state is not resized. Its should be already sized,
     * or its size set automatically during initialization.
     *
     *\param parentCoords Parent positions (one InVecCoord per parent)
     *\param expectedChildCoords expected position of the child corresponding to the parent positions
     */
    bool runTest( const helper::vector<InVecCoord>& parentCoords,
                  const OutVecCoord& expectedChildCoords)
    {
        if( deltaRange.second / errorMax <= g_minDeltaErrorRatio )
            ADD_FAILURE() << "The comparison threshold is too large for the finite difference delta";

        typedef component::linearsolver::EigenSparseMatrix<In,Out> EigenSparseMatrix;
        core::MechanicalParams mparams;
        mparams.setKFactor(1.0);
        mparams.setSymmetricMatrix(false);

        // transfer the parent values in the parent states
        for( size_t i=0; i<parentCoords.size(); i++ )
        {
            this->inDofs[i]->resize(parentCoords[i].size());
            WriteInVecCoord xin = inDofs[i]->writePositions();
            copyToData(xin,parentCoords[i]); // xin = parentNew[i]
        }

        /// Init
        sofa::simulation::getSimulation()->init(root);

        /// apply the mapping
        mapping->apply(&mparams, core::VecCoordId::position(), core::VecCoordId::position());
        mapping->applyJ(&mparams, core::VecDerivId::velocity(), core::VecDerivId::velocity());

        /// test apply: check if the child positions are the expected ones
        bool succeed=true;
        ReadOutVecCoord xout = outDofs->readPositions();
        if (expectedChildCoords.size() != xout.size()) {
            ADD_FAILURE() << "Size of output dofs is wrong: " << xout.size() << " expected: " << expectedChildCoords.size();
            succeed = false;
        }

        for( Index i=0; i<xout.size(); i++ )
        {
            if( !this->isSmall( difference(xout[i],expectedChildCoords[i]).norm(), errorMax ) ) {
                ADD_FAILURE() << "Position of mapped particle " << i << " is wrong: \n" << xout[i] <<"\nexpected: \n" << expectedChildCoords[i];
                succeed = false;
            }
        }


        /// test applyJ and everything related to Jacobians. First, create auxiliary vectors.
        const Index Nc=outDofs->getSize();
        helper::vector<Index> Np(inDofs.size());
        for(Index i=0; i<Np.size(); i++)
            Np[i] = inDofs[i]->getSize();

        helper::vector<InVecCoord> xp(Np.size()),xp1(Np.size());
        helper::vector<InVecDeriv> vp(Np.size()),fp(Np.size()),dfp(Np.size()),fp2(Np.size());
        OutVecCoord xc(Nc),xc1(Nc);
        OutVecDeriv vc(Nc),fc(Nc);

        // get position data
        for(Index i=0; i<Np.size(); i++)
            copyFromData( xp[i],inDofs[i]->readPositions() );
        copyFromData( xc,  outDofs->readPositions() ); // positions and have already been propagated
        //        cout<<"parent positions xp = "<< xp << endl;
        //        cout<<"child  positions xc = "<< xc << endl;

        // set random child forces and propagate them to the parent
        for( unsigned i=0; i<Nc; i++ ){
            fc[i] = Out::randomDeriv( 0.1, 1.0 );
//            cout<<"random child forces  fc[" << i <<"] = "<<fc[i]<<endl;
        }
        for(Index p=0; p<Np.size(); p++) {
            fp2[p]=InVecDeriv(Np[p], InDeriv() ); // null vector of appropriate size
            WriteInVecDeriv fin = inDofs[p]->writeForces();
            copyToData( fin, fp2[p] );  // reset parent forces before accumulating child forces
        }
        WriteOutVecDeriv fout = outDofs->writeForces();
        copyToData( fout, fc );
        mapping->applyJT( &mparams, core::VecDerivId::force(), core::VecDerivId::force() );
        for(Index i=0; i<Np.size(); i++) copyFromData( fp[i], inDofs[i]->readForces() );
        //        cout<<"parent forces fp = "<<fp<<endl;

        // set small parent velocities and use them to update the child
        for( Index p=0; p<Np.size(); p++ ){
            vp[p].resize(Np[p]);
            xp1[p].resize(Np[p]);
            for( unsigned i=0; i<Np[p]; i++ ){
                vp[p][i] = In::randomDeriv( this->epsilon() * deltaRange.first, this->epsilon() * deltaRange.second );
//                cout<<"parent velocities vp[" << p <<"] = " << vp[p] << endl;
                xp1[p][i] = xp[p][i] + vp[p][i];
//                cout<<"new parent positions xp1["<< p <<"] = " << xp1[p] << endl;
            }
        }

        // propagate small velocity
        for( Index p=0; p<Np.size(); p++ ){
            WriteInVecDeriv vin = inDofs[p]->writeVelocities();
            copyToData( vin, vp[p] );
        }
        mapping->applyJ( &mparams, core::VecDerivId::velocity(), core::VecDerivId::velocity() );
        ReadOutVecDeriv vout = outDofs->readVelocities();
        copyFromData( vc, vout);
        //        cout<<"child velocity vc = " << vc << endl;


        // apply geometric stiffness
        for( Index p=0; p<Np.size(); p++ ) {
            WriteInVecDeriv dxin = inDofs[p]->writeDx();
            copyToData( dxin, vp[p] );
            dfp[p] = InVecDeriv(Np[p], InDeriv() );
            WriteInVecDeriv fin = inDofs[p]->writeForces();
            copyToData( fin, dfp[p] );
        }
        mapping->applyDJT( &mparams, core::VecDerivId::force(), core::VecDerivId::force() );
        for( Index p=0; p<Np.size(); p++ ){
            copyFromData( dfp[p], inDofs[p]->readForces() ); // fp + df due to geometric stiffness
//            cout<<"dfp["<< p <<"] = " << dfp[p] << endl;
        }

        // Jacobian will be obsolete after applying new positions
        const helper::vector<defaulttype::BaseMatrix*>* J = mapping->getJs();
        OutVecDeriv Jv(Nc);
        for( Index p=0; p<Np.size(); p++ ){
            //cout<<"J["<< p <<"] = "<< endl << *(*J)[p] << endl;
            EigenSparseMatrix* JJ = dynamic_cast<EigenSparseMatrix*>((*J)[p]);
            assert(JJ!=NULL);
            JJ->addMult(Jv,vp[p]);
        }

        // ================ test applyJT()
        helper::vector<InVecDeriv> jfc(Np.size());
        for( Index p=0; p<Np.size(); p++ ) {
            jfc[p] = InVecDeriv( Np[p],InDeriv());
            EigenSparseMatrix* JJ = dynamic_cast<EigenSparseMatrix*>((*J)[p]);
            JJ->addMultTranspose(jfc[p],fc);
            if( this->vectorMaxDiff(jfc[p],fp[p])>this->epsilon()*errorMax ){
                succeed = false;
                ADD_FAILURE() << "applyJT test failed"<<std::endl<<"jfc["<< p <<"] = " << jfc[p] << std::endl<<" fp["<< p <<"] = " << fp[p] << std::endl;
            }
        }
        // ================ test getJs()
        // check that J.vp = vc
        if( this->vectorMaxDiff(Jv,vc)>this->epsilon()*errorMax ){
            succeed = false;
            for( Index p=0; p<Np.size(); p++ ) {
                std::cout<<"J["<< p <<"] = "<< std::endl << *(*J)[p] << std::endl;
                std::cout<<"vp["<< p <<"] = " << vp[p] << std::endl;
            }
            std::cout<<"Jvp = " << Jv << std::endl;
            std::cout<<"vc  = " << vc << std::endl;
            ADD_FAILURE() << "getJs() test failed"<<std::endl<<"Jvp = " << Jv << std::endl <<"vc  = " << vc << std::endl;
        }


        // compute parent forces from pre-treated child forces (in most cases, the pre-treatment does nothing)
        // the pre-treatement can be useful to be able to compute 2 comparable results of applyJT with a small displacement to test applyDJT
        for( Index p=0; p<Np.size(); p++ ) {
            fp[p].fill( InDeriv() );
            WriteInVecDeriv fin = inDofs[p]->writeForces();
            copyToData( fin, fp[p] );  // reset parent forces before accumulating child forces
        }
        copyToData( fout, fc );
        mapping->applyJT( &mparams, core::VecDerivId::force(), core::VecDerivId::force() );
        for( Index p=0; p<Np.size(); p++ )
            copyFromData( fp[p], inDofs[p]->readForces() );



        ///////////////////////////////


        // propagate small displacement
        for( Index p=0; p<Np.size(); p++ ){
            WriteInVecCoord pin = inDofs[p]->writePositions();
            copyToData( pin, xp1[p] );
//            cout<<"new parent positions xp1["<< p << "] = " << xp1[p] << endl;
        }
        mapping->apply ( &mparams, core::VecCoordId::position(), core::VecCoordId::position() );
        WriteOutVecCoord pout = outDofs->writePositions();
        copyFromData( xc1, pout );
//        cout<<"new child positions xc1 = " << xc1 << endl;

        // ================ test applyJ: compute the difference between propagated displacements and velocities
        OutVecDeriv dxc(Nc);
        for(unsigned i=0; i<Nc; i++ ){
            dxc[i] = difference( xc1[i], xc[i] );
        }
        if( this->vectorMaxDiff(dxc,vc)>this->epsilon()*errorMax ){
            succeed = false;
            ADD_FAILURE() << "applyJ test failed: the difference between child position change and child velocity (dt=1) should be less than  " << this->epsilon()*errorMax  << std::endl
                          << "position change = " << dxc << std::endl
                          << "velocity        = " << vc << std::endl;
        }



        // update parent force based on the same child forces
        for( Index p=0; p<Np.size(); p++ ){
            fp2[p].fill( InDeriv() );
            WriteInVecDeriv fin = inDofs[p]->writeForces();
            copyToData( fin, fp2[p] );  // reset parent forces before accumulating child forces
        }
        copyToData( fout, fc );
        mapping->applyJT( &mparams, core::VecDerivId::force(), core::VecDerivId::force() );
        helper::vector<InVecDeriv> fp12(Np.size());
        for( Index p=0; p<Np.size(); p++ ){
            copyFromData( fp2[p], inDofs[p]->readForces() );
//            cout<<"updated parent forces fp2["<< p <<"] = "<< fp2[p] << endl;
            fp12[p].resize(Np[p]);
            for(unsigned i=0; i<Np[p]; i++){
                fp12[p][i] = fp2[p][i]-fp[p][i];       // fp2 - fp
            }
//            cout<<"fp2["<< p <<"] - fp["<< p <<"] = " << fp12[p] << endl;
            // ================ test applyDJT()
            if( this->vectorMaxDiff(dfp[p],fp12[p])>this->epsilon()*errorMax ){
                succeed = false;
                ADD_FAILURE() << "applyDJT test failed" << std::endl <<
                                 "dfp["<<p<<"]    = " << dfp[p] << std::endl <<
                                 "fp2["<<p<<"]-fp["<<p<<"] = " << fp12[p] << std::endl;
            }
        }

        return succeed;
    }

    virtual ~MultiMapping_test()
    {
        if (root!=NULL)
            sofa::simulation::getSimulation()->unload(root);
    }

};

} // namespace sofa


#endif
