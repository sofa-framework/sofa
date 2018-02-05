/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sstream>

#include "Sofa_test.h"
#include <sofa/core/MechanicalParams.h>
#include <sofa/simulation/VectorOperations.h>
#include <SofaBaseLinearSolver/FullVector.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include <SceneCreator/SceneCreator.h>
#include <sofa/helper/vector.h>
#include <sofa/core/MultiMapping.h>

#include <Flexible/types/AffineTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/Multi2Mapping.inl>


namespace sofa {
namespace {
using namespace core;
using namespace component;
using defaulttype::Vec;
using defaulttype::Mat;
using sofa::helper::vector;
typedef std::size_t Index;

/**  Test suite for Multi2Mapping.
  */
template <typename _MultiMapping>
struct Multi2Mapping_test : public Sofa_test<typename _MultiMapping::Real>
{

    typedef _MultiMapping Mapping;

    typedef typename Mapping::In1 In1Type;
    typedef typename In1Type::VecCoord In1VecCoord;
    typedef typename In1Type::VecDeriv In1VecDeriv;
    typedef typename In1Type::Coord In1Coord;
    typedef typename In1Type::Deriv In1Deriv;
    typedef typename Mapping::In1DataVecCoord In1DataVecCoord;
    typedef typename Mapping::In1DataVecDeriv In1DataVecDeriv;
    typedef container::MechanicalObject<In1Type> In1DOFs;
    typedef typename In1DOFs::ReadVecCoord  ReadIn1VecCoord;
    typedef typename In1DOFs::WriteVecCoord WriteIn1VecCoord;
    typedef typename In1DOFs::WriteVecDeriv WriteIn1VecDeriv;
    typedef typename In1Type::Real Real;
    typedef Mat<In1Type::spatial_dimensions, In1Type::spatial_dimensions, Real> In1RotationMatrix;

    typedef typename Mapping::In2 In2Type;
    typedef typename In2Type::VecCoord In2VecCoord;
    typedef typename In2Type::VecDeriv In2VecDeriv;
    typedef typename In2Type::Coord In2Coord;
    typedef typename In2Type::Deriv In2Deriv;
    typedef typename Mapping::In2DataVecCoord In2DataVecCoord;
    typedef typename Mapping::In2DataVecDeriv In2DataVecDeriv;
    typedef container::MechanicalObject<In2Type> In2DOFs;
    typedef typename In2DOFs::ReadVecCoord  ReadIn2VecCoord;
    typedef typename In2DOFs::WriteVecCoord WriteIn2VecCoord;
    typedef typename In2DOFs::WriteVecDeriv WriteIn2VecDeriv;
    typedef Mat<In2Type::spatial_dimensions, In2Type::spatial_dimensions, Real> In2RotationMatrix;

    typedef typename Mapping::Out OutType;
    typedef typename OutType::VecCoord OutVecCoord;
    typedef typename OutType::VecDeriv OutVecDeriv;
    typedef typename OutType::Coord OutCoord;
    typedef typename OutType::Deriv OutDeriv;
    typedef typename Mapping::OutDataVecCoord OutDataVecCoord;
    typedef typename Mapping::OutDataVecDeriv OutDataVecDeriv;
    typedef container::MechanicalObject<OutType> OutDOFs;
    typedef typename OutDOFs::WriteVecCoord WriteOutVecCoord;
    typedef typename OutDOFs::WriteVecDeriv WriteOutVecDeriv;
    typedef typename OutDOFs::ReadVecCoord ReadOutVecCoord;
    typedef typename OutDOFs::ReadVecDeriv ReadOutVecDeriv;

    typedef core::Multi2Mapping <In1Type, In2Type, OutType> Multi2Mapping;

    typedef component::linearsolver::EigenSparseMatrix<In1Type, OutType> SparseJMatrixEigen1;
    typedef component::linearsolver::EigenSparseMatrix<In2Type, OutType> SparseJMatrixEigen2;
    typedef linearsolver::EigenSparseMatrix<In1Type, In2Type> SparseKMatrixEigen1;
    typedef linearsolver::EigenSparseMatrix<In2Type, In2Type> SparseKMatrixEigen2;

    Multi2Mapping* mapping; ///< the mapping to be tested
    vector<In1DOFs*>  in1Dofs; ///< mapping input
    vector<In2DOFs*>  in2Dofs; ///< mapping input
    OutDOFs* outDofs; ///< mapping output
    simulation::Node* root; ///< Root of the scene graph, created by the constructor an re-used in the tests
    simulation::Node::SPtr child; ///< Child node, created by setupScene
    simulation::Node::SPtr parentsIn1, parentsIn2; ///< Parent nodes, created by setupScene
    simulation::Simulation* simulation; ///< created by the constructor an re-used in the tests
    std::pair<Real, Real> deltaRange; ///< The minimum and maximum magnitudes of the change of each scalar value of the small displacement is deltaRange * numeric_limits<Real>::epsilon. This epsilon is 1.19209e-07 for float and 2.22045e-16 for double.
    Real errorMax; ///< The test is successfull if the (infinite norm of the) difference is less than  maxError * numeric_limits<Real>::epsilon


    /// Constructor
    Multi2Mapping_test() :deltaRange(1, 1000), errorMax(10)
    {
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());

    }

    virtual ~Multi2Mapping_test()
    {
        if (root!=NULL)
            sofa::simulation::getSimulation()->unload(root);
    }

    /** Returns OutCoord substraction a-b (should return a OutDeriv, but???)
    */
    OutDeriv difference(const OutCoord& c1, const OutCoord& c2)
    {
        return OutType::coordDifference(c1, c2);
    }

    /** Create scene with given number of parent states. Currently, only one child state is handled.
    * All the parents are set as child of the root node, while the child is in a child node.
    */
    void setupScene()
    {
        root = simulation->createNewGraph("root").get();

        /// Child node
        this->child = root->createChild("childNode");
        this->outDofs = modeling::addNew<OutDOFs>(child).get();
        this->mapping = modeling::addNew<Mapping>(child).get();
        this->mapping->addOutputModel(outDofs);

        /// Parent states, added to specific parentNode nodes. This is not a simulable scene.
        this->parentsIn1 = root->createChild("parentNodeIn1");
        this->parentsIn2 = root->createChild("parentNodeIn2");
        typename In1DOFs::SPtr inDof1 = modeling::addNew<In1DOFs>(parentsIn1);
        typename In2DOFs::SPtr inDof2 = modeling::addNew<In2DOFs>(parentsIn2);
        this->mapping->addInputModel1(inDof1.get());
        this->mapping->addInputModel2(inDof2.get());
        this->in1Dofs.push_back(inDof1.get());
        this->in2Dofs.push_back(inDof2.get());
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
    * This version which test Multi2Mapping works only when we have two parent for one child.
    *
    *\param parentCoords Parent positions (one InVecCoord per parent)
    *\param expectedChildCoords expected position of the child corresponding to the parent positions
    */
    bool runTest(const vector<In1VecCoord>& in1Coords, const vector<In2VecCoord>& in2Coords, const OutVecCoord& expectedChildCoords)
    {
        if (deltaRange.second / errorMax <= g_minDeltaErrorRatio) ADD_FAILURE() << "The comparison threshold is too large for the finite difference delta";

        core::MechanicalParams mparams;
        mparams.setKFactor(1.0);
        mparams.setSymmetricMatrix(false);

        // transfer the parent values in the parent states
        // --- Rigid dofs
        for (size_t i = 0; i < in1Coords.size(); i++)
        {
            this->in1Dofs[i]->resize(in1Coords[i].size());
            WriteIn1VecCoord xin1 = in1Dofs[i]->writePositions();
            copyToData(xin1, in1Coords[i]);
        }
        // --- Scale dofs
        for (size_t i = 0; i < in2Coords.size(); i++)
        {
            this->in2Dofs[i]->resize(in2Coords[i].size());
            WriteIn2VecCoord xin2 = in2Dofs[i]->writePositions();
            copyToData(xin2, in2Coords[i]);
        }

        /// Init
        sofa::simulation::getSimulation()->init(root);

        /// Apply the mapping
        // --- Use of the method apply
        this->mapping->apply(&mparams, core::VecCoordId::position(), core::VecCoordId::position());
        this->mapping->applyJ(&mparams, core::VecDerivId::velocity(), core::VecDerivId::velocity() );
        // ================ test apply : check if the child positions are the expected ones
        bool succeed = true;
        ReadOutVecCoord xout = this->outDofs->readPositions();
        for (Index i = 0; i < xout.size(); i++)
        {
            if (!this->isSmall(difference(xout[i], expectedChildCoords[i]).norm(), errorMax)) {
                ADD_FAILURE() << "Position of mapped particle " << i << " is wrong: \n" << xout[i] << "\nexpected: \n" << expectedChildCoords[i];
                succeed = false;
            }
        }

        // test applyJ and everything related to Jacobians. First, create auxiliary vectors.
        vector<Index> Np1(this->in1Dofs.size()), Np2(this->in2Dofs.size());
        const Index Nc = this->outDofs->getSize();
        for (Index i = 0; i < Np1.size(); i++) Np1[i] = this->in1Dofs[i]->getSize();
        for (Index i = 0; i < Np2.size(); i++) Np2[i] = this->in2Dofs[i]->getSize();

        // Variable
        vector<In1VecCoord> xIn1p(Np1.size()), xIn1p1(Np1.size());
        vector<In2VecCoord> xIn2p(Np2.size()), xIn2p1(Np2.size());
        vector<In1VecDeriv> vIn1p(Np1.size()), fIn1p(Np1.size()), dfIn1p(Np1.size()), fIn1p2(Np1.size());
        vector<In2VecDeriv> vIn2p(Np2.size()), fIn2p(Np2.size()), dfIn2p(Np2.size()), fIn2p2(Np2.size());
        OutVecCoord xc(Nc), xc1(Nc);
        OutVecDeriv vc(Nc), fc(Nc);

        // get position data
        for (Index i = 0; i < Np1.size(); i++) copyFromData(xIn1p[i], this->in1Dofs[i]->readPositions());
        for (Index i = 0; i < Np2.size(); i++) copyFromData(xIn2p[i], this->in2Dofs[i]->readPositions());
        copyFromData(xc, this->outDofs->readPositions()); // positions and have already been propagated

        // set random child forces and propagate them to the parent
        for (unsigned i = 0; i < Nc; i++) fc[i] = OutType::randomDeriv(0.1, 1);
        for (Index p = 0; p < Np1.size(); p++)
        {
            fIn1p2[p] = In1VecDeriv(Np1[p], In1Deriv()); // null vector of appropriate size
            WriteIn1VecDeriv fIn1 = this->in1Dofs[p]->writeForces();
            copyToData(fIn1, fIn1p2[p]);  // reset parent forces before accumulating child forces
        }
        for (Index p = 0; p < Np2.size(); p++)
        {
            fIn2p2[p] = In2VecDeriv(Np2[p], In2Deriv()); // null vector of appropriate size
            WriteIn2VecDeriv fIn2 = this->in2Dofs[p]->writeForces();
            copyToData(fIn2, fIn2p2[p]);  // reset parent forces before accumulating child forces
        }

        WriteOutVecDeriv fout = outDofs->writeForces();
        copyToData(fout, fc);
        this->mapping->applyJT( &mparams, core::VecDerivId::force(), core::VecDerivId::force() );
        for (Index i = 0; i < Np1.size(); i++) copyFromData(fIn1p[i], this->in1Dofs[i]->readForces());
        for (Index i = 0; i < Np2.size(); i++) copyFromData(fIn2p[i], this->in2Dofs[i]->readForces());

        // set small parent velocities and use them to update the child
        for (Index p = 0; p < Np1.size(); p++)
        {
            vIn1p[p].resize(Np1[p]); xIn1p1[p].resize(Np1[p]);
            for (unsigned i = 0; i < Np1[p]; i++)
            {
                vIn1p[p][i] = In1Type::randomDeriv(this->epsilon() * deltaRange.first, this->epsilon() * deltaRange.second);
                xIn1p1[p][i] = xIn1p[p][i] + vIn1p[p][i];
            }
        }
        for (Index p = 0; p < Np2.size(); p++)
        {
            vIn2p[p].resize(Np2[p]); xIn2p1[p].resize(Np2[p]);
            for (unsigned i = 0; i < Np2[p]; i++)
            {
                vIn2p[p][i] = In2Type::randomDeriv(this->epsilon() * deltaRange.first, this->epsilon() * deltaRange.second);
                xIn2p1[p][i] = xIn2p[p][i] + vIn2p[p][i];
            }
        }

        // propagate small velocity
        for (Index p = 0; p < Np1.size(); p++)
        {
            WriteIn1VecDeriv vIn1 = this->in1Dofs[p]->writeVelocities();
            copyToData(vIn1, vIn1p[p]);
        }
        for (Index p = 0; p < Np2.size(); p++)
        {
            WriteIn2VecDeriv vIn2 = this->in2Dofs[p]->writeVelocities();
            copyToData(vIn2, vIn2p[p]);
        }
        this->mapping->applyJ(&mparams, core::VecDerivId::velocity(), core::VecDerivId::velocity() );
        ReadOutVecDeriv vout = this->outDofs->readVelocities();
        copyFromData(vc, vout);

        // apply geometric stiffness
        for( Index p=0; p<Np1.size(); p++ )
        {
            WriteIn1VecDeriv dxin1 = this->in1Dofs[p]->writeDx(); // It seems it does not work, i dont why
            copyToData( dxin1, vIn1p[p] ); // It seems it does not work, i dont why
            this->in1Dofs[p]->dx.setValue(vIn1p[p]); // Then i replace it by this lines
            dfIn1p[p] = In1VecDeriv(Np1[p], In1Deriv());
            WriteIn1VecDeriv fin1 = this->in1Dofs[p]->writeForces();
            copyToData( fin1, dfIn1p[p] );
        }
        for( Index p=0; p<Np2.size(); p++ )
        {
            WriteIn2VecDeriv dxin2 = this->in2Dofs[p]->writeDx(); // It seems it does not work, i dont why
            copyToData( dxin2, vIn2p[p] ); // It seems it does not work, i dont why
            this->in2Dofs[p]->dx.setValue(vIn2p[p]); // Then i replace it by this lines
            dfIn2p[p] = In2VecDeriv(Np2[p], In2Deriv());
            WriteIn2VecDeriv fin2 = this->in2Dofs[p]->writeForces();
            copyToData( fin2, dfIn2p[p] );
        }
        mapping->applyDJT( &mparams, core::VecDerivId::force(), core::VecDerivId::force() );
        for( Index p=0; p<Np1.size(); p++ ) copyFromData( dfIn1p[p], in1Dofs[p]->readForces() ); // fp + df due to geometric stiffness
        for( Index p=0; p<Np2.size(); p++ ) copyFromData( dfIn2p[p], in2Dofs[p]->readForces() ); // fp + df due to geometric stiffness

        // Jacobian will be obsolete after applying new positions
        const vector<defaulttype::BaseMatrix*>* J = mapping->getJs();
        SparseJMatrixEigen1* J1 = dynamic_cast<SparseJMatrixEigen1*>((*J)[0]);
        SparseJMatrixEigen2* J2 = dynamic_cast<SparseJMatrixEigen2*>((*J)[1]);
        OutVecDeriv Jv(Nc);
        assert(J1 != NULL && J2 != NULL);
        J1->addMult(Jv, vIn1p[0]);
        J2->addMult(Jv, vIn2p[0]);

        // ================ test applyJT()
        vector<In1VecDeriv> jfcIn1(Np1.size()); vector<In2VecDeriv> jfcIn2(Np2.size());
        for (Index p = 0; p<Np1.size(); p++)
        {
            jfcIn1[p] = In1VecDeriv(Np1[p], In1Deriv());
            J1->addMultTranspose(jfcIn1[p], fc);
            if (this->vectorMaxDiff(jfcIn1[p], fIn1p[p])>this->epsilon()*errorMax)
            {
                succeed = false;
                ADD_FAILURE() << "applyJT (parent 1) test failed" << endl << "jfcIn1[" << p << "] = " << jfcIn1[p] << endl << " fp[" << p << "] = " << fIn1p[p] << endl;
            }
        }
        for (Index p = 0; p<Np2.size(); p++)
        {
            jfcIn2[p] = In2VecDeriv(Np2[p], In2Deriv());
            J2->addMultTranspose(jfcIn2[p], fc);
            if (this->vectorMaxDiff(jfcIn2[p], fIn2p[p])>this->epsilon()*errorMax)
            {
                succeed = false;
                ADD_FAILURE() << "applyJT (parent 2) test failed" << endl << "jfcIn2[" << p << "] = " << jfcIn2[p] << endl << " fp[" << p << "] = " << fIn2p[p] << endl;
            }
        }

        // ================ test getJs() : check that J.vp = vc
        if (this->vectorMaxDiff(Jv, vc) > this->epsilon()*errorMax){
            succeed = false;
            cout << "Jvp = " << Jv << endl;
            cout << "vc  = " << vc << endl;
            ADD_FAILURE() << "getJs() test failed" << endl << "Jvp = " << Jv << endl << "vc  = " << vc << endl;
        }

        // compute parent forces from pre-treated child forces (in most cases, the pre-treatment does nothing)
        // the pre-treatement can be useful to be able to compute 2 comparable results of applyJT with a small displacement to test applyDJT
        for (Index p = 0; p < Np1.size(); p++)
        {
            fIn1p[p].fill(In1Deriv());
            WriteIn1VecDeriv fin = in1Dofs[p]->writeForces();
            copyToData(fin, fIn1p[p]);  // reset parent forces before accumulating child forces
        }
        for (Index p = 0; p < Np2.size(); p++)
        {
            fIn2p[p].fill(In2Deriv());
            WriteIn2VecDeriv fin = in2Dofs[p]->writeForces();
            copyToData(fin, fIn2p[p]);  // reset parent forces before accumulating child forces
        }
        copyToData(fout, fc);
        this->mapping->applyJT( &mparams, core::VecDerivId::force(), core::VecDerivId::force() );
        for (Index i = 0; i < Np1.size(); i++) copyFromData(fIn1p[i], this->in1Dofs[i]->readForces());
        for (Index i = 0; i < Np2.size(); i++) copyFromData(fIn2p[i], this->in2Dofs[i]->readForces());

        // propagate small displacement
        for (Index p = 0; p < Np1.size(); p++)
        {
            WriteIn1VecCoord pin1 = in1Dofs[p]->writePositions();
            copyToData(pin1, xIn1p1[p]);

        }
        for (Index p = 0; p < Np2.size(); p++)
        {
            WriteIn2VecCoord pin2 = in2Dofs[p]->writePositions();
            copyToData(pin2, xIn2p1[p]);
        }
        this->mapping->apply(&mparams, core::VecCoordId::position(), core::VecCoordId::position());
        WriteOutVecCoord pout = this->outDofs->writePositions();
        copyFromData(xc1, pout);

        // ================ test applyJ: compute the difference between propagated displacements and velocities
        OutVecDeriv dxc(Nc);
        for (unsigned i = 0; i<Nc; i++) dxc[i] = difference(xc1[i], xc[i]);
        if (this->vectorMaxDiff(dxc, vc) > this->epsilon()*errorMax)
        {
            succeed = false;
            ADD_FAILURE() << "applyJ test failed: the difference between child position change and child velocity (dt=1) should be less than  " << this->epsilon()*errorMax << endl
                          << "position change = " << dxc << endl
                          << "velocity        = " << vc << endl;
        }

        // update parent force based on the same child forces
        for (Index p = 0; p < Np1.size(); p++)
        {
            fIn1p2[p].fill(In1Deriv());
            WriteIn1VecDeriv fin1 = in1Dofs[p]->writeForces();
            copyToData(fin1, fIn1p2[p]);  // reset parent forces before accumulating child forces
        }
        for (Index p = 0; p < Np2.size(); p++)
        {
            fIn2p2[p].fill(In2Deriv());
            WriteIn2VecDeriv fin2 = in2Dofs[p]->writeForces();
            copyToData(fin2, fIn2p2[p]);  // reset parent forces before accumulating child forces
        }
        copyToData(fout, fc);
        this->mapping->applyJT( &mparams, core::VecDerivId::force(), core::VecDerivId::force() );
        vector<In1VecDeriv> fIn1p12(Np1.size());
        vector<In2VecDeriv> fIn2p12(Np2.size());
        // ================ test applyDJT() (Case 1)
        for( Index p=0; p<Np1.size(); p++ )
        {
            copyFromData( fIn1p2[p], in1Dofs[p]->readForces() );
            fIn1p12[p].resize(Np1[p]);
            for(unsigned i=0; i<Np1[p]; i++) fIn1p12[p][i] = fIn1p2[p][i] - fIn1p[p][i];
            if( this->vectorMaxDiff(dfIn1p[p],fIn1p12[p]) > this->epsilon()*errorMax )
            {
                succeed = false;
                ADD_FAILURE() << "applyDJT test (parent 1) failed" << endl <<
                                 "dfIn1p["<<p<<"]    = " << dfIn1p[p] << endl <<
                                 "fIn1p2["<<p<<"]-fIn1p["<<p<<"] = " << fIn1p12[p] << endl;
            }
        }
        // ================ test applyDJT() (Case 2)
        for( Index p=0; p<Np2.size(); p++ )
        {
            copyFromData( fIn2p2[p], in2Dofs[p]->readForces() );
            fIn2p12[p].resize(Np2[p]);
            for(unsigned i=0; i<Np2[p]; i++) fIn2p12[p][i] = fIn2p2[p][i] - fIn2p[p][i];
            if( this->vectorMaxDiff(dfIn2p[p],fIn2p12[p]) > this->epsilon()*errorMax )
            {
                succeed = false;
                ADD_FAILURE() << "applyDJT test (parent 2) failed" << endl <<
                                 "dfIn2p["<<p<<"]    = " << dfIn2p[p] << endl <<
                                 "fIn2p2["<<p<<"]-fIn2p["<<p<<"] = " << fIn2p12[p] << endl;
            }
        }

        // =================== test updateForceMask
        // propagate forces coming from all child, each parent receiving a force should be in the mask
        for(Index i=0; i<Np1.size(); i++) in1Dofs[i]->forceMask.clear();
        for(Index i=0; i<Np2.size(); i++) in2Dofs[i]->forceMask.clear();
        outDofs->forceMask.assign(outDofs->getSize(),true);
        mapping->apply(&mparams, core::VecCoordId::position(), core::VecCoordId::position()); // to force mask update at the next applyJ
        for( unsigned i=0; i<Nc; i++ ) Out::set( fout[i], 1,1,1 ); // every child forces are non-nul
        for(Index p=0; p<Np1.size(); p++) {
            WriteInVecDeriv fin = in1Dofs[p]->writeForces();
            copyToData( fin, fIn1p2[p] );  // reset parent forces before accumulating child forces
        }
        for(Index p=0; p<Np2.size(); p++) {
            WriteInVecDeriv fin = in2Dofs[p]->writeForces();
            copyToData( fin, fIn2p2[p] );  // reset parent forces before accumulating child forces
        }
        mapping->applyJT( &mparams, core::VecDerivId::force(), core::VecDerivId::force() );
        for(Index i=0; i<Np1.size(); i++)
        {
            copyFromData( fIn1p[i], in1Dofs[i]->readForces() );
            for( unsigned j=0; j<Np1[i]; j++ ) {
                if( fIn1p[i][j] != InDeriv() && !in1Dofs[i]->forceMask.getEntry(j) ){
                    succeed = false;
                    ADD_FAILURE() << "updateForceMask did not propagate mask to every influencing parents 0-"<< i << std::endl;
                    break;
                }
            }
        }
        for(Index i=0; i<Np2.size(); i++)
        {
            copyFromData( fIn2p[i], in2Dofs[i]->readForces() );
            for( unsigned j=0; j<Np2[i]; j++ ) {
                if( fIn2p[i][j] != InDeriv() && !in2Dofs[i]->forceMask.getEntry(j) ){
                    succeed = false;
                    ADD_FAILURE() << "updateForceMask did not propagate mask to every influencing parents 1-"<< i << std::endl;
                    break;
                }
            }
        }

        return succeed;
    }

};
} // namespace
} // namespace sofa
