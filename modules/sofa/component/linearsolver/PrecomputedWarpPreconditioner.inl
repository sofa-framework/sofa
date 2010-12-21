/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
// Author: Hadrien Courtecuisse
//
// Copyright: See COPYING file that comes with this distribution

#ifndef SOFA_COMPONENT_LINEARSOLVER_PrecomputedWarpPreconditioner_INL
#define SOFA_COMPONENT_LINEARSOLVER_PrecomputedWarpPreconditioner_INL

#include "PrecomputedWarpPreconditioner.h"
#include <sofa/component/linearsolver/NewMatMatrix.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/component/linearsolver/SparseMatrix.h>
#include <iostream>
#include "sofa/helper/system/thread/CTime.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <math.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/component/forcefield/TetrahedronFEMForceField.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/linearsolver/MatrixLinearSolver.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/component/container/RotationFinder.inl>
#include <sofa/core/behavior/LinearSolver.h>

#include <sofa/helper/gl/DrawManager.h>
#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/Quater.h>

#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>
#include <sofa/component/linearsolver/PCGLinearSolver.h>

#ifdef SOFA_HAVE_CSPARSE
#include <sofa/component/linearsolver/SparseCholeskySolver.h>
#include <sofa/component/linearsolver/CompressedRowSparseMatrix.h>
#else
#include <sofa/component/linearsolver/CholeskySolver.h>
#endif


namespace sofa
{

namespace component
{

namespace linearsolver
{

using namespace sofa::component::odesolver;
using namespace sofa::component::linearsolver;

template<class TDataTypes>
PrecomputedWarpPreconditioner<TDataTypes>::PrecomputedWarpPreconditioner()
    : jmjt_twostep( initData(&jmjt_twostep,true,"jmjt_twostep","Use two step algorithm to compute JMinvJt") )
    , f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , use_file( initData(&use_file,true,"use_file","Dump system matrix in a file") )
    , share_matrix( initData(&share_matrix,true,"share_matrix","Share the compliance matrix in memory if they are related to the same file (WARNING: might require to reload Sofa when opening a new scene...)") )
    , solverName(initData(&solverName, std::string(""), "solverName", "Name of the solver to use to precompute the first matrix"))
    , use_rotations( initData(&use_rotations,true,"use_rotations","Use Rotations around the preconditioner") )
    , draw_rotations_scale( initData(&draw_rotations_scale,0.5,"draw_rotations_scale","Scale rotations in draw function") )
{
    first = true;
    _rotate = false;
    usePrecond = true;
}

template<class TDataTypes>
void PrecomputedWarpPreconditioner<TDataTypes>::setSystemMBKMatrix(const core::MechanicalParams* mparams)
{
    // Update the matrix only the first time
    if (first)
    {
        init_mFact = mparams->mFactor();
        init_bFact = mparams->bFactor();
        init_kFact = mparams->kFactor();
        Inherit::setSystemMBKMatrix(mparams);
        this->currentGroup->needInvert = true;
    }
    else
    {
        this->currentGroup->needInvert = usePrecond;
    }
}

//Solve x = R * M^-1 * R^t * b
template<class TDataTypes>
void PrecomputedWarpPreconditioner<TDataTypes>::solve (TMatrix& /*M*/, TVector& z, TVector& r)
{
    if (usePrecond)
    {
        if (use_rotations.getValue())
        {
            unsigned int k = 0;
            unsigned int l = 0;

            //Solve z = R^t * b
            while (l < internalData.MinvPtr->colSize())
            {
                z[l+0] = R[k + 0] * r[l + 0] + R[k + 3] * r[l + 1] + R[k + 6] * r[l + 2];
                z[l+1] = R[k + 1] * r[l + 0] + R[k + 4] * r[l + 1] + R[k + 7] * r[l + 2];
                z[l+2] = R[k + 2] * r[l + 0] + R[k + 5] * r[l + 1] + R[k + 8] * r[l + 2];
                l+=3;
                k+=9;
            }

            //Solve tmp = M^-1 * z
            T = (*internalData.MinvPtr) * z;

            //Solve z = R * tmp
            k = 0; l = 0;
            while (l < internalData.MinvPtr->colSize())
            {
                z[l+0] = R[k + 0] * T[l + 0] + R[k + 1] * T[l + 1] + R[k + 2] * T[l + 2];
                z[l+1] = R[k + 3] * T[l + 0] + R[k + 4] * T[l + 1] + R[k + 5] * T[l + 2];
                z[l+2] = R[k + 6] * T[l + 0] + R[k + 7] * T[l + 1] + R[k + 8] * T[l + 2];
                l+=3;
                k+=9;
            }
        }
        else
        {
            //Solve tmp = M^-1 * z
            z = (*internalData.MinvPtr) * r;
        }
    }
    else z = r;
}

template<class TDataTypes>
void PrecomputedWarpPreconditioner<TDataTypes>::loadMatrix()
{
    unsigned systemSize = this->currentGroup->systemMatrix->rowSize();
    dt = this->getContext()->getDt();

    EulerImplicitSolver* EulerSolver;
    this->getContext()->get(EulerSolver);
    factInt = 1.0; // christian : it is not a compliance... but an admittance that is computed !
    if (EulerSolver) factInt = EulerSolver->getPositionIntegrationFactor(); // here, we compute a compliance

    std::stringstream ss;
    ss << this->getContext()->getName() << "-" << systemSize << "-" << dt << ((sizeof(Real)==sizeof(float)) ? ".compf" : ".comp");
    std::string fname = ss.str();

    if (share_matrix.getValue()) internalData.setMinv(internalData.getSharedMatrix(fname));

    if (share_matrix.getValue() && internalData.MinvPtr->rowSize() == systemSize)
    {
        cout << "shared matrix : " << fname << " is already built" << endl;
    }
    else
    {
        internalData.MinvPtr->resize(systemSize,systemSize);
        std::ifstream compFileIn(fname.c_str(), std::ifstream::binary);

        if(compFileIn.good() && use_file.getValue())
        {
            cout << "file open : " << fname << " compliance being loaded" << endl;
            compFileIn.read((char*) (*internalData.MinvPtr)[0], systemSize * systemSize * sizeof(Real));
            compFileIn.close();
        }
        else
        {
            cout << "Precompute : " << fname << " compliance" << endl;
            if (solverName.getValue().empty()) loadMatrixWithCSparse();
            else loadMatrixWithSolver();

            if (use_file.getValue())
            {
                std::ofstream compFileOut(fname.c_str(), std::fstream::out | std::fstream::binary);
                compFileOut.write((char*)(*internalData.MinvPtr)[0], systemSize * systemSize*sizeof(Real));
                compFileOut.close();
            }
        }

        for (unsigned int j=0; j<systemSize; j++)
        {
            for (unsigned i=0; i<systemSize; i++)
            {
                internalData.MinvPtr->set(j,i,internalData.MinvPtr->element(j,i)/factInt);
            }
        }
    }

    R.resize(3*systemSize);
    T.resize(systemSize);
    for(unsigned int k = 0; k < systemSize/3; k++)
    {
        R[k*9] = R[k*9+4] = R[k*9+8] = 1.0f;
        R[k*9+1] = R[k*9+2] = R[k*9+3] = R[k*9+5] = R[k*9+6] = R[k*9+7] = 0.0f;
    }
}

template<class TDataTypes>
void PrecomputedWarpPreconditioner<TDataTypes>::loadMatrixWithCSparse()
{
#ifdef SOFA_HAVE_CSPARSE
    cout << "Compute the initial invert matrix with CS_PARSE" << endl;

    FullVector<Real> r;
    FullVector<Real> b;

    unsigned systemSize = this->currentGroup->systemMatrix->colSize();

    r.resize(systemSize);
    b.resize(systemSize);
    SparseCholeskySolver<CompressedRowSparseMatrix<Real>, FullVector<Real> > solver;

    for (unsigned int j=0; j<systemSize; j++) b.set(j,0.0);

    std::cout << "Precomputing constraint correction LU decomposition " << std::endl;
    solver.invert(*this->currentGroup->systemMatrix);

    for (unsigned int j=0; j<systemSize; j++)
    {
        std::cout.precision(2);
        std::cout << "Precomputing constraint correction : " << std::fixed << (float)j/(float)systemSize*100.0f << " %   " << '\xd';
        std::cout.flush();

        if (j>0) b.set(j-1,0.0);
        b.set(j,1.0);

        solver.solve(*this->currentGroup->systemMatrix,r,b);
        for (unsigned int i=0; i<systemSize; i++)
        {
            internalData.MinvPtr->set(j,i,r.element(i)*factInt);
        }
    }
    std::cout << "Precomputing constraint correction : " << std::fixed << 100.0f << " %   " << '\xd';
    std::cout.flush();

#else
    std::cout << "WARNING ; you don't have CS_parse CG will be use, (if also can specify solverName to accelerate the precomputation" << std::endl;
    loadMatrixWithSolver();
#endif
}

template<class TDataTypes>
void PrecomputedWarpPreconditioner<TDataTypes>::loadMatrixWithSolver()
{
    usePrecond = false;//Don'Use precond during precomputing

    cout << "Compute the initial invert matrix with solver" << endl;

    behavior::MechanicalState<DataTypes>* mstate = dynamic_cast< behavior::MechanicalState<DataTypes>* >(this->getContext()->getMechanicalState());
    if (mstate==NULL)
    {
        serr << "PrecomputedWarpPreconditioner can't find Mstate" << sendl;
        return;
    }
    const VecDeriv& v0 = *mstate->getV();
    unsigned dof_on_node = v0[0].size();
    unsigned nbNodes = v0.size();
    unsigned systemSize = nbNodes*dof_on_node;

    std::stringstream ss;
    //ss << this->getContext()->getName() << "_CPP.comp";
    ss << this->getContext()->getName() << "-" << systemSize << "-" << dt << ((sizeof(Real)==sizeof(float)) ? ".compf" : ".comp");
    std::ifstream compFileIn(ss.str().c_str(), std::ifstream::binary);

    EulerImplicitSolver* EulerSolver;
    this->getContext()->get(EulerSolver);

    // for the initial computation, the gravity has to be put at 0
    const Vec3d gravity = this->getContext()->getGravityInWorld();
    const Vec3d gravity_zero(0.0,0.0,0.0);
    this->getContext()->setGravityInWorld(gravity_zero);

    PCGLinearSolver<GraphScatteredMatrix,GraphScatteredVector>* PCGlinearSolver;
    CGLinearSolver<GraphScatteredMatrix,GraphScatteredVector>* CGlinearSolver;
    core::behavior::LinearSolver* linearSolver;

    if (solverName.getValue().empty())
    {
        this->getContext()->get(CGlinearSolver);
        this->getContext()->get(PCGlinearSolver);
        this->getContext()->get(linearSolver);
    }
    else
    {
        core::objectmodel::BaseObject* ptr = NULL;
        this->getContext()->get(ptr, solverName.getValue());
        PCGlinearSolver = dynamic_cast<PCGLinearSolver<GraphScatteredMatrix,GraphScatteredVector>*>(ptr);
        CGlinearSolver = dynamic_cast<CGLinearSolver<GraphScatteredMatrix,GraphScatteredVector>*>(ptr);
        linearSolver = dynamic_cast<core::behavior::LinearSolver*>(ptr);
    }

    if(EulerSolver && CGlinearSolver)
        sout << "use EulerImplicitSolver &  CGLinearSolver" << sendl;
    else if(EulerSolver && PCGlinearSolver)
        sout << "use EulerImplicitSolver &  PCGLinearSolver" << sendl;
    else if(EulerSolver && linearSolver)
        sout << "use EulerImplicitSolver &  LinearSolver" << sendl;
    else if(EulerSolver)
    {
        sout << "use EulerImplicitSolver" << sendl;
    }
    else
    {
        serr<<"PrecomputedContactCorrection must be associated with EulerImplicitSolver+LinearSolver for the precomputation\nNo Precomputation" << sendl;
        return;
    }
    VecDerivId lhId = core::VecDerivId::velocity();
    VecDerivId rhId = core::VecDerivId::force();


    mstate->vAvail(lhId);
    mstate->vAlloc(lhId);
    mstate->vAvail(rhId);
    mstate->vAlloc(rhId);
    std::cout << "System: (" << init_mFact << " * M + " << init_bFact << " * B + " << init_kFact << " * K) " << lhId << " = " << rhId << std::endl;
    if (linearSolver)
    {
        std::cout << "System Init Solver: " << linearSolver->getName() << " (" << linearSolver->getClassName() << ")" << std::endl;
        core::MechanicalParams mparams;
        mparams.setMFactor(init_mFact);
        mparams.setBFactor(init_bFact);
        mparams.setKFactor(init_kFact);
        linearSolver->setSystemMBKMatrix(&mparams);
    }

    //<TO REMOVE>
    //VecDeriv& force = *mstate->getVecDeriv(rhId.index);
    //Data<VecDeriv>* dataForce = mstate->writeVecDeriv(rhId);
    //VecDeriv& force = *dataForce->beginEdit();
    helper::WriteAccessor<Data<VecDeriv> > dataForce = *mstate->write(core::VecDerivId::externalForce());
    VecDeriv& force = dataForce.wref();

    force.clear();
    force.resize(nbNodes);

    ///////////////////////// CHANGE THE PARAMETERS OF THE SOLVER /////////////////////////////////
    double buf_tolerance=0, buf_threshold=0;
    int buf_maxIter=0;
    if(CGlinearSolver)
    {
        buf_tolerance = (double) CGlinearSolver->f_tolerance.getValue();
        buf_maxIter   = (int) CGlinearSolver->f_maxIter.getValue();
        buf_threshold = (double) CGlinearSolver->f_smallDenominatorThreshold.getValue();
        CGlinearSolver->f_tolerance.setValue(1e-35);
        CGlinearSolver->f_maxIter.setValue(5000);
        CGlinearSolver->f_smallDenominatorThreshold.setValue(1e-25);
    }
    else if(PCGlinearSolver)
    {
        buf_tolerance = (double) PCGlinearSolver->f_tolerance.getValue();
        buf_maxIter   = (int) PCGlinearSolver->f_maxIter.getValue();
        buf_threshold = (double) PCGlinearSolver->f_smallDenominatorThreshold.getValue();
        PCGlinearSolver->f_tolerance.setValue(1e-35);
        PCGlinearSolver->f_maxIter.setValue(5000);
        PCGlinearSolver->f_smallDenominatorThreshold.setValue(1e-25);
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////

    //<TO REMOVE>
    //VecDeriv& velocity = *mstate->getVecDeriv(lhId.index);
    //Data<VecDeriv>* dataVelocity = mstate->writeVecDeriv(lhId);
    //VecDeriv& velocity = *dataVelocity->beginEdit();
    helper::WriteAccessor<Data<VecDeriv> > dataVelocity = *mstate->write(core::VecDerivId::velocity());
    VecDeriv& velocity = dataVelocity.wref();

    VecDeriv velocity0 = velocity;
    //VecCoord& pos = *mstate->getX();
    helper::WriteAccessor<Data<VecCoord> > posData = *mstate->write(core::VecCoordId::position());
    VecCoord& pos = posData.wref();
    VecCoord pos0 = pos;

    for(unsigned int f = 0 ; f < nbNodes ; f++)
    {
        std::cout.precision(2);
        std::cout << "Precomputing constraint correction : " << std::fixed << (float)f/(float)nbNodes*100.0f << " %   " << '\xd';
        std::cout.flush();
        Deriv unitary_force;

        for (unsigned int i=0; i<dof_on_node; i++)
        {
            unitary_force.clear();
            unitary_force[i]=1.0;
            force[f] = unitary_force;

            velocity.clear();
            velocity.resize(nbNodes);

            if(f*dof_on_node+i <2 )
            {
                EulerSolver->f_verbose.setValue(true);
                EulerSolver->f_printLog.setValue(true);
                serr<<"getF : "<<force<<sendl;
            }

            if (linearSolver)
            {
                linearSolver->setSystemRHVector(rhId);
                linearSolver->setSystemLHVector(lhId);
                linearSolver->solveSystem();
            }

            if (linearSolver && f*dof_on_node+i == 0) linearSolver->freezeSystemMatrix(); // do not recompute the matrix for the rest of the precomputation

            if(f*dof_on_node+i < 2)
            {
                EulerSolver->f_verbose.setValue(false);
                EulerSolver->f_printLog.setValue(false);
                serr<<"getV : "<<velocity<<sendl;
            }
            for (unsigned int v=0; v<nbNodes; v++)
            {
                for (unsigned int j=0; j<dof_on_node; j++)
                {
                    internalData.MinvPtr->set(v*dof_on_node+j,f*dof_on_node+i,(Real)(velocity[v][j]*factInt));
                }
            }
        }
        unitary_force.clear();
        force[f] = unitary_force;
    }
    std::cout << "Precomputing constraint correction : " << std::fixed << 100.0f << " %   " << '\xd';
    std::cout.flush();

    ///////////////////////////////////////////////////////////////////////////////////////////////

    if (linearSolver) linearSolver->updateSystemMatrix(); // do not recompute the matrix for the rest of the precomputation

    ///////////////////////// RESET PARAMETERS AT THEIR PREVIOUS VALUE /////////////////////////////////
    // gravity is reset at its previous value
    this->getContext()->setGravityInWorld(gravity);

    if(CGlinearSolver)
    {
        CGlinearSolver->f_tolerance.setValue(buf_tolerance);
        CGlinearSolver->f_maxIter.setValue(buf_maxIter);
        CGlinearSolver->f_smallDenominatorThreshold.setValue(buf_threshold);
    }
    else if(PCGlinearSolver)
    {
        PCGlinearSolver->f_tolerance.setValue(buf_tolerance);
        PCGlinearSolver->f_maxIter.setValue(buf_maxIter);
        PCGlinearSolver->f_smallDenominatorThreshold.setValue(buf_threshold);
    }

    //Reset the velocity
    for (unsigned int i=0; i<velocity0.size(); i++) velocity[i]=velocity0[i];
    //Reset the position
    for (unsigned int i=0; i<pos0.size(); i++) pos[i]=pos0[i];

//         dataForce->endEdit();
//         dataVelocity->endEdit();

    mstate->vFree(lhId);
    mstate->vFree(rhId);

    usePrecond = true;
}

template<class TDataTypes>
void PrecomputedWarpPreconditioner<TDataTypes>::invert(TMatrix& M)
{
    if (usePrecond)
    {
        if (first)
        {
            first = false;
            if (this->currentGroup->systemMatrix==NULL) this->currentGroup->systemMatrix=&M;
            loadMatrix();
        }

        this->rotateConstraints();
    }
}

template<class TDataTypes>
void PrecomputedWarpPreconditioner<TDataTypes>::rotateConstraints()
{
    _rotate = true;
    if (! use_rotations.getValue()) return;

    unsigned systemSize3 = internalData.MinvPtr->colSize()/3;
    if (R.size() != systemSize3*9)
    {
        T.resize(internalData.MinvPtr->colSize());
        R.resize(systemSize3*9);
    }

    simulation::Node *node = dynamic_cast<simulation::Node *>(this->getContext());
    sofa::component::forcefield::TetrahedronFEMForceField<TDataTypes>* forceField = NULL;
    sofa::component::container::RotationFinder<TDataTypes>* rotationFinder = NULL;

    if (node != NULL)
    {
        forceField = node->get<component::forcefield::TetrahedronFEMForceField<TDataTypes> > ();
        if (forceField == NULL)
        {
            rotationFinder = node->get<component::container::RotationFinder<TDataTypes> > ();
            if (rotationFinder == NULL)
                sout << "No rotation defined : only defined for TetrahedronFEMForceField and RotationFinder!";

        }
    }

    Transformation Rotation;
    if (forceField != NULL)
    {
        for(unsigned int k = 0; k < systemSize3; k++)
        {
            forceField->getRotation(Rotation, k);
            for (int j=0; j<3; j++)
            {
                for (int i=0; i<3; i++)
                {
                    R[k*9+j*3+i] = (Real)Rotation[j][i];
                }
            }
        }
    }
    else if (rotationFinder != NULL)
    {
        const helper::vector<defaulttype::Mat<3,3,Real> > & rotations = rotationFinder->getRotations();
        for(unsigned int k = 0; k < systemSize3; k++)
        {
            Rotation = rotations[k];
            for (int j=0; j<3; j++)
            {
                for (int i=0; i<3; i++)
                {
                    R[k*9+j*3+i] = (Real)Rotation[j][i];
                }
            }
        }
    }
    else
    {
        serr << "No rotation defined : use Identity !!";
        for(unsigned int k = 0; k < systemSize3; k++)
        {
            R[k*9] = R[k*9+4] = R[k*9+8] = 1.0f;
            R[k*9+1] = R[k*9+2] = R[k*9+3] = R[k*9+5] = R[k*9+6] = R[k*9+7] = 0.0f;
        }
    }


}

template<class TDataTypes>
bool PrecomputedWarpPreconditioner<TDataTypes>::addJMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact)
{
    if (! _rotate) this->rotateConstraints();  //already rotate with Preconditionner
    _rotate = false;
    if (J->colSize() == 0) return true;

    if (SparseMatrix<double>* j = dynamic_cast<SparseMatrix<double>*>(J))
    {
        ComputeResult(result, *j, (float) fact);
        return true;
    }
    else if (SparseMatrix<float>* j = dynamic_cast<SparseMatrix<float>*>(J))
    {
        ComputeResult(result, *j, (float) fact);
        return true;
    }

    return false;
}

template<class TDataTypes> template<class JMatrix>
void PrecomputedWarpPreconditioner<TDataTypes>::ComputeResult(defaulttype::BaseMatrix * result,JMatrix& J, float fact)
{
    unsigned nl;
    internalData.JR.clear();
    internalData.JR.resize(J.rowSize(),J.colSize());

    if (use_rotations.getValue())
    {
        //compute JR = J * R
        nl = 0;
        for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != J.end(); jit1++)
        {
            int l = jit1->first;
            for (typename JMatrix::LElementConstIterator i1 = jit1->second.begin(); i1 != jit1->second.end();)
            {
                int c = i1->first;
                Real v0 = i1->second; i1++; Real v1 = i1->second; i1++; Real v2 = i1->second; i1++;
                internalData.JR.set(l,c+0,v0 * R[(c+0)*3+0] + v1 * R[(c+1)*3+0] + v2 * R[(c+2)*3+0] );
                internalData.JR.set(l,c+1,v0 * R[(c+0)*3+1] + v1 * R[(c+1)*3+1] + v2 * R[(c+2)*3+1] );
                internalData.JR.set(l,c+2,v0 * R[(c+0)*3+2] + v1 * R[(c+1)*3+2] + v2 * R[(c+2)*3+2] );
            }
            nl++;
        }
    }
    else
    {
        //compute JR = J * I
        nl = 0;
        for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != J.end(); jit1++)
        {
            int l = jit1->first;
            for (typename JMatrix::LElementConstIterator i1 = jit1->second.begin(); i1 != jit1->second.end();)
            {
                int c = i1->first;
                Real v0 = i1->second; i1++; Real v1 = i1->second; i1++; Real v2 = i1->second; i1++;
                internalData.JR.set(l,c+0,v0);
                internalData.JR.set(l,c+1,v1);
                internalData.JR.set(l,c+2,v2);
            }
            nl++;
        }
    }

    internalData.JRMinv.clear();
    internalData.JRMinv.resize(nl,internalData.MinvPtr->rowSize());

    //compute JRMinv = JR * Minv

    nl = 0;
    for (typename SparseMatrix<Real>::LineConstIterator jit1 = internalData.JR.begin(); jit1 != internalData.JR.end(); jit1++)
    {
        for (unsigned c = 0; c<internalData.MinvPtr->rowSize(); c++)
        {
            Real v = 0.0;
            for (typename SparseMatrix<Real>::LElementConstIterator i1 = jit1->second.begin(); i1 != jit1->second.end(); i1++)
            {
                v += internalData.MinvPtr->element(i1->first,c) * i1->second;
            }
            internalData.JRMinv.add(nl,c,v);
        }
        nl++;
    }

    //compute Result = JRMinv * (JR)t

    nl = 0;
    for (typename SparseMatrix<Real>::LineConstIterator jit1 = internalData.JR.begin(); jit1 != internalData.JR.end(); jit1++)
    {
        int l = jit1->first;
        for (typename SparseMatrix<Real>::LineConstIterator jit2 = internalData.JR.begin(); jit2 != internalData.JR.end(); jit2++)
        {
            int c = jit2->first;
            Real res = 0.0;
            for (typename SparseMatrix<Real>::LElementConstIterator i1 = jit2->second.begin(); i1 != jit2->second.end(); i1++)
            {
                res += internalData.JRMinv.element(nl,i1->first) * i1->second;
            }
            result->add(l,c,res*fact);
        }
        nl++;
    }
}

template<class TDataTypes>
void PrecomputedWarpPreconditioner<TDataTypes>::init()
{
    simulation::Node *node = dynamic_cast<simulation::Node *>(this->getContext());
    if (node != NULL) mstate = node->get<MState> ();
}

template<class TDataTypes>
void PrecomputedWarpPreconditioner<TDataTypes>::draw()
{
    if (! use_rotations.getValue()) return;
    if (draw_rotations_scale.getValue() <= 0.0) return;
    if (! this->getContext()->getShowBehaviorModels()) return;
    if (mstate==NULL) return;

    const VecCoord& x = *mstate->getX();

    if (R.size()!=x.size()*9) return;

    for (unsigned int i=0; i< x.size(); i++)
    {
        sofa::defaulttype::Matrix3 RotMat;

        for (int a=0; a<3; a++)
        {
            for (int b=0; b<3; b++)
            {
                RotMat[a][b] = R[i*9+a*3+b];
            }
        }

        sofa::defaulttype::Quat q;
        q.fromMatrix(RotMat);
        helper::gl::Axis::draw(DataTypes::getCPos(x[i]), q, this->draw_rotations_scale.getValue());
    }
}

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
