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
// Author: Hadrien Courtecuisse
//
// Copyright: See COPYING file that comes with this distribution

#ifndef SOFA_COMPONENT_LINEARSOLVER_PPRECOMPUTEDWARPPRECONDITIONER_INL
#define SOFA_COMPONENT_LINEARSOLVER_PPRECOMPUTEDWARPPRECONDITIONER_INL

#include "PrecomputedWarpPreconditioner.h"
//#include <SofaDenseSolver/NewMatMatrix.h>
#include <SofaBaseLinearSolver/FullMatrix.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <iostream>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <math.h>
#include <sofa/helper/system/thread/CTime.h>
#include <SofaSimpleFem/TetrahedronFEMForceField.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <SofaBaseLinearSolver/MatrixLinearSolver.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/core/behavior/RotationFinder.h>
#include <sofa/core/behavior/LinearSolver.h>

#include <sofa/helper/gl/Axis.h>
#include <sofa/helper/Quater.h>

#include <SofaImplicitOdeSolver/EulerImplicitSolver.h>
#include <SofaBaseLinearSolver/CGLinearSolver.h>

#ifdef SOFA_HAVE_CSPARSE
#include <SofaSparseSolver/SparseCholeskySolver.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>
#else
#include <SofaBaseLinearSolver/CholeskySolver.h>
#endif


namespace sofa
{

namespace component
{

namespace linearsolver
{

template<class TDataTypes>
PrecomputedWarpPreconditioner<TDataTypes>::PrecomputedWarpPreconditioner()
    : jmjt_twostep( initData(&jmjt_twostep,true,"jmjt_twostep","Use two step algorithm to compute JMinvJt") )
    , f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , use_file( initData(&use_file,true,"use_file","Dump system matrix in a file") )
    , share_matrix( initData(&share_matrix,true,"share_matrix","Share the compliance matrix in memory if they are related to the same file (WARNING: might require to reload Sofa when opening a new scene...)") )
    , solverName(initData(&solverName, std::string(""), "solverName", "Name of the solver to use to precompute the first matrix"))
    , use_rotations( initData(&use_rotations,true,"use_rotations","Use Rotations around the preconditioner") )
    , draw_rotations_scale( initData(&draw_rotations_scale,0.0,"draw_rotations_scale","Scale rotations in draw function") )
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
        first = false;
        init_mFact = mparams->mFactor();
        init_bFact = mparams->bFactor();
        init_kFact = mparams->kFactor();
        Inherit::setSystemMBKMatrix(mparams);
        loadMatrix(*this->currentGroup->systemMatrix);
    }

    this->currentGroup->needInvert = usePrecond;
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
            while (l < matrixSize)
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
            while (l < matrixSize)
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
void PrecomputedWarpPreconditioner<TDataTypes>::loadMatrix(TMatrix& M)
{
    dof_on_node = Deriv::size();
    systemSize = M.rowSize();
    nb_dofs = systemSize/dof_on_node;
    matrixSize = nb_dofs*dof_on_node;

    dt = this->getContext()->getDt();


    sofa::component::odesolver::EulerImplicitSolver* EulerSolver;
    this->getContext()->get(EulerSolver);
    factInt = 1.0; // christian : it is not a compliance... but an admittance that is computed !
    if (EulerSolver) factInt = EulerSolver->getPositionIntegrationFactor(); // here, we compute a compliance

    std::stringstream ss;
    ss << this->getContext()->getName() << "-" << systemSize << "-" << dt << ((sizeof(Real)==sizeof(float)) ? ".compf" : ".comp");
    std::string fname = ss.str();

    if (share_matrix.getValue()) internalData.setMinv(internalData.getSharedMatrix(fname));

    if (share_matrix.getValue() && internalData.MinvPtr->rowSize() == (defaulttype::BaseMatrix::Index)systemSize)
    {
        msg_info("PrecomputedWarpPreconditioner") << "shared matrix : " << fname << " is already built." ;
    }
    else
    {
        internalData.MinvPtr->resize(matrixSize,matrixSize);

        std::ifstream compFileIn(fname.c_str(), std::ifstream::binary);

        if(compFileIn.good() && use_file.getValue())
        {
            msg_info("PrecomputedWarpPreconditioner") << "file open : " << fname << " compliance being loaded" ;
            internalData.readMinvFomFile(compFileIn);
            compFileIn.close();
        }
        else
        {
            msg_info("PrecomputedWarpPreconditioner") << "Precompute : " << fname << " compliance." ;
            if (solverName.getValue().empty()) loadMatrixWithCSparse(M);
            else loadMatrixWithSolver();

            if (use_file.getValue())
            {
                std::ofstream compFileOut(fname.c_str(), std::fstream::out | std::fstream::binary);
                internalData.writeMinvFomFile(compFileOut);
                compFileOut.close();
            }
            compFileIn.close();
        }

        for (unsigned int j=0; j<matrixSize; j++)
        {
            Real * minvVal = (*internalData.MinvPtr)[j];
            for (unsigned i=0; i<matrixSize; i++) minvVal[i] /= (Real)factInt;
        }
    }

    R.resize(nb_dofs*9);
    T.resize(matrixSize);
    for(unsigned int k = 0; k < nb_dofs; k++)
    {
        R[k*9] = R[k*9+4] = R[k*9+8] = 1.0f;
        R[k*9+1] = R[k*9+2] = R[k*9+3] = R[k*9+5] = R[k*9+6] = R[k*9+7] = 0.0f;
    }
}

#ifdef SOFA_HAVE_CSPARSE
template<class TDataTypes>
void PrecomputedWarpPreconditioner<TDataTypes>::loadMatrixWithCSparse(TMatrix& M)
{
    msg_info("PrecomputedWarpPreconditioner") << "Compute the initial invert matrix with CS_PARSE" ;

    FullVector<Real> r;
    FullVector<Real> b;

    r.resize(systemSize);
    b.resize(systemSize);
    for (unsigned int j=0; j<systemSize; j++) b.set(j,0.0);

    SparseCholeskySolver<CompressedRowSparseMatrix<Real>, FullVector<Real> > solver;

    msg_info("PrecomputedWarpPreconditioner") << "Precomputing constraint correction LU decomposition " ;
    solver.invert(M);

    for (unsigned int j=0; j<nb_dofs; j++)
    {
        unsigned pid_j;
        pid_j = j;

        for (unsigned d=0; d<dof_on_node; d++)
        {
            sout.precision(2);
            sout << "Precomputing constraint correction : " << std::fixed << (float)(j*dof_on_node+d)*100.0f/(float)(nb_dofs*dof_on_node) << " %   " << '\xd';
            sout << sendl;

            b.set(pid_j*dof_on_node+d,1.0);
            solver.solve(M,r,b);

            Real * minvVal = (*internalData.MinvPtr)[j*dof_on_node+d];
            for (unsigned int i=0; i<nb_dofs; i++)
            {
                unsigned pid_i;
                pid_i=i;

                for (unsigned c=0; c<dof_on_node; c++)
                {
                    minvVal[i*dof_on_node+c] = (Real)(r.element(pid_i*dof_on_node+c)*factInt);
                }
            }

            b.set(pid_j*dof_on_node+d,0.0);
        }
    }

    sout << "Precomputing constraint correction : " << std::fixed << 100.0f << " %" << sendl;
}
#else
template<class TDataTypes>
void PrecomputedWarpPreconditioner<TDataTypes>::loadMatrixWithCSparse(TMatrix& /*M*/)
{
    msg_warning("PrecomputedWarpPreconditioner") << "you don't have CS_parse CG will be use, (if also can specify solverName to accelerate the precomputation" ;
    loadMatrixWithSolver();
}
#endif

template<class TDataTypes>
void PrecomputedWarpPreconditioner<TDataTypes>::loadMatrixWithSolver()
{
    usePrecond = false;//Don'Use precond during precomputing

    msg_info("PrecomputedWarpPreconditioner") << "Compute the initial invert matrix with solver" ;

    if (mstate==NULL)
    {
        serr << "PrecomputedWarpPreconditioner can't find Mstate" << sendl;
        return;
    }

    std::stringstream ss;
    //ss << this->getContext()->getName() << "_CPP.comp";
    ss << this->getContext()->getName() << "-" << systemSize << "-" << dt << ((sizeof(Real)==sizeof(float)) ? ".compf" : ".comp");
    std::ifstream compFileIn(ss.str().c_str(), std::ifstream::binary);

    sofa::component::odesolver::EulerImplicitSolver* EulerSolver;
    this->getContext()->get(EulerSolver);

    // for the initial computation, the gravity has to be put at 0
    const sofa::defaulttype::Vec3d gravity = this->getContext()->getGravity();
    const sofa::defaulttype::Vec3d gravity_zero(0.0,0.0,0.0);
    this->getContext()->setGravity(gravity_zero);

    CGLinearSolver<GraphScatteredMatrix,GraphScatteredVector>* CGlinearSolver;
    core::behavior::LinearSolver* linearSolver;

    if (solverName.getValue().empty())
    {
        this->getContext()->get(CGlinearSolver);
        this->getContext()->get(linearSolver);
    }
    else
    {
        core::objectmodel::BaseObject* ptr = NULL;
        this->getContext()->get(ptr, solverName.getValue());
        CGlinearSolver = dynamic_cast<CGLinearSolver<GraphScatteredMatrix,GraphScatteredVector>*>(ptr);
        linearSolver = ptr->toLinearSolver();
    }

    if(EulerSolver && CGlinearSolver)
        sout << "use EulerImplicitSolver &  CGLinearSolver" << sendl;
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
    sofa::core::VecDerivId lhId = core::VecDerivId::velocity();
    sofa::core::VecDerivId rhId = core::VecDerivId::force();


    mstate->vAvail(core::ExecParams::defaultInstance(), lhId);
    mstate->vAlloc(core::ExecParams::defaultInstance(), lhId);
    mstate->vAvail(core::ExecParams::defaultInstance(), rhId);
    mstate->vAlloc(core::ExecParams::defaultInstance(), rhId);
    msg_info("PrecomputedWarpPreconditioner") << "System: (" << init_mFact << " * M + " << init_bFact << " * B + " << init_kFact << " * K) " << lhId << " = " << rhId ;
    if (linearSolver)
    {
        msg_info("PrecomputedWarpPreconditioner") << "System Init Solver: " << linearSolver->getName() << " (" << linearSolver->getClassName() << ")" ;
        core::MechanicalParams mparams;
        mparams.setMFactor(init_mFact);
        mparams.setBFactor(init_bFact);
        mparams.setKFactor(init_kFact);
        linearSolver->setSystemMBKMatrix(&mparams);
    }

    helper::WriteAccessor<Data<VecDeriv> > dataForce = *mstate->write(core::VecDerivId::externalForce());
    VecDeriv& force = dataForce.wref();

    force.clear();
    force.resize(nb_dofs);

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
    ///////////////////////////////////////////////////////////////////////////////////////////////

    helper::WriteAccessor<Data<VecDeriv> > dataVelocity = *mstate->write(core::VecDerivId::velocity());
    VecDeriv& velocity = dataVelocity.wref();

    VecDeriv velocity0 = velocity;
    helper::WriteAccessor<Data<VecCoord> > posData = *mstate->write(core::VecCoordId::position());
    VecCoord& pos = posData.wref();
    VecCoord pos0 = pos;

    for(unsigned int j = 0 ; j < nb_dofs ; j++)
    {
        Deriv unitary_force;

        int pid_j;
        pid_j = j;

        for (unsigned int d=0; d<dof_on_node; d++)
        {
            std::stringstream tmp;
            tmp.precision(2);
            tmp << "Precomputing constraint correction : " << std::fixed << (float)(j*dof_on_node+d)*100.0f/(float)(nb_dofs*dof_on_node) << " %   " << '\xd';
            msg_info("PrecomputedWarpPreconditioner") << tmp.str() ;

            unitary_force.clear();
            unitary_force[d]=1.0;
            force[pid_j] = unitary_force;

            velocity.clear();
            velocity.resize(nb_dofs);

            if(pid_j*dof_on_node+d <2 )
            {
                EulerSolver->f_verbose.setValue(true);
                EulerSolver->f_printLog.setValue(true);
                msg_info() <<"getF : "<<force;
            }

            if (linearSolver)
            {
                linearSolver->setSystemRHVector(rhId);
                linearSolver->setSystemLHVector(lhId);
                linearSolver->solveSystem();
            }

            if (linearSolver && pid_j*dof_on_node+d == 0) linearSolver->freezeSystemMatrix(); // do not recompute the matrix for the rest of the precomputation

            if(pid_j*dof_on_node+d < 2)
            {
                EulerSolver->f_verbose.setValue(false);
                EulerSolver->f_printLog.setValue(false);
                msg_info()<<"getV : "<<velocity;
            }

            Real * minvVal = (*internalData.MinvPtr)[j*dof_on_node+d];
            for (unsigned int i=0; i<nb_dofs; i++)
            {
                unsigned pid_i;
                pid_i=i;

                for (unsigned int c=0; c<dof_on_node; c++)
                {
                    minvVal[i*dof_on_node+c] = (Real) (velocity[pid_i][c]*factInt);
                }
            }
        }

        unitary_force.clear();
        force[pid_j] = unitary_force;
    }
    msg_info("PrecomputedWarpPreconditioner") << "Precomputing constraint correction : " << std::fixed << 100.0f << " %" ;
    ///////////////////////////////////////////////////////////////////////////////////////////////

    if (linearSolver) linearSolver->freezeSystemMatrix(); // do not recompute the matrix for the rest of the precomputation

    ///////////////////////// RESET PARAMETERS AT THEIR PREVIOUS VALUE /////////////////////////////////
    // gravity is reset at its previous value
    this->getContext()->setGravity(gravity);

    if(CGlinearSolver)
    {
        CGlinearSolver->f_tolerance.setValue(buf_tolerance);
        CGlinearSolver->f_maxIter.setValue(buf_maxIter);
        CGlinearSolver->f_smallDenominatorThreshold.setValue(buf_threshold);
    }

    //Reset the velocity
    for (unsigned int i=0; i<velocity0.size(); i++) velocity[i]=velocity0[i];
    //Reset the position
    for (unsigned int i=0; i<pos0.size(); i++) pos[i]=pos0[i];

    mstate->vFree(core::ExecParams::defaultInstance(), lhId);
    mstate->vFree(core::ExecParams::defaultInstance(), rhId);

    usePrecond = true;
}

template<class TDataTypes>
void PrecomputedWarpPreconditioner<TDataTypes>::invert(TMatrix& M)
{
    if (first)
    {
        first = false;
        loadMatrix(M);
    }
    if (usePrecond) this->rotateConstraints();
}

template<class TDataTypes>
void PrecomputedWarpPreconditioner<TDataTypes>::rotateConstraints()
{
    _rotate = true;
    if (! use_rotations.getValue()) return;

    simulation::Node *node = dynamic_cast<simulation::Node *>(this->getContext());
    sofa::component::forcefield::TetrahedronFEMForceField<TDataTypes>* forceField = NULL;
    sofa::core::behavior::RotationFinder<TDataTypes>* rotationFinder = NULL;

    if (node != NULL)
    {
        forceField = node->get<component::forcefield::TetrahedronFEMForceField<TDataTypes> > ();
        if (forceField == NULL)
        {
            rotationFinder = node->get< sofa::core::behavior::RotationFinder<TDataTypes> > ();
            if (rotationFinder == NULL)
                sout << "No rotation defined : only defined for TetrahedronFEMForceField and RotationFinder!";

        }
    }

    Transformation Rotation;
    if (forceField != NULL)
    {
        for(unsigned int k = 0; k < nb_dofs; k++)
        {
            int pid;
            pid = k;

            forceField->getRotation(Rotation, pid);
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
        for(unsigned int k = 0; k < nb_dofs; k++)
        {
            int pid;
            pid = k;

            Rotation = rotations[pid];
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
        for(unsigned int k = 0; k < nb_dofs; k++)
        {
            R[k*9] = R[k*9+4] = R[k*9+8] = 1.0f;
            R[k*9+1] = R[k*9+2] = R[k*9+3] = R[k*9+5] = R[k*9+6] = R[k*9+7] = 0.0f;
        }
    }


}

template<class TDataTypes> template<class JMatrix>
void PrecomputedWarpPreconditioner<TDataTypes>::computeActiveDofs(JMatrix& J)
{
    isActiveDofs.clear();
    isActiveDofs.resize(systemSize);

    //compute JR = J * R
    for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != J.end(); jit1++)
    {
        for (typename JMatrix::LElementConstIterator i1 = jit1->second.begin(); i1 != jit1->second.end(); i1++)
        {
            isActiveDofs[i1->first] = true;
        }
    }

    internalData.invActiveDofs.clear();
    internalData.invActiveDofs.resize(systemSize);
    internalData.idActiveDofs.clear();

    for (unsigned c=0; c<systemSize; c++)
    {
        if (isActiveDofs[c])
        {
            internalData.invActiveDofs[c] = internalData.idActiveDofs.size();
            internalData.idActiveDofs.push_back(c);
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
        computeActiveDofs(*j);
        ComputeResult(result, *j, (float) fact);
        return true;
    }
    else if (SparseMatrix<float>* j = dynamic_cast<SparseMatrix<float>*>(J))
    {
        computeActiveDofs(*j);
        ComputeResult(result, *j, (float) fact);
        return true;
    }

    return false;
}

template<class TDataTypes> template<class JMatrix>
void PrecomputedWarpPreconditioner<TDataTypes>::ComputeResult(defaulttype::BaseMatrix * result,JMatrix& J, float fact)
{
    unsigned nl = 0;
    internalData.JRMinv.clear();
    internalData.JRMinv.resize(J.rowSize(),internalData.idActiveDofs.size());

    if (use_rotations.getValue())
    {
        internalData.JR.clear();
        internalData.JR.resize(J.rowSize(),J.colSize());

        //compute JR = J * R
        for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != J.end(); jit1++)
        {
            int l = jit1->first;
            for (typename JMatrix::LElementConstIterator i1 = jit1->second.begin(); i1 != jit1->second.end();)
            {
                int c = i1->first;
                Real v0 = (Real)i1->second; i1++; Real v1 = (Real)i1->second; i1++; Real v2 = (Real)i1->second; i1++;
                internalData.JR.set(l,c+0,v0 * R[(c+0)*3+0] + v1 * R[(c+1)*3+0] + v2 * R[(c+2)*3+0] );
                internalData.JR.set(l,c+1,v0 * R[(c+0)*3+1] + v1 * R[(c+1)*3+1] + v2 * R[(c+2)*3+1] );
                internalData.JR.set(l,c+2,v0 * R[(c+0)*3+2] + v1 * R[(c+1)*3+2] + v2 * R[(c+2)*3+2] );
            }
        }

        nl=0;
        for (typename SparseMatrix<Real>::LineConstIterator jit1 = internalData.JR.begin(); jit1 != internalData.JR.end(); jit1++)
        {
            for (unsigned c = 0; c<internalData.idActiveDofs.size(); c++)
            {
                int col = internalData.idActiveDofs[c];
                Real v = (Real)0.0;
                for (typename SparseMatrix<Real>::LElementConstIterator i1 = jit1->second.begin(); i1 != jit1->second.end(); i1++)
                {
                    v += (Real)(internalData.MinvPtr->element(i1->first,col) * i1->second);
                }
                internalData.JRMinv.set(nl,c,v);
            }
            nl++;
        }
    }
    else
    {
        nl=0;
        for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != J.end(); jit1++)
        {
            for (unsigned c = 0; c<internalData.idActiveDofs.size(); c++)
            {
                int col = internalData.idActiveDofs[c];
                Real v = (Real)0.0;
                for (typename JMatrix::LElementConstIterator i1 = jit1->second.begin(); i1 != jit1->second.end(); i1++)
                {
                    v += (Real)(internalData.MinvPtr->element(i1->first,col) * i1->second);
                }
                internalData.JRMinv.set(nl,c,v);
            }
            nl++;
        }
    }
    //compute Result = JRMinv * (JR)t
    nl = 0;
    for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != J.end(); jit1++)
    {
        int row = jit1->first;
        for (typename JMatrix::LineConstIterator jit2 = J.begin(); jit2 != J.end(); jit2++)
        {
            int col = jit2->first;
            Real res = (Real)0.0;
            for (typename JMatrix::LElementConstIterator i1 = jit2->second.begin(); i1 != jit2->second.end(); i1++)
            {
                res += (Real)(internalData.JRMinv.element(nl,internalData.invActiveDofs[i1->first]) * i1->second);
            }
            result->add(row,col,res*fact);
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
void PrecomputedWarpPreconditioner<TDataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (! use_rotations.getValue()) return;
    if (draw_rotations_scale.getValue() <= 0.0) return;
    if (! vparams->displayFlags().getShowBehaviorModels()) return;
    if (mstate==NULL) return;

    const VecCoord& x = mstate->read(core::ConstVecCoordId::position())->getValue();

    for (unsigned int i=0; i< nb_dofs; i++)
    {
        sofa::defaulttype::Matrix3 RotMat;

        for (int a=0; a<3; a++)
        {
            for (int b=0; b<3; b++)
            {
                RotMat[a][b] = R[i*9+a*3+b];
            }
        }

        int pid = i;

        sofa::defaulttype::Quat q;
        q.fromMatrix(RotMat);
        helper::gl::Axis::draw(DataTypes::getCPos(x[pid]), q, this->draw_rotations_scale.getValue());
    }
#endif /* SOFA_NO_OPENGL */
}

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
