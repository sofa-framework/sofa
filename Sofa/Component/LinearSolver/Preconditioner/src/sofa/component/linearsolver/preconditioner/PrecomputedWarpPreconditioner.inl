/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/component/linearsolver/preconditioner/PrecomputedWarpPreconditioner.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <cmath>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.inl>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/core/behavior/RotationFinder.h>
#include <sofa/core/behavior/LinearSolver.h>

#include <sofa/type/Quat.h>

#include <sofa/component/odesolver/backward/EulerImplicitSolver.h>
#include <sofa/component/linearsolver/iterative/CGLinearSolver.h>
#include <sofa/component/linearsolver/direct/EigenSimplicialLLT.h>
#include <sofa/component/linearsolver/direct/EigenDirectSparseSolver.inl>

#include <sofa/simulation/Node.h>

#include <sofa/component/linearsolver/preconditioner/PrecomputedMatrixSystem.h>


namespace sofa::component::linearsolver::preconditioner
{

template<class TDataTypes>
PrecomputedWarpPreconditioner<TDataTypes>::PrecomputedWarpPreconditioner()
    : d_jmjt_twostep(initData(&d_jmjt_twostep, true, "jmjt_twostep", "Use two step algorithm to compute JMinvJt") )
    , d_use_file(initData(&d_use_file, true, "use_file", "Dump system matrix in a file") )
    , d_share_matrix(initData(&d_share_matrix, true, "share_matrix", "Share the compliance matrix in memory if they are related to the same file (WARNING: might require to reload Sofa when opening a new scene...)") )
    , l_linearSolver(initLink("linearSolver", "Link towards the linear solver used to precompute the first matrix"))
    , d_use_rotations(initData(&d_use_rotations, true, "use_rotations", "Use Rotations around the preconditioner") )
    , d_draw_rotations_scale(initData(&d_draw_rotations_scale, 0.0, "draw_rotations_scale", "Scale rotations in draw function") )
{
    first = true;
    _rotate = false;
    usePrecond = true;

    jmjt_twostep.setOriginalData(&d_jmjt_twostep);
    use_file.setOriginalData(&d_use_file);
    share_matrix.setOriginalData(&d_share_matrix);
    use_rotations.setOriginalData(&d_use_rotations);
    draw_rotations_scale.setOriginalData(&d_draw_rotations_scale);

}

template <class TDataTypes>
void PrecomputedWarpPreconditioner<TDataTypes>::checkLinearSystem()
{
    if (!this->l_linearSystem)
    {
        auto* matrixLinearSystem=this->getContext()->template get<PrecomputedMatrixSystem<TMatrix, TVector> >();
        if(!matrixLinearSystem)
        {
            this->template createDefaultLinearSystem<PrecomputedMatrixSystem<TMatrix, TVector> >();
        }
    }
}

template<class TDataTypes>
void PrecomputedWarpPreconditioner<TDataTypes>::setSystemMBKMatrix(const core::MechanicalParams* mparams)
{
    // Update the matrix only the first time
    if (first)
    {
        first = false;
        init_mFact = mparams->mFactor();
        init_bFact = sofa::core::mechanicalparams::bFactor(mparams);
        init_kFact = mparams->kFactor();
        Inherit::setSystemMBKMatrix(mparams);
        loadMatrix(*this->getSystemMatrix());
    }

    this->linearSystem.needInvert = usePrecond;
}

//Solve x = R * M^-1 * R^t * b
template<class TDataTypes>
void PrecomputedWarpPreconditioner<TDataTypes>::solve (TMatrix& /*M*/, TVector& z, TVector& r)
{
    if (usePrecond)
    {
        if (d_use_rotations.getValue())
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


    sofa::component::odesolver::backward::EulerImplicitSolver* EulerSolver;
    this->getContext()->get(EulerSolver);
    factInt = 1.0; // christian : it is not a compliance... but an admittance that is computed !
    if (EulerSolver) factInt = EulerSolver->getPositionIntegrationFactor(); // here, we compute a compliance

    std::stringstream ss;
    ss << this->getContext()->getName() << "-" << systemSize << "-" << dt << ((sizeof(Real)==sizeof(float)) ? ".compf" : ".comp");
    std::string fname = ss.str();

    if (d_share_matrix.getValue()) internalData.setMinv(internalData.getSharedMatrix(fname));

    if (d_share_matrix.getValue() && internalData.MinvPtr->rowSize() == (linearalgebra::BaseMatrix::Index)systemSize)
    {
        msg_info() << "shared matrix : " << fname << " is already built." ;
    }
    else
    {
        internalData.MinvPtr->resize(matrixSize,matrixSize);

        std::ifstream compFileIn(fname.c_str(), std::ifstream::binary);

        if(compFileIn.good() && d_use_file.getValue())
        {
            msg_info() << "file open : " << fname << " compliance being loaded" ;
            internalData.readMinvFomFile(compFileIn);
            compFileIn.close();
        }
        else
        {
            msg_info() << "Precompute : " << fname << " compliance.";
            if (l_linearSolver.empty())
            {
                loadMatrixWithCholeskyDecomposition(M);
            }
            else
            {
                loadMatrixWithSolver();
            }

            if (d_use_file.getValue())
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

template<class TDataTypes>
void PrecomputedWarpPreconditioner<TDataTypes>::loadMatrixWithCholeskyDecomposition(TMatrix& M)
{
    msg_info() << "Compute the initial invert matrix with CS_PARSE" ;

    using namespace sofa::linearalgebra;

    FullVector<Real> r;
    FullVector<Real> b;

    r.resize(systemSize);
    b.resize(systemSize);
    for (unsigned int j=0; j<systemSize; j++) b.set(j,0.0);

    direct::EigenSimplicialLLT<Real> solver;

    msg_info() << "Precomputing constraint correction LU decomposition " ;
    solver.invert(M);

    std::stringstream tmpStr;

    for (unsigned int j=0; j<nb_dofs; j++)
    {
        unsigned pid_j;
        pid_j = j;

        for (unsigned d=0; d<dof_on_node; d++)
        {
            tmpStr.precision(2);
            tmpStr << "Precomputing constraint correction : " << std::fixed << (float)(j*dof_on_node+d)*100.0f/(float)(nb_dofs*dof_on_node) << " %   " << '\xd'
                   << "\n";

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

    tmpStr << "Precomputing constraint correction : " << std::fixed << 100.0f << " %" ;
    msg_info() << tmpStr.str();
}

template<class TDataTypes>
void PrecomputedWarpPreconditioner<TDataTypes>::loadMatrixWithSolver()
{
    usePrecond = false;//Don'Use precond during precomputing

    msg_info() << "Compute the initial invert matrix with solver" ;

    if (mstate==nullptr)
    {
        msg_error() << "PrecomputedWarpPreconditioner can't find Mstate";
        return;
    }

    std::stringstream ss;
    //ss << this->getContext()->getName() << "_CPP.comp";
    ss << this->getContext()->getName() << "-" << systemSize << "-" << dt << ((sizeof(Real)==sizeof(float)) ? ".compf" : ".comp");
    std::ifstream compFileIn(ss.str().c_str(), std::ifstream::binary);

    sofa::component::odesolver::backward::EulerImplicitSolver* EulerSolver;
    this->getContext()->get(EulerSolver);

    // for the initial computation, the gravity has to be put at 0
    const sofa::type::Vec3 gravity = this->getContext()->getGravity();
    static constexpr sofa::type::Vec3 gravity_zero(0_sreal, 0_sreal, 0_sreal);
    this->getContext()->setGravity(gravity_zero);

    component::linearsolver::iterative::CGLinearSolver<GraphScatteredMatrix,GraphScatteredVector>* CGlinearSolver;
    core::behavior::LinearSolver* linearSolver;

    if (l_linearSolver.get() == nullptr)
    {
        msg_error() << "No LinearSolver component found at path: " << l_linearSolver.getLinkedPath();
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    msg_info() << "LinearSolver path used: '" << l_linearSolver.getLinkedPath() << "'";

    core::objectmodel::BaseObject* ptr = l_linearSolver.get();
    CGlinearSolver = dynamic_cast<component::linearsolver::iterative::CGLinearSolver<GraphScatteredMatrix,GraphScatteredVector>*>(ptr);
    linearSolver = ptr->toLinearSolver();


    if(EulerSolver && CGlinearSolver) {
        msg_info() << "use EulerImplicitSolver &  CGLinearSolver";
    } else if(EulerSolver && linearSolver) {
        msg_info() << "use EulerImplicitSolver &  LinearSolver";
    } else if(EulerSolver) {
        msg_info() << "use EulerImplicitSolver";
    } else {
        msg_error() << "PrecomputedContactCorrection must be associated with EulerImplicitSolver+LinearSolver for the precomputation\nNo Precomputation";
        return;
    }
    sofa::core::VecDerivId lhId = core::vec_id::write_access::velocity;
    sofa::core::VecDerivId rhId = core::vec_id::write_access::force;


    mstate->vAvail(core::execparams::defaultInstance(), lhId);
    mstate->vAlloc(core::execparams::defaultInstance(), lhId);
    mstate->vAvail(core::execparams::defaultInstance(), rhId);
    mstate->vAlloc(core::execparams::defaultInstance(), rhId);
    msg_info() << "System: (" << init_mFact << " * M + " << init_bFact << " * B + " << init_kFact << " * K) " << lhId << " = " << rhId ;
    if (linearSolver)
    {
        msg_info() << "System Init Solver: " << linearSolver->getName() << " (" << linearSolver->getClassName() << ")" ;
        core::MechanicalParams mparams;
        mparams.setMFactor(init_mFact);
        mparams.setBFactor(init_bFact);
        mparams.setKFactor(init_kFact);
        linearSolver->setSystemMBKMatrix(&mparams);
    }

    helper::WriteAccessor<Data<VecDeriv> > dataForce = *mstate->write(core::vec_id::write_access::externalForce);
    VecDeriv& force = dataForce.wref();

    force.clear();
    force.resize(nb_dofs);

    ///////////////////////// CHANGE THE PARAMETERS OF THE SOLVER /////////////////////////////////
    Real buf_tolerance=0, buf_threshold=0;
    unsigned int buf_maxIter=0;
    if(CGlinearSolver)
    {
        buf_tolerance = CGlinearSolver->d_tolerance.getValue();
        buf_maxIter   = CGlinearSolver->d_maxIter.getValue();
        buf_threshold = CGlinearSolver->d_smallDenominatorThreshold.getValue();
        CGlinearSolver->d_tolerance.setValue(Real(1e-35));
        CGlinearSolver->d_maxIter.setValue(5000u);
        CGlinearSolver->d_smallDenominatorThreshold.setValue(Real(1e-25));
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////

    helper::WriteAccessor<Data<VecDeriv> > dataVelocity = *mstate->write(core::vec_id::write_access::velocity);
    VecDeriv& velocity = dataVelocity.wref();

    VecDeriv velocity0 = velocity;
    helper::WriteAccessor<Data<VecCoord> > posData = *mstate->write(core::vec_id::write_access::position);
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
            msg_info() << tmp.str() ;

            unitary_force.clear();
            unitary_force[d]=1.0;
            force[pid_j] = unitary_force;

            velocity.clear();
            velocity.resize(nb_dofs);

            if(pid_j*dof_on_node+d <2 )
            {
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
    msg_info() << "Precomputing constraint correction : " << std::fixed << 100.0f << " %" ;
    ///////////////////////////////////////////////////////////////////////////////////////////////

    if (linearSolver) linearSolver->freezeSystemMatrix(); // do not recompute the matrix for the rest of the precomputation

    ///////////////////////// RESET PARAMETERS AT THEIR PREVIOUS VALUE /////////////////////////////////
    // gravity is reset at its previous value
    this->getContext()->setGravity(gravity);

    if(CGlinearSolver)
    {
        CGlinearSolver->d_tolerance.setValue(buf_tolerance);
        CGlinearSolver->d_maxIter.setValue(buf_maxIter);
        CGlinearSolver->d_smallDenominatorThreshold.setValue(buf_threshold);
    }

    //Reset the velocity
    for (unsigned int i=0; i<velocity0.size(); i++) velocity[i]=velocity0[i];
    //Reset the position
    for (unsigned int i=0; i<pos0.size(); i++) pos[i]=pos0[i];

    mstate->vFree(core::execparams::defaultInstance(), lhId);
    mstate->vFree(core::execparams::defaultInstance(), rhId);

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
    if (! d_use_rotations.getValue()) return;

    const simulation::Node *node = dynamic_cast<simulation::Node *>(this->getContext());
    sofa::core::behavior::RotationFinder<TDataTypes>* rotationFinder = nullptr;

    if (node != nullptr)
    {
        rotationFinder = node->get< sofa::core::behavior::RotationFinder<TDataTypes> > ();
        msg_warning_when(rotationFinder == nullptr) << "No rotation defined : only applicable for components implementing RotationFinder!";
    }

    Transformation Rotation;
    if (rotationFinder != nullptr)
    {
        const type::vector<type::Mat<3,3,Real> > & rotations = rotationFinder->getRotations();
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
        msg_error() << "No rotation defined : use Identity !!";
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
bool PrecomputedWarpPreconditioner<TDataTypes>::addJMInvJt(linearalgebra::BaseMatrix* result, linearalgebra::BaseMatrix* J, SReal fact)
{
    if (! _rotate) this->rotateConstraints();  //already rotate with Preconditionner
    _rotate = false;
    if (J->colSize() == 0) return true;

    if (linearalgebra::SparseMatrix<double>* j = dynamic_cast<linearalgebra::SparseMatrix<double>*>(J))
    {
        computeActiveDofs(*j);
        ComputeResult(result, *j, (float) fact);
        return true;
    }
    else if (linearalgebra::SparseMatrix<float>* j = dynamic_cast<linearalgebra::SparseMatrix<float>*>(J))
    {
        computeActiveDofs(*j);
        ComputeResult(result, *j, (float) fact);
        return true;
    }

    return false;
}

template<class TDataTypes> template<class JMatrix>
void PrecomputedWarpPreconditioner<TDataTypes>::ComputeResult(linearalgebra::BaseMatrix * result,JMatrix& J, float fact)
{
    unsigned nl = 0;
    internalData.JRMinv.clear();
    internalData.JRMinv.resize(J.rowSize(),internalData.idActiveDofs.size());

    if (d_use_rotations.getValue())
    {
        internalData.JR.clear();
        internalData.JR.resize(J.rowSize(),J.colSize());

        //compute JR = J * R
        for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != J.end(); jit1++)
        {
            int l = jit1->first;
            for (typename JMatrix::LElementConstIterator i1 = jit1->second.begin(); i1 != jit1->second.end();)
            {
                const int c = i1->first;
                Real v0 = (Real)i1->second; i1++; Real v1 = (Real)i1->second; i1++; Real v2 = (Real)i1->second; i1++;
                internalData.JR.set(l,c+0,v0 * R[(c+0)*3+0] + v1 * R[(c+1)*3+0] + v2 * R[(c+2)*3+0] );
                internalData.JR.set(l,c+1,v0 * R[(c+0)*3+1] + v1 * R[(c+1)*3+1] + v2 * R[(c+2)*3+1] );
                internalData.JR.set(l,c+2,v0 * R[(c+0)*3+2] + v1 * R[(c+1)*3+2] + v2 * R[(c+2)*3+2] );
            }
        }

        nl=0;
        for (typename linearalgebra::SparseMatrix<Real>::LineConstIterator jit1 = internalData.JR.begin(); jit1 != internalData.JR.end(); jit1++)
        {
            for (unsigned c = 0; c<internalData.idActiveDofs.size(); c++)
            {
                int col = internalData.idActiveDofs[c];
                Real v = (Real)0.0;
                for (typename linearalgebra::SparseMatrix<Real>::LElementConstIterator i1 = jit1->second.begin(); i1 != jit1->second.end(); i1++)
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
    Inherit1::init();
    const simulation::Node *node = dynamic_cast<simulation::Node *>(this->getContext());
    if (node != nullptr) mstate = node->get<MState> ();
}

template<class TDataTypes>
void PrecomputedWarpPreconditioner<TDataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (! d_use_rotations.getValue()) return;
    if (d_draw_rotations_scale.getValue() <= 0.0) return;
    if (! vparams->displayFlags().getShowBehaviorModels()) return;
    if (mstate==nullptr) return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    const VecCoord& x = mstate->read(core::vec_id::read_access::position)->getValue();
    const Real& scale = this->d_draw_rotations_scale.getValue();

    for (unsigned int i=0; i< nb_dofs; i++)
    {
        sofa::type::Matrix3 RotMat;

        for (int a=0; a<3; a++)
        {
            for (int b=0; b<3; b++)
            {
                RotMat[a][b] = R[i*9+a*3+b];
            }
        }

        int pid = i;

        sofa::type::Quat<SReal> q;
        q.fromMatrix(RotMat);
        vparams->drawTool()->drawFrame(DataTypes::getCPos(x[pid]), q, sofa::type::Vec3(scale,scale,scale));
    }

}

} // namespace sofa::component::linearsolver::preconditioner
