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
#ifndef SOFA_CORE_COLLISION_LINEARSOLVERCONTACTCORRECTION_INL
#define SOFA_CORE_COLLISION_LINEARSOLVERCONTACTCORRECTION_INL

#include "LinearSolverConstraintCorrection.h"
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/MechanicalVisitor.h>

#include <sstream>
#include <list>

namespace sofa
{

namespace component
{

namespace constraintset
{
#define MAX_NUM_CONSTRAINT_PER_NODE 100
#define EPS_UNITARY_FORCE 0.01

template<class DataTypes>
LinearSolverConstraintCorrection<DataTypes>::LinearSolverConstraintCorrection(behavior::MechanicalState<DataTypes> *mm)
    : wire_optimization(initData(&wire_optimization, false, "wire_optimization", "constraints are reordered along a wire-like topology (from tip to base)"))
    , solverName(initData(&solverName, std::string(""), "solverName", "name of the constraint solver"))
    , mstate(mm), odesolver(NULL), linearsolver(NULL)
{
}

template<class DataTypes>
LinearSolverConstraintCorrection<DataTypes>::~LinearSolverConstraintCorrection()
{
}



//////////////////////////////////////////////////////////////////////////
//   Precomputation of the Constraint Correction for all type of data
//////////////////////////////////////////////////////////////////////////

template<class DataTypes>
void LinearSolverConstraintCorrection<DataTypes>::init()
{
    mstate = dynamic_cast< behavior::MechanicalState<DataTypes>* >(getContext()->getMechanicalState());
    objectmodel::BaseContext* c = this->getContext();
    //     odesolver = c->get< behavior::OdeSolver >();
    //     linearsolver = c->get< behavior::LinearSolver >();
    odesolver=getOdeSolver(c);
    if (solverName.getValue() == "") linearsolver=getLinearSolver(c);
    else linearsolver=getLinearSolverByName(c,solverName.getValue());
    if (odesolver == NULL)
    {
        serr << "LinearSolverConstraintCorrection: ERROR no OdeSolver found."<<sendl;
        return;
    }
    if (linearsolver == NULL)
    {
        serr << "LinearSolverConstraintCorrection: ERROR no LinearSolver found."<<sendl;
        return;
    }

    int n = mstate->getSize()*Deriv::size();

    std::stringstream ss;
    ss << this->getContext()->getName() << ".comp";
    std::string file=ss.str();
    sout << "try to open : " << ss.str() << endl;
    if (sofa::helper::system::DataRepository.findFile(file))
    {
        std::string invName=sofa::helper::system::DataRepository.getFile(ss.str());
        std::ifstream compFileIn(invName.c_str(), std::ifstream::binary);
        refMinv.resize(n,n);
        //complianceLoaded = true;
        compFileIn.read((char*)refMinv.ptr(), n*n*sizeof(double));
        compFileIn.close();
    }
}

template<class DataTypes>
void LinearSolverConstraintCorrection<DataTypes>::getCompliance(defaulttype::BaseMatrix* W)
{
    if (!mstate || !odesolver || !linearsolver) return;

    // use the OdeSolver to get the position integration factor
    //const double factor = 1.0;
    //const double factor = odesolver->getPositionIntegrationFactor(); // dt
    const double factor = odesolver->getPositionIntegrationFactor(); //*odesolver->getPositionIntegrationFactor(); // dt*dt

    const unsigned int numDOFs = mstate->getSize();
    const unsigned int N = Deriv::size();
    const unsigned int numDOFReals = numDOFs*N;
    if (refMinv.rowSize() > 0)			// What's for ??
    {
        std::cout<<"refMinv.rowSize() > 0"<<std::endl;
        J.resize(numDOFReals,numDOFReals);
        for (unsigned int i=0; i<numDOFReals; ++i)
            J.set(i,i,1);
        linearsolver::FullMatrix<Real> Minv;
        Minv.resize(numDOFReals,numDOFReals);
        // use the Linear solver to compute J*inv(M)*Jt, where M is the mechanical linear system matrix
        linearsolver->addJMInvJt(&Minv, &J, factor);
        double err=0,fact=0;
        for (unsigned int i=0; i<numDOFReals; ++i)
            for (unsigned int j=0; j<numDOFReals; ++j)
            {
                //sout << "Minv("<<i<<","<<j<<") = "<<Minv.element(i,j)<<"\t refMinv("<<i<<","<<j<<") = "<<refMinv.element(i,j)<<sendl;
                if (fabs(refMinv.element(i,j)) > 1.0e-30)
                {
                    err += fabs(Minv.element(i,j)-refMinv.element(i,j))/refMinv.element(i,j);
                    fact += fabs(Minv.element(i,j)/refMinv.element(i,j));
                }
                else
                {
                    err += fabs(Minv.element(i,j)-refMinv.element(i,j));
                    fact += 1.0f;
                }
            }
        sout << "LinearSolverConstraintCorrection: mean relative error: "<<err/(SReal)(numDOFReals*numDOFReals)<<sendl;
        sout << "LinearSolverConstraintCorrection: mean relative factor: "<<fact/(SReal)(numDOFReals*numDOFReals)<<sendl;
        refMinv.resize(0,0);
    }

    // Compute J
    const MatrixDeriv& c = *mstate->getC();
    const unsigned int totalNumConstraints = W->rowSize();

    J.resize(totalNumConstraints, numDOFReals);

    MatrixDerivRowConstIterator rowItEnd = c.end();

    for (MatrixDerivRowConstIterator rowIt = c.begin(); rowIt != rowItEnd; ++rowIt)
    {
        const int cid = rowIt.index();

        MatrixDerivColConstIterator colItEnd = rowIt.end();

        for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
        {
            const unsigned int dof = colIt.index();
            const Deriv n = colIt.val();

            for (unsigned int r = 0; r < N; ++r)
            {
                J.add(cid, dof * N + r, n[r]);
            }
        }
    }

    // use the Linear solver to compute J*inv(M)*Jt, where M is the mechanical linear system matrix
    linearsolver->addJMInvJt(W, &J, factor);
}

template<class DataTypes>
void LinearSolverConstraintCorrection<DataTypes>::getComplianceMatrix(defaulttype::BaseMatrix* Minv) const
{
    if (!mstate || !odesolver || !linearsolver) return;

    // use the OdeSolver to get the position integration factor
    //const double factor = 1.0;
    //const double factor = odesolver->getPositionIntegrationFactor(); // dt
    const double factor = odesolver->getPositionIntegrationFactor(); //*odesolver->getPositionIntegrationFactor(); // dt*dt

    const unsigned int numDOFs = mstate->getSize();
    const unsigned int N = Deriv::size();
    const unsigned int numDOFReals = numDOFs*N;
    static linearsolver::SparseMatrix<SReal> J; //local J
    if (J.rowSize() != numDOFReals)
    {
        J.resize(numDOFReals,numDOFReals);
        for (unsigned int i=0; i<numDOFReals; ++i)
            J.set(i,i,1);
    }

    Minv->resize(numDOFReals,numDOFReals);
    // use the Linear solver to compute J*inv(M)*Jt, where M is the mechanical linear system matrix
    linearsolver->addJMInvJt(Minv, &J, factor);
    double err=0,fact=0;
    for (unsigned int i=0; i<numDOFReals; ++i)
        for (unsigned int j=0; j<numDOFReals; ++j)
        {
            //sout << "Minv("<<i<<","<<j<<") = "<<Minv.element(i,j)<<"\t refMinv("<<i<<","<<j<<") = "<<refMinv.element(i,j)<<sendl;
            if (fabs(refMinv.element(i,j)) > 1.0e-30)
            {
                err += fabs(Minv->element(i,j)-refMinv.element(i,j))/refMinv.element(i,j);
                fact += fabs(Minv->element(i,j)/refMinv.element(i,j));
            }
            else
            {
                err += fabs(Minv->element(i,j)-refMinv.element(i,j));
                fact += 1.0f;
            }
        }
    sout << "LinearSolverConstraintCorrection: mean relative error: "<<err/(SReal)(numDOFReals*numDOFReals)<<sendl;
    sout << "LinearSolverConstraintCorrection: mean relative factor: "<<fact/(SReal)(numDOFReals*numDOFReals)<<sendl;
}

template<class DataTypes>
void LinearSolverConstraintCorrection<DataTypes>::applyContactForce(const defaulttype::BaseVector *f)
{
    core::VecDerivId forceID(core::VecDerivId::V_FIRST_DYNAMIC_INDEX);
    core::VecDerivId dxID = core::VecDerivId::dx();

    mstate->vAlloc(forceID);
    mstate->vOp(forceID);
    //    mstate->vAlloc(dxID);

    //double dt = this->getContext()->getDt();

    const unsigned int numDOFs = mstate->getSize();

    helper::WriteAccessor<Data<VecDeriv> > dataDx = *mstate->write(dxID);
    VecDeriv& dx = dataDx.wref();

    dx.clear();
    dx.resize(numDOFs);
    for (unsigned int i=0; i< numDOFs; i++)
        dx[i] = Deriv();

    helper::WriteAccessor<Data<VecDeriv> > dataForce = *mstate->write(forceID);
    VecDeriv& force = dataForce.wref();

    force.clear();
    force.resize(numDOFs);
    for (unsigned int i=0; i< numDOFs; i++)
        force[i] = Deriv();
#if 0
    const unsigned int N = Deriv::size();
    const unsigned int numDOFReals = numDOFs*N;
    F.resize(numDOFReals);
    const linearsolver::FullVector<Real>* fcast = dynamic_cast< const linearsolver::FullVector<Real>* >(f);
    if (fcast)
        J.mulTranspose(F, *fcast); // fast
    else
        J.mulTranspose(F, f); // slow but generic
    for (unsigned int i=0; i< numDOFs; i++)
        for (unsigned int r=0; r<N; ++r)
            force[i][r] = F[i*N+r];
#else
    const MatrixDeriv& c = *mstate->getC();

    MatrixDerivRowConstIterator rowItEnd = c.end();

    for (MatrixDerivRowConstIterator rowIt = c.begin(); rowIt != rowItEnd; ++rowIt)
    {
        const double fC1 = f->element(rowIt.index());

        if (fC1 != 0.0)
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                force[colIt.index()] += colIt.val() * fC1;
            }
        }
    }
#endif
    //for (unsigned int i=0; i< numDOFs; i++)
    //    sout << "f("<<i<<")="<<force[i]<<sendl;
    linearsolver->setSystemRHVector(forceID);
    linearsolver->setSystemLHVector(dxID);
    linearsolver->solveSystem();
    //TODO: tell the solver not to recompute the matrix

    // use the OdeSolver to get the position integration factor
    const double positionFactor = odesolver->getPositionIntegrationFactor();

    // use the OdeSolver to get the position integration factor
    const double velocityFactor = odesolver->getVelocityIntegrationFactor();


    helper::WriteAccessor<Data<VecCoord> > xData     = *mstate->write(core::VecCoordId::position());
    helper::WriteAccessor<Data<VecDeriv> > vData     = *mstate->write(core::VecDerivId::velocity());
    helper::ReadAccessor<Data<VecCoord> >  xfreeData = *mstate->read(core::ConstVecCoordId::freePosition());
    helper::ReadAccessor<Data<VecDeriv> >  vfreeData = *mstate->read(core::ConstVecDerivId::freeVelocity());
    VecCoord& x = xData.wref();
    VecDeriv& v = vData.wref();
    const VecCoord& x_free = xfreeData.ref();
    const VecDeriv& v_free = vfreeData.ref();

    for (unsigned int i=0; i< numDOFs; i++)
    {
        //sout << "dx("<<i<<")="<<dx[i]<<sendl;
        Deriv dxi = dx[i]*positionFactor;
        Deriv dvi = dx[i]*velocityFactor;
        x[i] = x_free[i] + dxi;
        v[i] = v_free[i] + dvi;
        dx[i] = dxi;

        if (this->f_printLog.getValue()) std::cout << "dx[" << i << "] = " << dx[i] << std::endl;
    }

    mstate->vFree(forceID);
}


template<class DataTypes>
void LinearSolverConstraintCorrection<DataTypes>::applyPredictiveConstraintForce(const defaulttype::BaseVector *f)
{
    helper::WriteAccessor<Data<VecDeriv> > forceData = *mstate->write(core::VecDerivId::externalForce());
    VecDeriv& force = forceData.wref();

    const unsigned int numDOFs = mstate->getSize();

    force.clear();
    force.resize(numDOFs);
    for (unsigned int i=0; i< numDOFs; i++)
        force[i] = Deriv();

    const MatrixDeriv& c = *mstate->getC();

    MatrixDerivRowConstIterator rowItEnd = c.end();

    for (MatrixDerivRowConstIterator rowIt = c.begin(); rowIt != rowItEnd; ++rowIt)
    {
        const double fC1 = f->element(rowIt.index());

        if (fC1 != 0.0)
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                force[colIt.index()] += colIt.val() * fC1;
            }
        }
    }
}


template<class DataTypes>
void LinearSolverConstraintCorrection<DataTypes>::resetContactForce()
{
    helper::WriteAccessor<Data<VecDeriv> > forceData = *mstate->write(core::VecDerivId::force());
    VecDeriv& force = forceData.wref();
    for( unsigned i=0; i<force.size(); ++i )
        force[i] = Deriv();
}


template<class DataTypes>
bool LinearSolverConstraintCorrection<DataTypes>::hasConstraintNumber(int index)
{
    const MatrixDeriv& c = *mstate->getC();

    return c.readLine(index) != c.end();
}


template<class DataTypes>
void LinearSolverConstraintCorrection<DataTypes>::verify_constraints()
{
    // New design prevents duplicated constraints.
}

template<class DataTypes>
void LinearSolverConstraintCorrection<DataTypes>::resetForUnbuiltResolution(double * f, std::list<int>& renumbering)
{
    verify_constraints();

    const MatrixDeriv& constraints = *mstate->getC();

    constraint_disp.clear();
    constraint_disp.resize(mstate->getSize());

    constraint_force.clear();
    constraint_force.resize(mstate->getSize());

    constraint_dofs.clear();
    id_to_localIndex.clear();

    ////// TODO : supprimer le classement par indice max
    //std::vector<unsigned int> VecMaxDof;
    //VecMaxDof.resize(numConstraints);

    const unsigned int nbConstraints = constraints.size();
    std::vector<unsigned int> VecMinDof;
    VecMinDof.resize(nbConstraints);

    int maxIndex = -1;
    unsigned int c = 0;

    MatrixDerivRowConstIterator rowItEnd = constraints.end();

    for (MatrixDerivRowConstIterator rowIt = constraints.begin(); rowIt != rowItEnd; ++rowIt)
    {
        I_last_Dforce.clear();

        const int indexC = rowIt.index();

        // resize table if necessary
        if (indexC > maxIndex)
        {
            id_to_localIndex.resize(indexC + 1, -1);   // debug : -1 value allows to know if the table is badly filled
            maxIndex = indexC;
        }

        if(id_to_localIndex[indexC] != -1)
        {
            serr << " WARNING: id_to_localIndex[" << indexC << "] has already a constraint : " << id_to_localIndex[indexC] << " concurrent constraint =" << c << sendl;
        }

        // buf the table of local indices
        id_to_localIndex[indexC] = c;

        // debug //
        //if (c==0)
        //	f[indexC]=1.0;

        // buf the value of force applied on concerned dof : constraint_force
        // buf a table of indice of involved dof : constraint_dofs
        double fC = f[indexC];

        if (fC != 0.0)
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                const unsigned int dof = colIt.index();
                constraint_force[dof] += colIt.val() * fC;
                I_last_Dforce.push_back(dof);
            }
        }

        //////////// for wire optimization ////////////

        MatrixDerivColConstIterator colItEnd = rowIt.end();

        //VecMaxDof[c] = 0;

        /*for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
        {
            const unsigned int dof = colIt.index();
            constraint_dofs.push_back(dof);
            if (dof > VecMaxDof[c])
                    VecMaxDof[c] = dof;
        }*/

        VecMinDof[c] = mstate->getSize()+1;

        for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
        {
            const unsigned int dof = colIt.index();
            constraint_dofs.push_back(dof);
            if (dof < VecMinDof[c])
                VecMinDof[c] = dof;
        }

        c++;
    }

    if (wire_optimization.getValue())
    {
        std::vector< std::vector<int> > ordering_per_dof;
        ordering_per_dof.resize(mstate->getSize());

        MatrixDerivRowConstIterator rowItEnd = constraints.end();
        unsigned int c = 0;

        for (MatrixDerivRowConstIterator rowIt = constraints.begin(); rowIt != rowItEnd; ++rowIt)
        {
            ordering_per_dof[VecMinDof[c]].push_back(rowIt.index());
            c++;
        }

        renumbering.clear();

        for (int dof = 0; dof < mstate->getSize(); dof++)
        {
            for (unsigned int c = 0; c < ordering_per_dof[dof].size(); c++)
            {
                renumbering.push_back(ordering_per_dof[dof][c]);
            }
        }
    }

    // debug
    //std::cout<<"in resetConstraintForce : constraint_force ="<<constraint_force<<std::endl;

    // constraint_dofs buff the DOF that are involved with the constraints
    constraint_dofs.unique();

    I_last_Dforce.sort();
    I_last_Dforce.unique();

    // debug
    /*std::cout<<"in resetConstraintForce I_last_Dforce.size() = "<<I_last_Dforce.size()<<"value : "<<std::endl;
    std::list<int>::const_iterator lit(I_last_Dforce.begin()), lend(I_last_Dforce.end());
    for(;lit!=lend;++lit)
    {
            int dof =*lit;
            std::cout<<dof<<" - ";
    }
    std::cout<<" "<<std::endl;
    */

    /////////////// SET INFO FOR LINEAR SOLVER /////////////

    core::VecDerivId forceID(core::VecDerivId::V_FIRST_DYNAMIC_INDEX);
    core::VecDerivId dxID = core::VecDerivId::dx();

    linearsolver->setSystemRHVector(forceID);
    linearsolver->setSystemLHVector(dxID);

    systemMatrix_buf   = linearsolver->getSystemBaseMatrix();
    systemRHVector_buf = linearsolver->getSystemRHBaseVector();
    systemLHVector_buf = linearsolver->getSystemLHBaseVector();

    // systemRHVector_buf is set to constraint_force;
    //std::cerr<<"WARNING: resize is called"<<std::endl;
    const unsigned int derivDim = Deriv::size();
    const unsigned int systemSize = mstate->getSize() * derivDim;
    systemRHVector_buf->resize(systemSize) ;
    systemLHVector_buf->resize(systemSize) ;
    //std::cerr<<"resize ok"<<std::endl;

    for ( int i=0; i<mstate->getSize(); i++)
    {
        for  (unsigned int j=0; j<derivDim; j++)
            systemRHVector_buf->set(i*derivDim+j, constraint_force[i][j]);
    }

    // debug !!
    //double values[12];
    //values[0]=0.0;
    //addConstraintDisplacement(values, 0,0) ;
    //std::cout<<"values[0] ="<<values[0]<<std::endl;

    ///////// new : précalcul des liste d'indice ///////
    Vec_I_list_dof.clear(); // clear = the list is fill during the block compliance computation
    Vec_I_list_dof.resize(nbConstraints);
    last_disp = 0;
    last_force = nbConstraints - 1;
    _new_force = true;
}

template<class DataTypes>
void LinearSolverConstraintCorrection<DataTypes>::addConstraintDisplacement(double *d, int begin, int end)
{
    const MatrixDeriv& constraints = *mstate->getC();
    const unsigned int derivDim = Deriv::size();

    last_disp = begin;

    linearsolver->partial_solve(Vec_I_list_dof[last_disp], Vec_I_list_dof[last_force], _new_force);

    _new_force = false;

    for (int i = begin; i <= end; i++)
    {
        MatrixDerivRowConstIterator rowIt = constraints.readLine(id_to_localIndex[i]);

        if (rowIt != constraints.end())
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                const unsigned int dof = colIt.index();
                Deriv disp;

                for(unsigned int j = 0; j < derivDim; j++)
                {
                    disp[j] = (Real)(systemLHVector_buf->element(dof * derivDim + j) * odesolver->getPositionIntegrationFactor());
                }

                d[i] += colIt.val() * disp;
            }
        }
    }
}

template<class DataTypes>
void LinearSolverConstraintCorrection<DataTypes>::setConstraintDForce(double *df, int begin, int end, bool update)
{
    const MatrixDeriv& constraints = *mstate->getC();
    const unsigned int derivDim = Deriv::size();

    if (!update)
        return;

    _new_force = true;

    Deriv DF_c;

    for (int i = begin; i <= end; i++)
    {
        MatrixDerivRowConstIterator rowIt = constraints.readLine(id_to_localIndex[i]);

        if (rowIt != constraints.end())
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                const Deriv n = colIt.val();
                const unsigned int dof = colIt.index();

                constraint_force[dof] += n * df[i];
                DF_c +=  n * df[i];
            }
        }
    }

    /*
    if (df[begin]<0)
    {
    std::cout<<" DF_c : "<< DF_c<< std::endl;
    }
    */

    last_force = begin;
    //debug

    std::list<int>::const_iterator it_dof(Vec_I_list_dof[last_force].begin()), it_end(Vec_I_list_dof[last_force].end());
    for(; it_dof!=it_end; ++it_dof)
    {
        int dof =(*it_dof) ;
        //std::cout<<"dof -  "<<dof <<std::endl;
        for  (unsigned int j=0; j<derivDim; j++)
            systemRHVector_buf->set(dof * derivDim + j, constraint_force[dof][j]);
    }
}

template<class DataTypes>
void LinearSolverConstraintCorrection<DataTypes>::getBlockDiagonalCompliance(defaulttype::BaseMatrix* W, int begin, int end)
{
    if (!mstate || !odesolver || !linearsolver) return;

    // use the OdeSolver to get the position integration factor
    const double factor = odesolver->getPositionIntegrationFactor(); //*odesolver->getPositionIntegrationFactor(); // dt*dt

    const unsigned int numDOFs = mstate->getSize();
    const unsigned int N = Deriv::size();
    const unsigned int numDOFReals = numDOFs*N;

    // Compute J
    const MatrixDeriv& constraints = *mstate->getC();
    const unsigned int totalNumConstraints = W->rowSize();

    J.resize(totalNumConstraints, numDOFReals);

    for (int i = begin; i <= end; i++)
    {
        int c1 = id_to_localIndex[i];

        MatrixDerivRowConstIterator rowIt = constraints.readLine(c1);

        if (rowIt != constraints.end())
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            unsigned int dof_buf = 0;
            int debug = 0;

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                const unsigned int dof = colIt.index();
                const Deriv n = colIt.val();

                for (unsigned int r = 0; r < N; ++r)
                    J.add(i, dof * N + r, n[r]);

                if (debug!=0)
                {
                    int test = dof_buf - dof;
                    if (test>2 || test< -2)
                        sout << "YES !!!! for constraint id1 dof1 = " << dof_buf << " dof2 = " << dof << sendl;
                }

                dof_buf = dof;
            }
        }
    }

    // use the Linear solver to compute J*inv(M)*Jt, where M is the mechanical linear system matrix
    linearsolver->addJMInvJt(W, &J, factor);

    // construction of  Vec_I_list_dof : vector containing, for each constraint block, the list of dof concerned

    ListIndex list_dof;

    for (int i = begin; i <= end; i++)
    {
        int c = id_to_localIndex[i];

        MatrixDerivRowConstIterator rowIt = constraints.readLine(c);

        if (rowIt != constraints.end())
        {
            MatrixDerivColConstIterator colItEnd = rowIt.end();

            for (MatrixDerivColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
            {
                list_dof.push_back(colIt.index());
            }
        }
    }

    list_dof.sort();
    list_dof.unique();

    for (int i = begin; i <= end; i++)
    {
        Vec_I_list_dof[i] = list_dof;
    }
}

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif
