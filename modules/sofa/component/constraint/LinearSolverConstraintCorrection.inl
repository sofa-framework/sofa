/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_CORE_COMPONENTMODEL_COLLISION_LINEARSOLVERCONTACTCORRECTION_INL
#define SOFA_CORE_COMPONENTMODEL_COLLISION_LINEARSOLVERCONTACTCORRECTION_INL

#include "LinearSolverConstraintCorrection.h"
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/component/mass/UniformMass.h>
#include <sofa/simulation/tree/MechanicalVisitor.h>

//compliance computation include
#include <sofa/component/odesolver/CGImplicitSolver.h>
//#include <glib.h>
#include <sstream>
#include <list>

namespace sofa
{

namespace component
{

namespace constraint
{
#define	MAX_NUM_CONSTRAINT_PER_NODE 100
#define EPS_UNITARY_FORCE 0.01

using namespace sofa::component::odesolver;

template<class DataTypes>
LinearSolverConstraintCorrection<DataTypes>::LinearSolverConstraintCorrection(behavior::MechanicalState<DataTypes> *mm)
    : mstate(mm), odesolver(NULL), linearsolver(NULL)
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
    linearsolver=getLinearSolver(c);
    if (odesolver == NULL)
    {
        std::cerr << "LinearSolverConstraintCorrection: ERROR no OdeSolver found."<<std::endl;
        return;
    }
    if (linearsolver == NULL)
    {
        std::cerr << "LinearSolverConstraintCorrection: ERROR no LinearSolver found."<<std::endl;
        return;
    }

    int n = mstate->getSize()*Deriv::size();

    std::stringstream ss;

    ss << this->getContext()->getName() << ".comp";

    std::ifstream compFileIn(ss.str().c_str(), std::ifstream::binary);

    std::cout << "try to open : " << ss.str() << endl;

    if(compFileIn.good())
    {
        std::cout << "file open : " << ss.str() << " compliance being loaded" << endl;
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
    if (refMinv.rowSize() > 0)
    {
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
                //std::cout << "Minv("<<i<<","<<j<<") = "<<Minv.element(i,j)<<"\t refMinv("<<i<<","<<j<<") = "<<refMinv.element(i,j)<<std::endl;
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
        std::cout << "LinearSolverConstraintCorrection: mean relative error: "<<err/(numDOFReals*numDOFReals)<<std::endl;
        std::cout << "LinearSolverConstraintCorrection: mean relative factor: "<<fact/(numDOFReals*numDOFReals)<<std::endl;
        refMinv.resize(0,0);
    }
    // Compute J
    VecConst& constraints = *mstate->getC();
    const unsigned int numConstraints = constraints.size();
    const unsigned int totalNumConstraints = W->rowSize();

    J.resize(totalNumConstraints, numDOFReals);
    for(unsigned int c1 = 0; c1 < numConstraints; c1++)
    {
        int cid = mstate->getConstraintId()[c1];
        for(unsigned int i = 0; i < constraints[c1].size(); i++)
        {
            int dof = constraints[c1][i].index;
            Deriv n = constraints[c1][i].data;
            for (unsigned int r=0; r<N; ++r)
                J.add(cid, dof*N+r, n[r]);
        }
    }

    // use the Linear solver to compute J*inv(M)*Jt, where M is the mechanical linear system matrix
    linearsolver->addJMInvJt(W, &J, factor);
}

template<class DataTypes>
void LinearSolverConstraintCorrection<DataTypes>::applyContactForce(const defaulttype::BaseVector *f)
{
    behavior::BaseMechanicalState::VecId forceID(behavior::BaseMechanicalState::VecId::V_DERIV, behavior::BaseMechanicalState::VecId::V_FIRST_DYNAMIC_INDEX);
    behavior::BaseMechanicalState::VecId dxID(behavior::BaseMechanicalState::VecId::dx()); //behavior::BaseMechanicalState::VecId::V_DERIV, behavior::BaseMechanicalState::VecId::V_FIRST_DYNAMIC_INDEX+1);
    mstate->vAlloc(forceID);
    mstate->vOp(forceID);
//    mstate->vAlloc(dxID);
    mstate->setDx(forceID);
    VecDeriv& force = *mstate->getDx();
    mstate->setDx(dxID);
    VecDeriv& dx = *mstate->getDx();
    mstate->setDx(behavior::BaseMechanicalState::VecId::dx());
    //VecConst& constraints = *mstate->getC();
    //unsigned int numConstraints = constraints.size();
    VecCoord& x = *mstate->getX();
    VecDeriv& v = *mstate->getV();
    VecDeriv v_free = *mstate->getVfree();
    VecCoord x_free = *mstate->getXfree();
    //double dt = this->getContext()->getDt();

    const unsigned int numDOFs = mstate->getSize();

    dx.clear();
    dx.resize(numDOFs);
    for (unsigned int i=0; i< numDOFs; i++)
        dx[i] = Deriv();

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
        J.mulTranspose(F, fcast); // slow but generic
    for (unsigned int i=0; i< numDOFs; i++)
        for (unsigned int r=0; r<N; ++r)
            force[i][r] = F[i*N+r];
#else
    const VecConst& constraints = *mstate->getC();
    unsigned int numConstraints = constraints.size();

    for(unsigned int c1 = 0; c1 < numConstraints; c1++)
    {
        int indexC1 = mstate->getConstraintId()[c1];
        Real fC1 = f->element(indexC1);
        //std::cout << "fC("<<indexC1<<")="<<fC1<<std::endl;
        if (fC1 != 0.0)
        {
            int sizeC1 = constraints[c1].size();
            for(int i = 0; i < sizeC1; i++)
            {
                //std::cout << "f("<<constraints[c1][i].index<<") += "<< (constraints[c1][i].data * fC1) << std::endl;
                force[constraints[c1][i].index] += constraints[c1][i].data * fC1;
            }
        }
    }
#endif
    //for (unsigned int i=0; i< numDOFs; i++)
    //    std::cout << "f("<<i<<")="<<force[i]<<std::endl;
    linearsolver->setSystemRHVector(forceID);
    linearsolver->setSystemLHVector(dxID);
    linearsolver->solveSystem(); //TODO: tell the solver not to recompute the matrix

    // use the OdeSolver to get the position integration factor
    const Real positionFactor = odesolver->getPositionIntegrationFactor();

    // use the OdeSolver to get the position integration factor
    const Real velocityFactor = odesolver->getVelocityIntegrationFactor();

    for (unsigned int i=0; i< numDOFs; i++)
    {
        //std::cout << "dx("<<i<<")="<<dx[i]<<std::endl;
        Deriv dxi = dx[i]*positionFactor;
        Deriv dvi = dx[i]*velocityFactor;
        x[i] = x_free[i] + dxi;
        v[i] = v_free[i] + dvi;
        dx[i] = dxi;
    }
    mstate->vFree(forceID);
}

template<class DataTypes>
void LinearSolverConstraintCorrection<DataTypes>::resetContactForce()
{
    VecDeriv& force = *mstate->getF();
    for( unsigned i=0; i<force.size(); ++i )
        force[i] = Deriv();
}

} // namespace collision

} // namespace component

} // namespace sofa

#endif
