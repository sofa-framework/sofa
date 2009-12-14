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

#include <sofa/component/constraint/LMConstraintSolver.h>
#include <sofa/core/componentmodel/behavior/LinearSolver.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/component/linearsolver/FullVector.h>

#include <sofa/defaulttype/Quat.h>

#include <sofa/core/ObjectFactory.h>

#include <Eigen/LU>

namespace sofa
{

namespace component
{

namespace constraint
{

using linearsolver::FullVector;
using linearsolver::FullMatrix;
LMConstraintSolver::LMConstraintSolver():
    constraintAcc( initData( &constraintAcc, false, "constraintAcc", "Constraint the acceleration")),
    constraintVel( initData( &constraintVel, false, "constraintVel", "Constraint the velocity")),
    constraintPos( initData( &constraintPos, false, "constraintPos", "Constraint the position")),
    numIterations( initData( &numIterations, (unsigned int)25, "numIterations", "Number of iterations for Gauss-Seidel when solving the Constraints")),
    maxError( initData( &maxError, 0.0000001, "maxError", "Max error for Gauss-Seidel algorithm when solving the constraints")),
    A(NULL), c(NULL), Lambda(NULL)
{
}


LMConstraintSolver::~LMConstraintSolver()
{
    if (A) delete A;
    if (c) delete c;
    if (Lambda) delete Lambda;
}

void LMConstraintSolver::init()
{
    helper::vector< core::componentmodel::behavior::BaseConstraintCorrection* > listConstraintCorrection;
    ((simulation::Node*) getContext())->get<core::componentmodel::behavior::BaseConstraintCorrection>(&listConstraintCorrection, core::objectmodel::BaseContext::SearchDown);
    for (unsigned int i=0; i<listConstraintCorrection.size(); ++i)
    {
        core::componentmodel::behavior::BaseMechanicalState* constrainedDof;
        listConstraintCorrection[i]->getContext()->get(constrainedDof);
        if (constrainedDof)
        {
            constraintCorrections.insert(std::make_pair(constrainedDof, listConstraintCorrection[i]));
        }
    }
}



bool LMConstraintSolver::needPriorStatePropagation()
{
    using core::componentmodel::behavior::BaseLMConstraint;
    bool needPriorPropagation=false;
    {
        helper::vector< BaseLMConstraint* > c;
        this->getContext()->get<BaseLMConstraint>(&c, core::objectmodel::BaseContext::SearchDown);
        for (unsigned int i=0; i<c.size(); ++i)
        {
            if (!c[i]->isCorrectionComputedWithSimulatedDOF())
            {
                needPriorPropagation=true;
                if (f_printLog.getValue()) sout << "Propagating the State because of "<< c[i]->getName() << sendl;
                break;
            }
        }
    }
    return needPriorPropagation;
}



bool LMConstraintSolver::prepareStates(double /*dt*/, VecId Order)
{
    //Get the matrices through mappings
    //************************************************************
    // Update the State of the Mapped dofs                      //
    //************************************************************
#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::TRACE_ARGUMENT arg;
#endif
    using core::componentmodel::behavior::BaseLMConstraint ;
    LMConstraintVisitor.clear();

    if      (Order==VecId::dx())
    {
        if (!constraintAcc.getValue()) return false;
        orderState=BaseLMConstraint::ACC;
        if (needPriorStatePropagation())
        {
            simulation::MechanicalPropagateDxVisitor propagateState(Order,false);
            propagateState.execute(this->getContext());
        }
        // calling writeConstraintEquations
        LMConstraintVisitor.setOrder(orderState);
        LMConstraintVisitor.setTags(getTags()).execute(this->getContext());

#ifdef SOFA_DUMP_VISITOR_INFO
        arg.push_back(std::make_pair("Order", "Acceleration"));
#endif
    }
    else if (Order==VecId::velocity())
    {
        if (!constraintVel.getValue()) return false;
        orderState=BaseLMConstraint::VEL;
        if (needPriorStatePropagation())
        {
            simulation::MechanicalPropagateVVisitor propagateState(Order,false);
            propagateState.execute(this->getContext());
        }


        // calling writeConstraintEquations
        LMConstraintVisitor.setOrder(orderState);
        LMConstraintVisitor.setTags(getTags()).execute(this->getContext());

#ifdef SOFA_DUMP_VISITOR_INFO
        arg.push_back(std::make_pair("Order", "Velocity"));
#endif

    }
    else
    {
        if (!constraintPos.getValue()) return false;
        orderState=BaseLMConstraint::POS;

        if (needPriorStatePropagation())
        {
            simulation::MechanicalPropagateXVisitor propagateState(Order,false);
            propagateState.execute(this->getContext());
        }
        // calling writeConstraintEquations
        LMConstraintVisitor.setOrder(orderState);
        LMConstraintVisitor.setTags(getTags()).execute(this->getContext());

#ifdef SOFA_DUMP_VISITOR_INFO
        arg.push_back(std::make_pair("Order", "Position"));
#endif
    }

    //************************************************************
    // Find the number of constraints                           //
    //************************************************************
    numConstraint=0;
    const helper::vector< BaseLMConstraint* > &LMConstraints=LMConstraintVisitor.getConstraints();
    for (unsigned int mat=0; mat<LMConstraints.size(); ++mat)
    {
        numConstraint += LMConstraints[mat]->getNumConstraint(orderState);
    }
    if (numConstraint == 0)
    {
        return false; //Nothing to solve
    }

    if (f_printLog.getValue())
    {
        if (Order==VecId::dx())            sout << "Applying the constraint on the acceleration"<<sendl;
        else if (Order==VecId::velocity()) sout << "Applying the constraint on the velocity"<<sendl;
        else if (Order==VecId::position()) sout << "Applying the constraint on the position"<<sendl;
    }

    if (A) delete A;
    if (c) delete c;
    if (Lambda) delete Lambda;

    setDofs.clear();
    dofUsed.clear();
    invMass_Ltrans.clear();
    return true;
}



bool LMConstraintSolver::buildSystem(double /*dt*/, VecId)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::printNode("SystemCreation");
#endif
    const helper::vector< BaseLMConstraint* > &LMConstraints=LMConstraintVisitor.getConstraints();
    //Informations to build the matrices
    //Dofs to be constrained
    for (unsigned int mat=0; mat<LMConstraints.size(); ++mat)
    {
        BaseLMConstraint *constraint=LMConstraints[mat];
        setDofs.insert(constraint->getSimulatedMechModel1());
        setDofs.insert(constraint->getSimulatedMechModel2());
    }
    for (SetDof::iterator it=setDofs.begin(); it!=setDofs.end();)
    {
        SetDof::iterator currentIt=it;
        ++it;

        if (!(*currentIt)->getContext()->getMass()) setDofs.erase(currentIt);
    }


    //Store the indices used in order to speed up the correction propagation
    //************************************************************
    // Build the Right Hand Term
    //************************************************************
    c = new VectorEigen((int)numConstraint);
    buildRightHandTerm(orderState,LMConstraints, *c);


    //************************************************************
    // Build M^-1
    //************************************************************
    buildInverseMassMatrices(setDofs, invMassMatrix);

    //************************************************************
    //Building L
    //************************************************************
    DofToMatrix LMatrices;
    //Init matrices to the good size
    for (SetDof::iterator it=setDofs.begin(); it!=setDofs.end(); ++it)
    {
        const core::componentmodel::behavior::BaseMechanicalState* dofs=*it;
        SparseMatrixEigen L(numConstraint,dofs->getSize()*dofs->getDerivDimension());
        L.startFill(numConstraint*(1+dofs->getSize()));//TODO: give a better estimation of non-zero coefficients
        LMatrices.insert(std::make_pair(dofs, L));
    }
    buildLMatrices(orderState, LMConstraints, LMatrices, dofUsed);


    //************************************************************
    // Building A=J0.M0^-1.J0^T + J1.M1^-1.J1^T + ... and M^-1.J^T
    //************************************************************
    //Store the matrix M^-1.L^T for each dof in order to apply the modification of the state
    SparseMatrixEigen AEi((int)numConstraint,(int)numConstraint);
    buildLeftMatrix(invMassMatrix, LMatrices,AEi, invMass_Ltrans);

    //Convert the Sparse Matrix AEi into a Dense Matrix-> faster access to elements, and possilibity to use a direct LU solution
    A=new MatrixEigen((int)numConstraint, (int)numConstraint);
    A->setZero((int)numConstraint, (int)numConstraint);

    MatrixEigen &AEigen = *A;

    for (int k=0; k<AEi.outerSize(); ++k)
        for (SparseMatrixEigen::InnerIterator it(AEi,k); it; ++it)
        {
            AEigen(it.row(),it.col()) = it.value();
        }

#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::printCloseNode("SystemCreation");
#endif
    return true;
}



bool LMConstraintSolver::solveSystem(double /*dt*/, VecId)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::printNode("SystemSolution");
#endif

    //************************************************************
    // Solving the System using Eigen2
    //************************************************************
    if (f_printLog.getValue())
    {
        sout << "A= J0.M0^-1.J0^T + J1.M1^-1.J1^T + ...: "<<sendl;
        sout <<"\n" << *A << sendl;
        sout << "for a constraint: " << ""<<sendl;
        sout << "\n" << *c << sendl;
    }

    const helper::vector< BaseLMConstraint* > &LMConstraints=LMConstraintVisitor.getConstraints();

    Lambda=new VectorEigen(numConstraint);
    Lambda->setZero(numConstraint);

    bool solutionFound=solveConstraintSystemUsingGaussSeidel(orderState,LMConstraints, *A, *c, *Lambda);
#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::printCloseNode("SystemSolution");
#endif
    return solutionFound;
}



bool LMConstraintSolver::applyCorrection(double /*dt*/, VecId id, bool isPositionChangesUpdateVelocity)
{
    //************************************************************
    // Constraint Correction
    //************************************************************
#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::printNode("SystemCorrection");
#endif
    //************************************************************
    // Updating the state vectors
    // get the displacement. deltaState = M^-1.J^T.lambda : lambda being the solution of the system
    for (SetDof::iterator itDofs=setDofs.begin(); itDofs!=setDofs.end(); itDofs++)
    {
        sofa::core::componentmodel::behavior::BaseMechanicalState* dofs=*itDofs;
        const VectorEigen &LambdaVector=*Lambda;
        constraintStateCorrection(id, isPositionChangesUpdateVelocity,invMass_Ltrans[dofs] , LambdaVector, dofUsed[dofs], dofs);
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::printCloseNode("SystemCorrection");
#endif
    return true;
}








//----------------------------------------------------------------------------------------------//
// Specific method to build the matrices

void LMConstraintSolver::buildLeftMatrix(const DofToMatrix& invMassMatrix, DofToMatrix& LMatrix, SparseMatrixEigen &LeftMatrix, DofToMatrix &invMass_Ltrans) const
{
    for (DofToMatrix::const_iterator itDofs=invMassMatrix.begin(); itDofs!=invMassMatrix.end(); itDofs++)
    {
        const sofa::core::componentmodel::behavior::BaseMechanicalState* dofs=itDofs->first;
        const unsigned int dimensionDofs=dofs->getDerivDimension();
        const SparseMatrixEigen &invMass=itDofs->second;
        SparseMatrixEigen &L=LMatrix[dofs];
        L.endFill();
        if (f_printLog.getValue()) sout << "Matrix L for " << dofs->getName() << "\n" << L << sendl;
        //************************************************************
        //Accumulation
        //Simple way to compute
        //const SparseMatrixEigen &invM_LtransMatrix = invMass->matrix*L.transpose();
        if (f_printLog.getValue())  sout << "Matrix M-1 for " << dofs->getName() << "\n" << invMass << sendl;

        //Taking into account that invM is block tridiagonal

        SparseMatrixEigen invM_LtransMatrix(L.rows(),invMass.rows());
        invM_LtransMatrix.startFill(L.nonZeros()*dimensionDofs);
        for (int k=0; k<L.outerSize(); ++k)
        {
            int accumulatingDof=-1;
            unsigned int column=0;
            std::vector< SReal > value(dimensionDofs,0);
            for (SparseMatrixEigen::InnerIterator it(L,k); it; ++it)
            {
                const unsigned int currentIndexDof = it.col()/dimensionDofs;
                const unsigned int d = it.col()%dimensionDofs;
                if (accumulatingDof < 0)
                {
                    accumulatingDof = currentIndexDof;
                    column = it.row();
                    value.resize( dimensionDofs, 0 );
                }
                else if (accumulatingDof != (int)currentIndexDof)
                {
                    for (unsigned int iM=0; iM<dimensionDofs; ++iM)
                    {
                        SReal finalValue=SReal(0);
                        for (unsigned int iL=0; iL<dimensionDofs; ++iL)
                        {
                            finalValue += invMass.coeff(dimensionDofs*accumulatingDof+iM,dimensionDofs*accumulatingDof+iL) * value[iL];
                        }
                        invM_LtransMatrix.fill(column,dimensionDofs*accumulatingDof+iM)=finalValue;
                    }

                    accumulatingDof = currentIndexDof;
                    column = it.row();
                    value.resize( dimensionDofs, 0 );
                }

                value[d] = it.value();
            }
            if (accumulatingDof >= 0)
            {
                for (unsigned int iM=0; iM<dimensionDofs; ++iM)
                {
                    SReal finalValue=SReal(0);
                    for (unsigned int iL=0; iL<dimensionDofs; ++iL)
                    {
                        finalValue += invMass.coeff(dimensionDofs*accumulatingDof+iM,dimensionDofs*accumulatingDof+iL) * value[iL];
                    }
                    invM_LtransMatrix.fill(column,dimensionDofs*accumulatingDof+iM)=finalValue;
                }
            }
        }
        invM_LtransMatrix.endFill();

        //Optimized way is
        invMass_Ltrans.insert( std::make_pair(dofs,invM_LtransMatrix.transpose()) );
        LeftMatrix = LeftMatrix + L*invMass_Ltrans[dofs];
    }
}


void LMConstraintSolver::buildLMatrices( ConstOrder Order,
        const helper::vector< core::componentmodel::behavior::BaseLMConstraint* > &LMConstraints,
        DofToMatrix &LMatrices,
        DofToMask &dofUsed) const
{
    typedef core::componentmodel::behavior::BaseMechanicalState::ConstraintBlock ConstraintBlock;
    unsigned constraintOffset=0;
    //We Take one by one the constraint, and write their equations in the corresponding matrix L
    for (unsigned int mat=0; mat<LMConstraints.size(); ++mat)
    {
        core::componentmodel::behavior::BaseLMConstraint *constraint=LMConstraints[mat];

        const core::componentmodel::behavior::BaseMechanicalState *dof1=constraint->getSimulatedMechModel1();
        const core::componentmodel::behavior::BaseMechanicalState *dof2=constraint->getSimulatedMechModel2();

        //Get the entries in the Vector of constraints corresponding to the constraint equations
        std::list< unsigned int > indicesUsed[2];
        constraint->getIndicesUsed1(Order, indicesUsed[0]);

        DofToMatrix::iterator itL1=LMatrices.find(dof1);
        if (itL1 != LMatrices.end())
        {
            buildLMatrix(itL1->first, indicesUsed[0], constraintOffset, itL1->second, dofUsed[dof1]);
        }

        DofToMatrix::iterator itL2=LMatrices.find(dof2);
        if (dof1 != dof2 && itL2 != LMatrices.end())
        {
            constraint->getIndicesUsed2(Order, indicesUsed[1]);
            buildLMatrix(itL2->first, indicesUsed[1],constraintOffset, itL2->second,dofUsed[dof2]);
        }
        constraintOffset += indicesUsed[0].size();
    }

}



void LMConstraintSolver::buildInverseMassMatrices( const SetDof &setDofs, DofToMatrix& invMassMatrices)
{
    for (SetDof::const_iterator itDofs=setDofs.begin(); itDofs!=setDofs.end(); itDofs++)
    {
        const sofa::core::componentmodel::behavior::BaseMechanicalState* dofs=*itDofs;
        const unsigned int dimensionDofs=dofs->getDerivDimension();


        //Verify if the Big M^-1 matrix has not already been computed and stored in memory
        DofToMatrix::iterator mFound = invMassMatrices.find(dofs);
        bool needToConstructMassMatrix = true;
        if (mFound != invMassMatrices.end())
        {
            //WARNING HACK! we should find a way to know when the Mass changed: in some case the number of Dof can be the same, but the mass changed
            if ((int) (dofs->getSize()*dimensionDofs) != mFound->second.rows())
                needToConstructMassMatrix=true;
            else
                needToConstructMassMatrix=false;
        }

        //************************************************************
        //Building M^-1
        //Called only ONCE!
        core::componentmodel::behavior::BaseMass *mass=dynamic_cast< core::componentmodel::behavior::BaseMass *>(dofs->getContext()->getMass());

        if (needToConstructMassMatrix)
        {
            SparseMatrixEigen invMass(dofs->getSize()*dimensionDofs, dofs->getSize()*dimensionDofs);
            invMass.startFill(dofs->getSize()*dimensionDofs*dimensionDofs);

            DofToConstraintCorrection::iterator constraintCorrectionFound=constraintCorrections.find(dofs);
            if (constraintCorrectionFound != constraintCorrections.end())
            {
                buildInverseMassMatrix(dofs, constraintCorrectionFound->second, invMass);
            }
            else
            {
                buildInverseMassMatrix(dofs, mass, invMass);
            }
            invMass.endFill();

            //Store the matrix in memory
            if (mFound != invMassMatrices.end())
                invMassMatrices[dofs]=invMass;
            else
                invMassMatrices.insert( std::make_pair(dofs,invMass) );
        }
    }
}

void LMConstraintSolver::buildInverseMassMatrix( const sofa::core::componentmodel::behavior::BaseMechanicalState* /*mstate*/, const core::componentmodel::behavior::BaseConstraintCorrection* constraintCorrection, SparseMatrixEigen& invMass) const
{
    //Get the Full Matrix from the constraint correction
    FullMatrix<SReal> computationInvM;
    constraintCorrection->getComplianceMatrix(&computationInvM);

    //Then convert it into a Sparse Matrix: as it is done only at the init, or when topological changes occur, this should not be a burden for the framerate
    for (unsigned int i=0; i<computationInvM.rowSize(); ++i)
    {
        for (unsigned int j=0; j<computationInvM.colSize(); ++j)
        {
            SReal value=computationInvM.element(i,j);
            if (value != 0) invMass.fill(i,j)=value;
        }
    }
}

void LMConstraintSolver::buildInverseMassMatrix( const sofa::core::componentmodel::behavior::BaseMechanicalState* mstate, const core::componentmodel::behavior::BaseMass* mass, SparseMatrixEigen& invMass) const
{
    const unsigned int dimensionDofs=mstate->getDerivDimension();
    //Build M, using blocks of [ dimensionDofs x dimensionDofs ] given for each particle
    //computationM is a block corresponding the mass matrix of a particle
    FullMatrix<SReal> computationM(dimensionDofs, dimensionDofs);
    MatrixEigen invMEigen((int)dimensionDofs,(int)dimensionDofs);

    for (int i=0; i<mstate->getSize(); ++i)
    {
        mass->getElementMass(i,&computationM);

        //Translate the FullMatrix into a Eigen Matrix to invert it
        MatrixEigen mapMEigen=Eigen::Map<MatrixEigen>(computationM[0],(int)computationM.rowSize(),(int)computationM.colSize());
        mapMEigen.computeInverse(&invMEigen);

        //Store into the sparse matrix the block corresponding to the inverse of the mass matrix of a particle
        for (unsigned int r=0; r<dimensionDofs; ++r)
        {
            for (unsigned int c=0; c<dimensionDofs; ++c)
            {
                if (invMEigen(r,c) != 0)
                    invMass.fill(i*dimensionDofs+r,i*dimensionDofs+c)=invMEigen(r,c);
            }
        }
    }
}

void LMConstraintSolver::buildLMatrix( const sofa::core::componentmodel::behavior::BaseMechanicalState *dof,
        const std::list<unsigned int> &idxEquations, unsigned int constraintOffset,
        SparseMatrixEigen& L, sofa::helper::set< unsigned int > &dofUsed) const
{
    const unsigned int dimensionDofs=dof->getDerivDimension();
    typedef core::componentmodel::behavior::BaseMechanicalState::ConstraintBlock ConstraintBlock;
    //Get blocks of values from the Mechanical States
    std::list< ConstraintBlock > blocks;
    blocks =dof->constraintBlocks( idxEquations );

    std::list< ConstraintBlock >::iterator itBlock;
    //Fill the matrices
    const unsigned int numEquations=idxEquations.size();

    for (unsigned int eq=0; eq<numEquations; ++eq)
    {
        for (itBlock=blocks.begin(); itBlock!=blocks.end(); itBlock++)
        {
            const ConstraintBlock &b=(*itBlock);
            const defaulttype::BaseMatrix &m=b.getMatrix();
            const unsigned int column=b.getColumn()*dimensionDofs;
            for (unsigned int j=0; j<m.colSize(); ++j)
            {
                SReal value=m.element(eq,j);
                if (value!=0)
                {
                    // Use fill!
                    L.fill(constraintOffset+eq, column+j) = m.element(eq,j);
                    dofUsed.insert((column+j)/dimensionDofs);
                }
            }
        }

    }
    for (itBlock=blocks.begin(); itBlock!=blocks.end(); ++itBlock)
    {
        delete itBlock->getMatrix();
    }
}

void LMConstraintSolver::buildRightHandTerm( ConstOrder Order, const helper::vector< core::componentmodel::behavior::BaseLMConstraint* > &LMConstraints, VectorEigen &c) const
{
    unsigned int offset=0;
    for (unsigned int mat=0; mat<LMConstraints.size(); ++mat)
    {
        helper::vector<SReal> correction; LMConstraints[mat]->getCorrections(Order,correction);
        for (unsigned int numC=0; numC<correction.size(); ++numC)
        {
            c(offset+numC)=correction[numC];
        }
        offset += correction.size();
    }
}

bool LMConstraintSolver::solveConstraintSystemUsingGaussSeidel( ConstOrder Order, const helper::vector< core::componentmodel::behavior::BaseLMConstraint* > &LMConstraints, const MatrixEigen &A, const VectorEigen  &c, VectorEigen &Lambda) const
{
    if (f_printLog.getValue()) sout << "Using Gauss-Seidel solution"<<sendl;

    const unsigned int numConstraint=A.rows();
    //-- Initialization of X, solution of the system
    bool continueIteration=true;
    unsigned int iteration=0;
    double error=0;
    for (; iteration < numIterations.getValue() && continueIteration; ++iteration)
    {
        unsigned int idxConstraint=0;
        VectorEigen varEigen;
        VectorEigen previousIterationEigen;
        continueIteration=false;
        //Iterate on all the Constraint components
        for (unsigned int componentConstraint=0; componentConstraint<LMConstraints.size(); ++componentConstraint)
        {
            BaseLMConstraint *constraint=LMConstraints[componentConstraint];
            //Get the vector containing all the constraint stored in one component
            const std::vector< BaseLMConstraint::ConstraintGroup* > &constraintOrder=constraint->getConstraintsOrder(Order);

            for (unsigned int constraintEntry=0; constraintEntry<constraintOrder.size(); ++constraintEntry)
            {
                //-------------------------------------
                //Initialize the variables, and store X^(k-1) in previousIteration
                unsigned int numConstraintToProcess=constraintOrder[constraintEntry]->getNumConstraint();
                varEigen = VectorEigen::Zero(numConstraintToProcess);
                previousIterationEigen = VectorEigen::Zero(numConstraintToProcess);
                for (unsigned int i=0; i<numConstraintToProcess; ++i)
                {
                    previousIterationEigen(i)=Lambda(idxConstraint+i);
                    Lambda(idxConstraint+i)=0;
                }
                //    operation: A.X^k --> var

                varEigen = A.block(idxConstraint,0,numConstraintToProcess,numConstraint)*Lambda;
                error=0;
                bool groupDeactivated=false;
                unsigned int i=0;
                for (i=0; i<numConstraintToProcess; ++i)
                {
                    //X^(k)= (c^(0)-A[c,c]*X^(k-1))/A[c,c]
                    Lambda(idxConstraint+i)=(c(idxConstraint+i) - varEigen(i))/A(idxConstraint+i,idxConstraint+i);
                    if (constraintOrder[constraintEntry]->getConstraint(i).nature == BaseLMConstraint::UNILATERAL && Lambda(idxConstraint+i) < 0)
                    {
                        groupDeactivated=true;
                        if (f_printLog.getValue()) sout << "Constraint : " << i << " from group " << idxConstraint << " Deactivated" << sendl;
                        break;
                    }
                    error += pow(previousIterationEigen(i)-Lambda(idxConstraint+i),2);
                }
                //One of the Unilateral Constraint is not active anymore. We deactivate the whole group
                if (groupDeactivated)
                {
                    for (unsigned int j=0; j<numConstraintToProcess; ++j)
                    {
                        if (j<i) error -= pow(previousIterationEigen(j)-Lambda(idxConstraint+j),2);
                        Lambda(idxConstraint+j) = 0;
                        error += pow(previousIterationEigen(j)-Lambda(idxConstraint+j),2);
                    }
                }
                error = sqrt(error);
                //****************************************************************
                if (this->f_printLog.getValue())
                {
                    if (f_printLog.getValue()) sout << "Error is : " <<  error << " for system of constraint " << idxConstraint<< "[" << numConstraintToProcess << "]" << "/" << A.cols()
                                << " between " << constraint->getSimulatedMechModel1()->getName() << " and " << constraint->getSimulatedMechModel2()->getName() << ""<<sendl;
                }
                //****************************************************************
                //Update only if the error is higher than a threshold. If no "big changes" occured, we set: X[c]^(k) = X[c]^(k-1)
                if (error < maxError.getValue())
                {
                    for (unsigned int i=0; i<numConstraintToProcess; ++i)
                    {
                        Lambda(idxConstraint+i)=previousIterationEigen(i);
                    }
                }
                else
                {
                    continueIteration=true;
                }
                idxConstraint+=numConstraintToProcess;
            }
        }
        if (this->f_printLog.getValue())
        {
            if (f_printLog.getValue()) sout << "ITERATION " << iteration << " ENDED\n"<<sendl;
        }
    }
    if (iteration == numIterations.getValue())
    {
        serr << "no convergence in Gauss-Seidel for " << Order <<sendl;
        return false;
    }

    if (f_printLog.getValue()) sout << "Gauss-Seidel done in " << iteration << " iterations "<<sendl;
    return true;
}

void LMConstraintSolver::constraintStateCorrection(VecId Order, bool isPositionChangesUpdateVelocity,
        const SparseMatrixEigen  &invM_Ltrans,
        const VectorEigen  &c,
        const sofa::helper::set< unsigned int > &dofUsed,
        sofa::core::componentmodel::behavior::BaseMechanicalState* dofs) const
{

    //Correct Dof
    //    operation: M0^-1.J0^T.lambda -> A

    VectorEigen A = invM_Ltrans*c;
    if (f_printLog.getValue())
    {
        sout << "M^-1.L^T " << "\n" << invM_Ltrans << sendl;
        sout << "Lambda " << dofs->getName() << "\n" << c << sendl;
        sout << "Correction " << dofs->getName() << "\n" << A << sendl;
    }
    const unsigned int dimensionDofs=dofs->getDerivDimension();

    unsigned int offset=0;
    //In case of position correction, we need to update the velocities
    if (Order==VecId::position())
    {
        //Detect Rigid Bodies
        if (dofs->getCoordDimension() == 7 && dofs->getDerivDimension() == 6)
        {
            VectorEigen Acorrection=VectorEigen::Zero(dofs->getSize()*(3+4));
            //We have to transform the Euler Rotations into a quaternion
            offset=0;

            for (int l=0; l<A.rows(); l+=6)
            {
                offset=l/6;
                Acorrection(l+0+offset)=A(l+0);
                Acorrection(l+1+offset)=A(l+1);
                Acorrection(l+2+offset)=A(l+2);

                defaulttype::Quaternion q=defaulttype::Quaternion::createQuaterFromEuler(defaulttype::Vector3(A(l+3),A(l+4),A(l+5)));

                Acorrection(l+3+offset)=q[0];
                Acorrection(l+4+offset)=q[1];
                Acorrection(l+5+offset)=q[2];
                Acorrection(l+6+offset)=q[3];
            }
            offset=0;
            FullVector<SReal> v(Acorrection.data(),Acorrection.rows());

            if (f_printLog.getValue())  sout << "Lambda Corrected for Rigid " << "\n" << Acorrection << sendl;
            dofs->addBaseVectorToState(VecId::position(),&v,offset );

        }
        else
        {
            std::set< unsigned int >::const_iterator it;
            for (it=dofUsed.begin(); it!=dofUsed.end(); it++)
            {
                unsigned int offset=(*it);
                FullVector<SReal> v(&(A.data()[offset*dimensionDofs]),dimensionDofs);
                dofs->addVectorToState(VecId::position(),&v,offset );
            }
        }

        if (isPositionChangesUpdateVelocity)
        {
            const double h=1.0/getContext()->getDt();

            std::set< unsigned int >::const_iterator it;
            for (it=dofUsed.begin(); it!=dofUsed.end(); it++)
            {
                unsigned int offset=(*it);
                FullVector<SReal> v(&(A.data()[offset*dimensionDofs]),dimensionDofs);
                for (unsigned int i=0; i<dimensionDofs; ++i) v[i]*=h;
                dofs->addVectorToState(VecId::velocity(),&v,offset );
            }
        }

    }
    else
    {
        std::set< unsigned int >::const_iterator it;
        for (it=dofUsed.begin(); it!=dofUsed.end(); it++)
        {
            unsigned int offset=(*it);
            FullVector<SReal> v(&(A.data()[offset*dimensionDofs]),dimensionDofs);
            dofs->addVectorToState(Order,&v,offset );
        }

    }
}


int LMConstraintSolverClass = core::RegisterObject("A Constraint Solver working specifically with LMConstraint based components")
        .add< LMConstraintSolver >();

SOFA_DECL_CLASS(LMConstraintSolver);


} // namespace constraint

} // namespace component

} // namespace sofa
