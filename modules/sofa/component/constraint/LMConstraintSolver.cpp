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
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/component/linearsolver/FullVector.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>

#include <sofa/defaulttype/Quat.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/AdvancedTimer.h>

#include <Eigen/LU>
#include <Eigen/QR>

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
    graphGSError( initData(&graphGSError,"graphGSError","Graph of residuals at each iteration") ),
    traceKineticEnergy( initData( &traceKineticEnergy, true, "traceKineticEnergy", "Trace the evolution of the Kinetic Energy throughout the solution of the system")),
    graphKineticEnergy( initData(&graphKineticEnergy,"graphKineticEnergy","Graph of the kinetic energy of the system") ),
    W(NULL), c(NULL), Lambda(NULL)
{
    graphGSError.setGroup("Statistics");
    graphGSError.setWidget("graph");
    graphGSError.setReadOnly(true);

    graphKineticEnergy.setGroup("Statistics");
    graphKineticEnergy.setWidget("graph");
    graphKineticEnergy.setReadOnly(true);

    traceKineticEnergy.setGroup("Statistics");
    this->f_listening.setValue(true);
}


LMConstraintSolver::~LMConstraintSolver()
{
    if (W) delete W;
    if (c) delete c;
    if (Lambda) delete Lambda;
}

void LMConstraintSolver::init()
{
    helper::vector< core::behavior::BaseConstraintCorrection* > listConstraintCorrection;
    ((simulation::Node*) getContext())->get<core::behavior::BaseConstraintCorrection>(&listConstraintCorrection, core::objectmodel::BaseContext::SearchDown);
    for (unsigned int i=0; i<listConstraintCorrection.size(); ++i)
    {
        core::behavior::BaseMechanicalState* constrainedDof;
        listConstraintCorrection[i]->getContext()->get(constrainedDof);
        if (constrainedDof)
        {
            constraintCorrections.insert(std::make_pair(constrainedDof, listConstraintCorrection[i]));
        }
    }

    graphKineticEnergy.setDisplayed(traceKineticEnergy.getValue());
}



bool LMConstraintSolver::needPriorStatePropagation()
{
    return true;

    using core::behavior::BaseLMConstraint;
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
    using core::behavior::BaseLMConstraint ;
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

        simulation::MechanicalProjectJacobianMatrixVisitor().execute(this->getContext());
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
        else
        {
            simulation::MechanicalProjectVelocityVisitor projectVel(this->getContext()->getTime());
            projectVel.execute(this->getContext());
        }

        // calling writeConstraintEquations
        LMConstraintVisitor.setOrder(orderState);
        LMConstraintVisitor.setTags(getTags()).execute(this->getContext());

        simulation::MechanicalProjectJacobianMatrixVisitor().execute(this->getContext());

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
        else
        {
            simulation::MechanicalProjectPositionVisitor projectPos(this->getContext()->getTime());
            projectPos.execute(this->getContext());
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


    if (Order==VecId::position())      sofa::helper::AdvancedTimer::valSet("numConstraintsPosition", numConstraint);
    else if (Order==VecId::velocity()) sofa::helper::AdvancedTimer::valSet("numConstraintsVelocity", numConstraint);
    else if (Order==VecId::dx())       sofa::helper::AdvancedTimer::valSet("numConstraintsAcceleration", numConstraint);



    if (f_printLog.getValue())
    {
        if (Order==VecId::dx())            sout << "Applying the constraint on the acceleration"<<sendl;
        else if (Order==VecId::velocity()) sout << "Applying the constraint on the velocity"<<sendl;
        else if (Order==VecId::position()) sout << "Applying the constraint on the position"<<sendl;
    }

    if (W) delete W;
    if (c) delete c;
    if (Lambda) delete Lambda;

    setDofs.clear();
    dofUsed.clear();
    invMass_Ltrans.clear();
    return true;
}



bool LMConstraintSolver::buildSystem(double /*dt*/, VecId id)
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
#ifdef SOFA_DUMP_VISITOR_INFO
        else if (sofa::simulation::Visitor::IsExportStateVectorEnabled())
        {
            sofa::simulation::Visitor::printNode("Input_"+(*currentIt)->getName());
            sofa::simulation::Visitor::printVector(*currentIt, id);
            sofa::simulation::Visitor::printCloseNode("Input_"+(*currentIt)->getName());
        }
#endif
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
        const core::behavior::BaseMechanicalState* dofs=*it;
        SparseMatrixEigen L(numConstraint,dofs->getSize()*dofs->getDerivDimension());
        L.startFill(numConstraint*(1+dofs->getSize()));//TODO: give a better estimation of non-zero coefficients
        LMatrices.insert(std::make_pair(dofs, L));
    }
    buildLMatrices(orderState, LMConstraints, LMatrices, dofUsed);

    //Remove empty L Matrices
    for (DofToMatrix::iterator it=LMatrices.begin(); it!=LMatrices.end();)
    {
        SparseMatrixEigen& matrix= it->second;
        DofToMatrix::iterator itCurrent=it;
        ++it;
        if (!matrix.nonZeros()) //Empty Matrix: act as an obstacle
        {
            LMatrices.erase(itCurrent);
            invMassMatrix.erase(itCurrent->first);
            setDofs.erase(const_cast<sofa::core::behavior::BaseMechanicalState*>(itCurrent->first));
        }
    }

    //************************************************************
    // Building W=L0.M0^-1.L0^T + L1.M1^-1.L1^T + ... and M^-1.L^T
    //************************************************************
    //Store the matrix M^-1.L^T for each dof in order to apply the modification of the state
    SparseMatrixEigen WEi((int)numConstraint,(int)numConstraint);
    buildLeftMatrix(invMassMatrix, LMatrices,WEi, invMass_Ltrans);

    //Convert the Sparse Matrix AEi into a Dense Matrix-> faster access to elements, and possilibity to use a direct LU solution
    W=new MatrixEigen((int)numConstraint, (int)numConstraint);
    W->setZero((int)numConstraint, (int)numConstraint);


    MatrixEigen &WEigen = *W;
    for (int k=0; k<WEi.outerSize(); ++k)
        for (SparseMatrixEigen::InnerIterator it(WEi,k); it; ++it)
        {
            WEigen(it.row(),it.col()) = it.value();
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
        sout << "W= L0.M0^-1.L0^T + L1.M1^-1.L1^T + ...: "<<sendl;
        sout <<"\n" << *W << sendl;
        sout << "for a constraint: " << ""<<sendl;
        sout << "\n" << *c << sendl;
    }

    const helper::vector< BaseLMConstraint* > &LMConstraints=LMConstraintVisitor.getConstraints();

    Lambda=new VectorEigen(numConstraint);
    Lambda->setZero(numConstraint);

    bool solutionFound=solveConstraintSystemUsingGaussSeidel(orderState,LMConstraints, *W, *c, *Lambda);
#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::printCloseNode("SystemSolution");
#endif
    return solutionFound;
}



bool LMConstraintSolver::applyCorrection(double /*dt*/, VecId id)
{
    //************************************************************
    // Constraint Correction
    //************************************************************
#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::printNode("SystemCorrection");
#endif
    //************************************************************
    // Updating the state vectors
    // get the displacement. deltaState = M^-1.L^T.lambda : lambda being the solution of the system
    for (SetDof::iterator itDofs=setDofs.begin(); itDofs!=setDofs.end(); itDofs++)
    {
        sofa::core::behavior::BaseMechanicalState* dofs=*itDofs;
        const VectorEigen &LambdaVector=*Lambda;
        bool updateVelocities=!constraintVel.getValue();
        constraintStateCorrection(id, updateVelocities,invMass_Ltrans[dofs] , LambdaVector, dofUsed[dofs], dofs);

#ifdef SOFA_DUMP_VISITOR_INFO
        if (sofa::simulation::Visitor::IsExportStateVectorEnabled())
        {
            sofa::simulation::Visitor::printNode("Output_"+dofs->getName());
            sofa::simulation::Visitor::printVector(dofs, id);
            sofa::simulation::Visitor::printCloseNode("Output_"+dofs->getName());
        }
#endif
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
        const sofa::core::behavior::BaseMechanicalState* dofs=itDofs->first;
        const SparseMatrixEigen &invMass=itDofs->second;

        SparseMatrixEigen &L=LMatrix[dofs]; L.endFill();

        if (f_printLog.getValue()) sout << "Matrix L for " << dofs->getName() << "\n" << L << sendl;
        if (f_printLog.getValue())  sout << "Matrix M-1 for " << dofs->getName() << "\n" << invMass << sendl;

        const SparseMatrixEigen &invM_LTrans=invMass.marked<Eigen::SelfAdjoint|Eigen::UpperTriangular>()*L.transpose();
        invMass_Ltrans.insert( std::make_pair(dofs,invM_LTrans) );
        LeftMatrix += L*invM_LTrans;
    }
}


void LMConstraintSolver::buildLMatrices( ConstOrder Order,
        const helper::vector< core::behavior::BaseLMConstraint* > &LMConstraints,
        DofToMatrix &LMatrices,
        DofToMask &dofUsed) const
{
    typedef core::behavior::BaseMechanicalState::ConstraintBlock ConstraintBlock;
    unsigned constraintOffset=0;
    //We Take one by one the constraint, and write their equations in the corresponding matrix L
    for (unsigned int mat=0; mat<LMConstraints.size(); ++mat)
    {
        core::behavior::BaseLMConstraint *constraint=LMConstraints[mat];

        const core::behavior::BaseMechanicalState *dof1=constraint->getSimulatedMechModel1();
        const core::behavior::BaseMechanicalState *dof2=constraint->getSimulatedMechModel2();

        //Get the entries in the Vector of constraints corresponding to the constraint equations
        std::list< unsigned int > equationsUsed;
        constraint->getEquationsUsed(Order, equationsUsed);

        DofToMatrix::iterator itL1=LMatrices.find(dof1);
        if (itL1 != LMatrices.end())
        {
            buildLMatrix(itL1->first, equationsUsed, constraintOffset, itL1->second, dofUsed[dof1]);
        }

        DofToMatrix::iterator itL2=LMatrices.find(dof2);
        if (dof1 != dof2 && itL2 != LMatrices.end())
        {
            buildLMatrix(itL2->first, equationsUsed,constraintOffset, itL2->second,dofUsed[dof2]);
        }
        constraintOffset += equationsUsed.size();
    }
}



void LMConstraintSolver::buildInverseMassMatrices( const SetDof &setDofs, DofToMatrix& invMassMatrices)
{
    for (SetDof::const_iterator itDofs=setDofs.begin(); itDofs!=setDofs.end(); itDofs++)
    {
        const sofa::core::behavior::BaseMechanicalState* dofs=*itDofs;
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
        core::behavior::BaseMass *mass=dynamic_cast< core::behavior::BaseMass *>(dofs->getContext()->getMass());

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

void LMConstraintSolver::buildInverseMassMatrix( const sofa::core::behavior::BaseMechanicalState* /*mstate*/, const core::behavior::BaseConstraintCorrection* constraintCorrection, SparseMatrixEigen& invMass) const
{
    //Get the Full Matrix from the constraint correction
    FullMatrix<SReal> computationInvM;
    constraintCorrection->getComplianceMatrix(&computationInvM);
    //Then convert it into a Sparse Matrix: as it is done only at the init, or when topological changes occur, this should not be a burden for the framerate
    for (unsigned int i=0; i<computationInvM.rowSize(); ++i)
    {
        for (unsigned int j=i; j<computationInvM.colSize(); ++j)
        {
            SReal value=computationInvM.element(i,j);
            if (value != 0)
            {
                invMass.fill(i,j)=value;
            }
        }
    }
}

void LMConstraintSolver::buildInverseMassMatrix( const sofa::core::behavior::BaseMechanicalState* mstate, const core::behavior::BaseMass* mass, SparseMatrixEigen& invMass) const
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
        mapMEigen.marked<Eigen::SelfAdjoint|Eigen::UpperTriangular>().computeInverse(&invMEigen);

        //Store into the sparse matrix the block corresponding to the inverse of the mass matrix of a particle
        for (unsigned int r=0; r<dimensionDofs; ++r)
        {
            for (unsigned int c=r; c<dimensionDofs; ++c)
            {
                if (invMEigen(r,c) != 0)
                {
                    invMass.fill(i*dimensionDofs+r,i*dimensionDofs+c)=invMEigen(r,c);
                }
            }
        }
    }
}

void LMConstraintSolver::buildLMatrix( const sofa::core::behavior::BaseMechanicalState *dof,
        const std::list<unsigned int> &idxEquations, unsigned int constraintOffset,
        SparseMatrixEigen& L, sofa::helper::set< unsigned int > &dofUsed) const
{
    const unsigned int dimensionDofs=dof->getDerivDimension();
    typedef core::behavior::BaseMechanicalState::ConstraintBlock ConstraintBlock;
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

void LMConstraintSolver::buildRightHandTerm( ConstOrder Order, const helper::vector< core::behavior::BaseLMConstraint* > &LMConstraints, VectorEigen &c) const
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


bool LMConstraintSolver::solveConstraintSystemUsingGaussSeidel( ConstOrder Order, const helper::vector< core::behavior::BaseLMConstraint* > &LMConstraints, MatrixEigen &W, VectorEigen &c, VectorEigen &Lambda)
{
    if (f_printLog.getValue()) sout << "Using Gauss-Seidel solution"<<sendl;

    std::string orderName;
    switch (Order)
    {
    case BaseLMConstraint::ACC: orderName="Acceleration"; break;
    case BaseLMConstraint::VEL: orderName="Velocity"; break;
    case BaseLMConstraint::POS: orderName="Position"; break;
    }

    helper::vector<double> &vError=(*graphGSError.beginEdit())["Error "+ orderName];

    vError.push_back(c.sum());
    graphGSError.endEdit();

    VectorEigen LambdaPrevious=Lambda;

    //-- Initialization of X, solution of the system
    bool continueIteration=true;
    unsigned int iteration=0;
    double error=0;

    for (; iteration < numIterations.getValue() && continueIteration; ++iteration)
    {
        unsigned int idxConstraint=0;
        VectorEigen LambdaPreviousIteration;
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

                LambdaPreviousIteration = Lambda.block(idxConstraint,0,numConstraintToProcess,1);
                //TODO CHANGE by reference
                const MatrixEigen Wblock=W.block(idxConstraint,idxConstraint,numConstraintToProcess, numConstraintToProcess);

                constraint->LagrangeMultiplierEvaluation(Wblock.data(),c.data()+idxConstraint, Lambda.data()+idxConstraint,
                        constraintOrder[constraintEntry]);

                error=0;
                if (constraintOrder[constraintEntry]->isActive())
                {
                    const VectorEigen& LambdaBlock=Lambda.block(idxConstraint,0,numConstraintToProcess,1);
                    const MatrixEigen& LambdaBlockCorrection=(LambdaBlock-LambdaPreviousIteration);
                    c -= W.block(0,idxConstraint,numConstraint, numConstraintToProcess)*LambdaBlockCorrection;
                    error += LambdaBlockCorrection.norm();
                }
                else
                {
                    Lambda.block(idxConstraint,0,numConstraintToProcess,1).setZero();
//                      Lambda.block(idxConstraint,0,numConstraintToProcess,1)=LambdaPreviousIteration;
                    idxConstraint+=numConstraintToProcess;
                    continue;
                }
                //****************************************************************

                if (f_printLog.getValue())
                    sout <<"["<< iteration << "/" << numIterations.getValue() <<"]" <<"Error is : " <<  error << " for system of constraint " << idxConstraint<< "[" << numConstraintToProcess << "]" << "/" << W.cols()
                            << " between " << constraint->getSimulatedMechModel1()->getName() << " and " << constraint->getSimulatedMechModel2()->getName() << ""<<sendl;

                //****************************************************************
                //Update only if the error is higher than a threshold. If no "big changes" occured, we set: X[c]^(k) = X[c]^(k-1)
                if (error < maxError.getValue()/(SReal)numConstraintToProcess)
                {
                    Lambda.block(idxConstraint,0,numConstraintToProcess,1)=LambdaPreviousIteration;
                }
                else continueIteration=true;

                idxConstraint+=numConstraintToProcess;
            }
        }

        if (Order == BaseLMConstraint::VEL && traceKineticEnergy.getValue())
        {
            if (iteration == 0)
            {
                graphKineticEnergy.beginEdit()->clear();
                graphKineticEnergy.endEdit();
            }
            VectorEigen LambdaSave=Lambda;
            Lambda -= LambdaPrevious;
            if (continueIteration) computeKineticEnergy();
            Lambda = LambdaSave;
        }

        LambdaPrevious=Lambda;

        helper::vector<double> &vError=(*graphGSError.beginEdit())["Error "+ orderName];
        vError.push_back(c.norm());
        graphGSError.endEdit();

        if (this->f_printLog.getValue())
        {
            if (f_printLog.getValue()) sout << "ITERATION " << iteration << " ENDED\n"<<sendl;
        }
    }

    if (iteration == numIterations.getValue() && f_printLog.getValue())
        serr << "no convergence in Gauss-Seidel for " << orderName;

    if (f_printLog.getValue())
        sout << "Gauss-Seidel done in " << iteration << " iterations "<<sendl;

    if (Order == BaseLMConstraint::VEL && traceKineticEnergy.getValue()) return false;
    return true;
}

void LMConstraintSolver::constraintStateCorrection(VecId Order, bool isPositionChangesUpdateVelocity,
        const SparseMatrixEigen  &invM_Ltrans,
        const VectorEigen  &c,
        const sofa::helper::set< unsigned int > &dofUsed,
        sofa::core::behavior::BaseMechanicalState* dofs) const
{

    //Correct Dof
    //    operation: M0^-1.L0^T.lambda -> A

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


void LMConstraintSolver::computeKineticEnergy()
{
    helper::vector<double> &vError=(*graphKineticEnergy.beginEdit())["KineticEnergy"];

    applyCorrection(0,VecId::velocity());
    double kineticEnergy=0;
    for (SetDof::const_iterator itDofs=setDofs.begin(); itDofs!=setDofs.end(); itDofs++)
    {
        const sofa::core::behavior::BaseMechanicalState* dofs=*itDofs;
        const core::behavior::BaseMass *mass=dynamic_cast< core::behavior::BaseMass *>(dofs->getContext()->getMass());
        if (mass) kineticEnergy += mass->getKineticEnergy();
    }
    vError.push_back(kineticEnergy);
    graphKineticEnergy.endEdit();
}

void LMConstraintSolver::handleEvent(core::objectmodel::Event *e)
{
    if (dynamic_cast<sofa::simulation::AnimateBeginEvent*>(e))
    {
        graphGSError.beginEdit()->clear();
        graphGSError.endEdit();
    }
}



int LMConstraintSolverClass = core::RegisterObject("A Constraint Solver working specifically with LMConstraint based components")
        .add< LMConstraintSolver >();

SOFA_DECL_CLASS(LMConstraintSolver);


} // namespace constraint

} // namespace component

} // namespace sofa
