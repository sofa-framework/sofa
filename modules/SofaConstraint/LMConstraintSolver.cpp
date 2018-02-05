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

#include <SofaConstraint/LMConstraintSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <SofaBaseLinearSolver/FullMatrix.h>
#include <SofaBaseLinearSolver/FullVector.h>
#include <sofa/simulation/AnimateBeginEvent.h>

#include <sofa/defaulttype/Quat.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/AdvancedTimer.h>

#include <Eigen/LU>
#include <Eigen/QR>

namespace sofa
{

namespace component
{

namespace constraintset
{

using linearsolver::FullVector;
using linearsolver::FullMatrix;

LMConstraintSolver::LMConstraintSolver()
    : constraintAcc( initData( &constraintAcc, false, "constraintAcc", "Constraint the acceleration"))
    , constraintVel( initData( &constraintVel, true, "constraintVel", "Constraint the velocity"))
    , constraintPos( initData( &constraintPos, true, "constraintPos", "Constraint the position"))
    , numIterations( initData( &numIterations, (unsigned int)25, "numIterations", "Number of iterations for Gauss-Seidel when solving the Constraints"))
    , maxError( initData( &maxError, 0.0000001, "maxError", "threshold for the residue of the Gauss-Seidel algorithm"))
    , graphGSError( initData(&graphGSError,"graphGSError","Graph of residuals at each iteration") )
    , traceKineticEnergy( initData( &traceKineticEnergy, false, "traceKineticEnergy", "Trace the evolution of the Kinetic Energy throughout the solution of the system"))
    , graphKineticEnergy( initData(&graphKineticEnergy,"graphKineticEnergy","Graph of the kinetic energy of the system") )
    , LMConstraintVisitor( sofa::core::ExecParams::defaultInstance() )
{
    graphGSError.setGroup("Statistics");
    graphGSError.setWidget("graph");

    graphKineticEnergy.setGroup("Statistics");
    graphKineticEnergy.setWidget("graph");

    traceKineticEnergy.setGroup("Statistics");
    this->f_listening.setValue(true);

    numIterations.setRequired(true);
    maxError.setRequired(true);
}

void LMConstraintSolver::init()
{
    sofa::core::behavior::ConstraintSolver::init();

    helper::vector< core::behavior::BaseConstraintCorrection* > listConstraintCorrection;
    ((simulation::Node*) getContext())->get<core::behavior::BaseConstraintCorrection>(&listConstraintCorrection, core::objectmodel::BaseContext::SearchDown);
    for (unsigned int i=0; i<listConstraintCorrection.size(); ++i)
    {
        core::behavior::BaseMechanicalState* constrainedDof;
        listConstraintCorrection[i]->getContext()->get(constrainedDof);
        if (constrainedDof)
        {
            constraintCorrections.insert(std::make_pair(constrainedDof, listConstraintCorrection[i]));
            listConstraintCorrection[i]->removeConstraintSolver(this);
            listConstraintCorrection[i]->addConstraintSolver(this);
        }
    }

    graphKineticEnergy.setDisplayed(traceKineticEnergy.getValue());
}
void LMConstraintSolver::removeConstraintCorrection(core::behavior::BaseConstraintCorrection * /*s*/)
{
    //TODO: remove the pair containing s from constraintCorrections
}

void LMConstraintSolver::convertSparseToDense(const SparseMatrixEigen& sparseM, MatrixEigen& denseM) const
{
    denseM=MatrixEigen::Zero(sparseM.rows(), sparseM.cols());

    for (int k=0; k<sparseM.outerSize(); ++k)
        for (SparseMatrixEigen::InnerIterator it(sparseM,k); it; ++it)
        {
            denseM(it.row(),it.col()) = it.value();
        }
}

bool LMConstraintSolver::needPriorStatePropagation(core::ConstraintParams::ConstOrder order) const
{
    using core::behavior::BaseLMConstraint;
    bool needPriorPropagation=false;
    {
        helper::vector< BaseLMConstraint* > c;
        this->getContext()->get<BaseLMConstraint>(&c, core::objectmodel::BaseContext::SearchDown);
        for (unsigned int i=0; i<c.size(); ++i)
        {
            if (!c[i]->isCorrectionComputedWithSimulatedDOF(order))
            {
                needPriorPropagation=true;
                msg_info() << "Propagating the State because of "<< c[i]->getName() ;
                break;
            }
        }
    }
    return needPriorPropagation;
}

bool LMConstraintSolver::prepareStates(const core::ConstraintParams *cparams, MultiVecId id, MultiVecId /*res2*/)
{
    //Get the matrices through mappings
    //************************************************************
    // Update the State of the Mapped dofs                      //
    //************************************************************
    core::MechanicalParams mparams;
#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::TRACE_ARGUMENT arg;
#endif
    using core::behavior::BaseLMConstraint ;
    LMConstraintVisitor.clear();

    VecId vid = id.getDefaultId();

    orderState=cparams->constOrder();
    LMConstraintVisitor.setMultiVecId(id);
    if      (orderState==core::ConstraintParams::ACC)
    {
        if (!constraintAcc.getValue()) return false;
        if (needPriorStatePropagation(orderState))
        {
            simulation::MechanicalPropagateDxVisitor propagateState(&mparams, core::VecDerivId(vid), false, false);
            propagateState.execute(this->getContext());
        }
        // calling writeConstraintEquations
        LMConstraintVisitor.setOrder(orderState);
        LMConstraintVisitor.setTags(getTags()).execute(this->getContext());

        msg_info() << "prepareStates for accelerations";

        simulation::MechanicalProjectJacobianMatrixVisitor(&mparams).execute(this->getContext());
#ifdef SOFA_DUMP_VISITOR_INFO
        arg.push_back(std::make_pair("Order", "Acceleration"));
#endif
    }
    else if (orderState==core::ConstraintParams::VEL)
    {
        if (!constraintVel.getValue()) return false;

        msg_info() << "prepareStates for velocities";

        simulation::MechanicalProjectVelocityVisitor projectState(&mparams, this->getContext()->getTime(), core::VecDerivId(vid));
        projectState.execute(this->getContext());
        if (needPriorStatePropagation(orderState))
        {
            simulation::MechanicalPropagateOnlyVelocityVisitor propagateState(&mparams, 0.0, core::VecDerivId(vid),false);
            propagateState.execute(this->getContext());
        }

        // calling writeConstraintEquations
        LMConstraintVisitor.setOrder(orderState);
        LMConstraintVisitor.setTags(getTags()).execute(this->getContext());

        simulation::MechanicalProjectJacobianMatrixVisitor(&mparams).execute(this->getContext());

#ifdef SOFA_DUMP_VISITOR_INFO
        arg.push_back(std::make_pair("Order", "Velocity"));
#endif

    }
    else
    {
        if (!constraintPos.getValue()) return false;
        msg_info() << "prepareStates for positions";

        simulation::MechanicalProjectPositionVisitor projectPos(&mparams, this->getContext()->getTime(), core::VecCoordId(vid));
        projectPos.execute(this->getContext());
        if (needPriorStatePropagation(orderState))
        {
            simulation::MechanicalPropagateOnlyPositionVisitor propagateState(&mparams, 0.0, core::VecCoordId(vid), false);
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
        msg_info()    << "LMConstraintSolver::prepareStates, constraint " << LMConstraints[mat]->getName()
                    << " between "<< LMConstraints[mat]->getSimulatedMechModel1()->getName()
                    << " and " << LMConstraints[mat]->getSimulatedMechModel2()->getName()
                    <<", add "<<  LMConstraints[mat]->getNumConstraint(orderState)
                    << " constraint for order "<< orderState ;
    }
    if (numConstraint == 0)
    {
        return false; //Nothing to solve
    }


    if      (orderState==core::ConstraintParams::POS)
    {
        sofa::helper::AdvancedTimer::valSet("numConstraintsPosition", numConstraint);
        msg_info() << "Applying the constraint on the position";
    }
    else if (orderState==core::ConstraintParams::VEL)
    {
        sofa::helper::AdvancedTimer::valSet("numConstraintsVelocity", numConstraint);
        msg_info() << "Applying the constraint on the velocity" ;
    }
    else if (orderState==core::ConstraintParams::ACC)
    {
        sofa::helper::AdvancedTimer::valSet("numConstraintsAcceleration", numConstraint);
        msg_info() << "Applying the constraint on the acceleration" ;
    }
    else msg_warning() << "Order Not recognized " << orderState ;

    setDofs.clear();
    dofUsed.clear();
    invMass_Ltrans.clear();
    return true;
}

bool LMConstraintSolver::buildSystem(const core::ConstraintParams *cParams, MultiVecId id, MultiVecId /*res2*/)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::printNode("SystemCreation");
#endif

    sofa::helper::AdvancedTimer::stepBegin("SolveConstraints "  + id.getName() + " BuildSystem Prepare");
    const helper::vector< sofa::core::behavior::BaseLMConstraint* > &LMConstraints=LMConstraintVisitor.getConstraints();
    //Informations to build the matrices
    //Dofs to be constrained
    for (unsigned int mat=0; mat<LMConstraints.size(); ++mat)
    {
        sofa::core::behavior::BaseLMConstraint *constraint=LMConstraints[mat];
        if (constraint->getNumConstraint(cParams->constOrder()))
        {
            setDofs.insert(constraint->getSimulatedMechModel1());
            setDofs.insert(constraint->getSimulatedMechModel2());
        }
    }
    for (SetDof::iterator it=setDofs.begin(); it!=setDofs.end();)
    {
        SetDof::iterator currentIt=it;
        ++it;

        if (!(*currentIt)->getContext()->getMass())
        {
            setDofs.erase(currentIt);
        }
#ifdef SOFA_DUMP_VISITOR_INFO
        else if (sofa::simulation::Visitor::IsExportStateVectorEnabled())
        {
            sofa::simulation::Visitor::printNode("Input_"+(*currentIt)->getName());
            sofa::simulation::Visitor::printVector(*currentIt, id.getDefaultId());
            sofa::simulation::Visitor::printCloseNode("Input_"+(*currentIt)->getName());
        }
#endif
    }
    sofa::helper::AdvancedTimer::stepEnd  ("SolveConstraints "  + id.getName() + " BuildSystem Prepare");


    //Store the indices used in order to speed up the correction propagation
    //************************************************************
    // Build the Right Hand Term
    //************************************************************
    sofa::helper::AdvancedTimer::stepBegin("SolveConstraints "  + id.getName() + " BuildSystem C");
    c = VectorEigen::Zero((int)numConstraint);
    buildRightHandTerm(LMConstraints, c, id, orderState);
    sofa::helper::AdvancedTimer::stepEnd("SolveConstraints "  + id.getName() + " BuildSystem C");


    //************************************************************
    // Build M^-1
    //************************************************************
    sofa::helper::AdvancedTimer::stepBegin("SolveConstraints "  + id.getName() + " BuildSystem M^-1");
    buildInverseMassMatrices(setDofs, invMassMatrix);
    sofa::helper::AdvancedTimer::stepEnd("SolveConstraints "  + id.getName() + " BuildSystem M^-1");

    //************************************************************
    //Building L
    //************************************************************
    sofa::helper::AdvancedTimer::stepBegin("SolveConstraints "  + id.getName() + " BuildSystem L");
    LMatrices.clear();
    //Init matrices to the good size
    for (SetDof::iterator it=setDofs.begin(); it!=setDofs.end(); ++it)
    {
        const core::behavior::BaseMechanicalState* dofs=*it;
        SparseMatrixEigen L(numConstraint,dofs->getSize()*dofs->getDerivDimension());
        L.reserve(numConstraint*(1+dofs->getSize()));//TODO: give a better estimation of non-zero coefficients
        LMatrices.insert(std::make_pair(dofs, L));
    }
    buildLMatrices(orderState, LMConstraints, LMatrices, dofUsed);

    //Remove empty L Matrices
    for (DofToMatrix::iterator it=LMatrices.begin(); it!=LMatrices.end();)
    {
        SparseMatrixEigen& matrix= it->second;
        matrix.finalize();
        DofToMatrix::iterator itCurrent=it;
        ++it;
        if (!matrix.nonZeros()) //Empty Matrix: act as an obstacle
        {
            invMassMatrix.erase(itCurrent->first);
            setDofs.erase(const_cast<sofa::core::behavior::BaseMechanicalState*>(itCurrent->first));
            LMatrices.erase(itCurrent);
        }

    }
    sofa::helper::AdvancedTimer::stepEnd("SolveConstraints "  + id.getName() + " BuildSystem L");


    //************************************************************
    // Building W=L0.M0^-1.L0^T + L1.M1^-1.L1^T + ... and M^-1.L^T
    //************************************************************
    //Store the matrix M^-1.L^T for each dof in order to apply the modification of the state
    sofa::helper::AdvancedTimer::stepBegin("SolveConstraints "  + id.getName() + " BuildSystem W");
    SparseMatrixEigen WEi((int)numConstraint,(int)numConstraint);
    buildLeftMatrix(invMassMatrix, LMatrices,WEi, invMass_Ltrans);
    sofa::helper::AdvancedTimer::stepEnd("SolveConstraints "  + id.getName() + " BuildSystem W");

    sofa::helper::AdvancedTimer::stepBegin("SolveConstraints "  + id.getName() + " BuildSystem conversionW");
    //Convert the Sparse Matrix AEi into a Dense Matrix-> faster access to elements, and possilibity to use a direct LU solution
    convertSparseToDense(WEi,W);

    sofa::helper::AdvancedTimer::stepEnd("SolveConstraints "  + id.getName() + " BuildSystem conversionW");
#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::printCloseNode("SystemCreation");
#endif
    return true;
}

bool LMConstraintSolver::solveSystem(const core::ConstraintParams* cParams, MultiVecId id, MultiVecId /*res2*/)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::printNode("SystemConstraintSolution");
#endif

    //************************************************************
    // Solving the System using Eigen2
    //************************************************************
    msg_info() << "W= L0.M0^-1.L0^T + L1.M1^-1.L1^T + ...: " << msgendl
               << W << msgendl
               << "for a constraint: " << msgendl
               << c ;

    const helper::vector< sofa::core::behavior::BaseLMConstraint* > &LMConstraints=LMConstraintVisitor.getConstraints();

    //"Cold" start
    Lambda=VectorEigen::Zero(numConstraint);
    bool solutionFound=solveConstraintSystemUsingGaussSeidel(id, cParams->constOrder(), LMConstraints,
            W, c, Lambda);
#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::printCloseNode("SystemConstraintSolution");
#endif
    return solutionFound;
}

bool LMConstraintSolver::applyCorrection(const core::ConstraintParams* cparams, MultiVecId id, MultiVecId /*res2*/)
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
    for (SetDof::iterator itDofs=setDofs.begin(); itDofs!=setDofs.end(); ++itDofs)
    {
        sofa::core::behavior::BaseMechanicalState* dofs=*itDofs;
        bool updateVelocities=!constraintVel.getValue();
        VecId v = id.getId(dofs);
        constraintStateCorrection(v, cparams->constOrder(), updateVelocities,invMass_Ltrans[dofs] , Lambda, dofUsed[dofs], dofs);

#ifdef SOFA_DUMP_VISITOR_INFO
        if (sofa::simulation::Visitor::IsExportStateVectorEnabled())
        {
            sofa::simulation::Visitor::printNode("Output_"+dofs->getName());
            sofa::simulation::Visitor::printVector(dofs, v);
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
    for (SetDof::const_iterator itDofs=setDofs.begin(); itDofs!=setDofs.end(); ++itDofs)
    {
        const sofa::core::behavior::BaseMechanicalState* dofs=*itDofs;
        const SparseMatrixEigen &invMass=invMassMatrix.find(dofs)->second;
        const SparseMatrixEigen &L=LMatrix[dofs];

        msg_info() << "Matrix L for " << dofs->getName() << "\n" << L << msgendl
                   << "Matrix M-1 for " << dofs->getName() << "\n" << invMass ;

        const SparseMatrixEigen &invM_LTrans=invMass*L.transpose();
        invMass_Ltrans.insert( std::make_pair(dofs,invM_LTrans) );
        LeftMatrix += L*invM_LTrans;
    }
}

void LMConstraintSolver::buildLMatrices( ConstOrder Order,
        const helper::vector< core::behavior::BaseLMConstraint* > &LMConstraints,
        DofToMatrix &LMatrices,
        DofToMask &dofUsed) const
{
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

        if (equationsUsed.empty()) continue;

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
    for (SetDof::const_iterator itDofs=setDofs.begin(); itDofs!=setDofs.end(); ++itDofs)
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
        core::behavior::BaseMass *mass=dofs->getContext()->getMass();

        if (needToConstructMassMatrix)
        {
            SparseMatrixEigen invMass(dofs->getSize()*dimensionDofs, dofs->getSize()*dimensionDofs);
            invMass.reserve(dofs->getSize()*dimensionDofs*dimensionDofs);

            DofToConstraintCorrection::iterator constraintCorrectionFound=constraintCorrections.find(dofs);
            if (constraintCorrectionFound != constraintCorrections.end())
            {
                buildInverseMassMatrix(dofs, constraintCorrectionFound->second, invMass);
            }
            else
            {
                buildInverseMassMatrix(dofs, mass, invMass);
            }
            invMass.finalize();
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
    for (Index i=0; i<computationInvM.rowSize(); ++i)
    {
        invMass.startVec(i);
        for (Index j=0; j<computationInvM.colSize(); ++j)
        {
            SReal value=computationInvM.element(i,j);
            if (value != 0)
            {
                invMass.insertBack(i,j)=value;
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

    for (size_t i=0; i<mstate->getSize(); ++i)
    {
        mass->getElementMass(i,&computationM);

        //Translate the FullMatrix into a Eigen Matrix to invert it
        MatrixEigen mapMEigen=Eigen::Map<MatrixEigen>(computationM[0],(int)computationM.rowSize(),(int)computationM.colSize());
        invMEigen=mapMEigen.inverse();

        //Store into the sparse matrix the block corresponding to the inverse of the mass matrix of a particle
        for (unsigned int r=0; r<dimensionDofs; ++r)
        {
            invMass.startVec(i*dimensionDofs+r);
            for (unsigned int c=0; c<dimensionDofs; ++c)
            {
                if (invMEigen(r,c) != 0)
                {
                    invMass.insertBack(i*dimensionDofs+r,i*dimensionDofs+c)=invMEigen(r,c);
                }
            }
        }
    }
}

void LMConstraintSolver::buildLMatrix( const sofa::core::behavior::BaseMechanicalState *dof,
        const std::list<unsigned int> &idxEquations, unsigned int constraintOffset,
        SparseMatrixEigen& L, std::set< unsigned int > &dofUsed) const
{
    const unsigned int dimensionDofs=dof->getDerivDimension();
    typedef core::behavior::BaseMechanicalState::ConstraintBlock ConstraintBlock;
    //Get blocks of values from the Mechanical States
    std::list< ConstraintBlock > blocks=dof->constraintBlocks( idxEquations );


    //Fill the matrices
    const unsigned int numEquations=idxEquations.size();

    for (unsigned int eq=0; eq<numEquations; ++eq)
    {
        const int idxRow=constraintOffset+eq;

        for (std::list< ConstraintBlock >::const_iterator itBlock=blocks.begin(); itBlock!=blocks.end(); ++itBlock)
        {
            const ConstraintBlock &b=(*itBlock);
            const defaulttype::BaseMatrix &m=b.getMatrix();
            const unsigned int column=b.getColumn()*dimensionDofs;

            for (Index j=0; j<m.colSize(); ++j)
            {
                SReal value=m.element(eq,j);
                if (value!=0)
                {
                    L.insert(idxRow, column+j) = m.element(eq,j);
                    dofUsed.insert((column+j)/dimensionDofs);
                }
            }
        }
    }
    for (std::list< ConstraintBlock >::iterator itBlock=blocks.begin(); itBlock!=blocks.end(); ++itBlock)
    {
        delete itBlock->getMatrix();
    }
}

void LMConstraintSolver::buildRightHandTerm( const helper::vector< core::behavior::BaseLMConstraint* > &LMConstraints, VectorEigen &c,
        MultiVecId /*id*/, ConstOrder Order) const
{

    FullVector<SReal> c_fullvector(c.data(), (Index)c.rows());
    for (unsigned int mat=0; mat<LMConstraints.size(); ++mat)  LMConstraints[mat]->getConstraintViolation(&c_fullvector, Order);
}

bool LMConstraintSolver::solveConstraintSystemUsingGaussSeidel( MultiVecId id, ConstOrder Order,
        const helper::vector< core::behavior::BaseLMConstraint* > &LMConstraints,
        const MatrixEigen &W, const VectorEigen &c, VectorEigen &Lambda)
{
    msg_info() <<  "Using Gauss-Seidel solution" ;

    std::string orderName;
    switch (Order)
    {
    case core::ConstraintParams::ACC :
        orderName="Acceleration";
        break;
    case core::ConstraintParams::VEL :
        orderName="Velocity";
        break;
    case core::ConstraintParams::POS :
    case core::ConstraintParams::POS_AND_VEL :
        orderName="Position";
        break;
    }

    const SReal invNormC=1.0/c.norm();

    helper::vector<double> &vError=(*graphGSError.beginEdit())["Error "+ orderName];

    vError.push_back(c.sum());
    graphGSError.endEdit();

    VectorEigen LambdaPrevious=Lambda;

    //Store the invalid block of the W matrix
    std::set< int > emptyBlock;
    helper::vector< std::pair<MatrixEigen,VectorEigen> > constraintToBlock;
    helper::vector< MatrixEigen> constraintInvWBlock;
    helper::vector< MatrixEigen> constraintWBlock;


    //Preparation Step:
    {
        //Iterate on all the Constraint components
        unsigned int idxConstraint=0;
        for (unsigned int componentConstraint=0; componentConstraint<LMConstraints.size(); ++componentConstraint)
        {
            const sofa::core::behavior::BaseLMConstraint *constraint=LMConstraints[componentConstraint];
            //Get the vector containing all the constraint stored in one component
            const helper::vector< sofa::core::behavior::ConstraintGroup* > &constraintOrder=constraint->getConstraintsOrder(Order);

            unsigned int numConstraintToProcess=0;
            for (unsigned int constraintEntry=0; constraintEntry<constraintOrder.size(); ++constraintEntry, idxConstraint += numConstraintToProcess)
            {
                numConstraintToProcess=constraintOrder[constraintEntry]->getNumConstraint();

                const MatrixEigen &wConstraint=W.block(idxConstraint,idxConstraint,numConstraintToProcess, numConstraintToProcess);
                if (wConstraint.diagonal().isZero(1e-15)) emptyBlock.insert(idxConstraint);
                constraintWBlock.push_back(wConstraint);
                constraintInvWBlock.push_back(wConstraint.inverse());

                const VectorEigen &cb=c.block(idxConstraint         , 0,
                        numConstraintToProcess, 1);
                const MatrixEigen &wb=W.block(idxConstraint         , 0,
                        numConstraintToProcess, W.cols());
                constraintToBlock.push_back(std::make_pair(wb,cb));
            }
        }
    }
    //-- Initialization of X, solution of the system
    bool continueIteration=true;
    bool correctionDone=true;
    unsigned int iteration=0;

    const unsigned int numIterationsGS=numIterations.getValue();
    const unsigned int numLMConstraintComponents=LMConstraints.size();

    for (; iteration < numIterationsGS && continueIteration && correctionDone; ++iteration)
    {
        unsigned int idxConstraint=0;
        VectorEigen LambdaPreviousIteration;
        continueIteration=false;
        correctionDone=false;
        unsigned int idxBlocks=0;
        //Iterate on all the Constraint components

        for (unsigned int componentConstraint=0; componentConstraint<numLMConstraintComponents; ++componentConstraint)
        {
            sofa::core::behavior::BaseLMConstraint *constraint=LMConstraints[componentConstraint];
            //Get the vector containing all the constraint stored in one component
            const helper::vector< sofa::core::behavior::ConstraintGroup* > &constraintOrder=constraint->getConstraintsOrder(Order);

            unsigned int numConstraintToProcess=0;
            for (unsigned int constraintEntry=0; constraintEntry<constraintOrder.size(); ++constraintEntry, idxConstraint += numConstraintToProcess, ++idxBlocks)
            {
                //-------------------------------------
                //Initialize the variables, and store X^(k-1) in previousIteration
                numConstraintToProcess=constraintOrder[constraintEntry]->getNumConstraint();

                //Invalid constraints: due to projective constraints, or constraint expressed for Obstacle objects: we just ignore them
                if (emptyBlock.find(idxConstraint) != emptyBlock.end()) continue;

                LambdaPreviousIteration = Lambda.block(idxConstraint,0,numConstraintToProcess,1);
                //Set to Zero sigma for the current set of constraints
                Lambda.block(idxConstraint,0,numConstraintToProcess,1).setZero();


                const MatrixEigen &Wblock   =constraintWBlock[idxBlocks];
                const MatrixEigen &invWblock=constraintInvWBlock[idxBlocks];

                const std::pair<MatrixEigen,VectorEigen> &blocks=constraintToBlock[idxBlocks];
                const VectorEigen &cb=blocks.second;
                const MatrixEigen &wb=blocks.first;

                //Compute Sigma
                VectorEigen sigma = cb; sigma.noalias() = sigma - (wb * Lambda);

                VectorEigen newLambda; newLambda.noalias() = invWblock*sigma;
                constraint->LagrangeMultiplierEvaluation(Wblock.data(),sigma.data(), newLambda.data(),
                        constraintOrder[constraintEntry]);

                //****************************************************************
                if (!constraintOrder[constraintEntry]->isActive())
                {
                    Lambda.block(idxConstraint,0,numConstraintToProcess,1).setZero();

                    msg_info() << "["<< iteration << "/" << numIterations.getValue() <<"] ###Deactivated###"
                               << (LambdaPreviousIteration - Lambda.block(idxConstraint,0,numConstraintToProcess,1)).sum()
                               << " for system n°" << idxConstraint << "/" << W.cols() << " of " << numConstraintToProcess << " equations "
                               << " between " << constraint->getSimulatedMechModel1()->getName() << " and " << constraint->getSimulatedMechModel2()->getName() ;

                    correctionDone |=    ( (LambdaPreviousIteration - Lambda.block(idxConstraint,0,numConstraintToProcess,1)).sum() != 0);
                }
                else
                {
                    Lambda.block(idxConstraint,0,numConstraintToProcess,1)=newLambda;

                    msg_info() <<"["<< iteration << "/" << numIterations.getValue() <<"] "
                               << (LambdaPreviousIteration - Lambda.block(idxConstraint,0,numConstraintToProcess,1)).sum()
                               << " for system n°" << idxConstraint << "/" << W.cols() << " of " << numConstraintToProcess << " equations "
                               << " between " << constraint->getSimulatedMechModel1()->getName() << " and " << constraint->getSimulatedMechModel2()->getName() ;

                    correctionDone |=  ( (LambdaPreviousIteration - Lambda.block(idxConstraint,0,numConstraintToProcess,1)).sum() != 0);
                }
            }
        }

        if (Order == core::ConstraintParams::VEL && traceKineticEnergy.getValue())
        {
            if (iteration == 0)
            {
                graphKineticEnergy.beginEdit()->clear();
                graphKineticEnergy.endEdit();
            }
            VectorEigen LambdaSave=Lambda;
            Lambda.noalias() = Lambda - LambdaPrevious;
            computeKineticEnergy(id);
            Lambda = LambdaSave;
            LambdaPrevious=Lambda;
        }


        const SReal residual=(c-W*Lambda).norm()*invNormC;
        vError.push_back( residual );
        graphGSError.endEdit();
        continueIteration = (residual > maxError.getValue());

        msg_info() << "ITERATION " << iteration << " ENDED:  Residual for iteration " << iteration << " = " << residual ;
    }

    msg_info_when(iteration == numIterations.getValue()) << "no convergence in Gauss-Seidel for " << orderName;

    msg_info() << "Gauss-Seidel done in " << iteration << " iterations ";

    if (Order == core::ConstraintParams::VEL && traceKineticEnergy.getValue()) return false;
    return true;
}

void LMConstraintSolver::constraintStateCorrection(VecId id,  core::ConstraintParams::ConstOrder order,
        bool isPositionChangesUpdateVelocity,
        const SparseMatrixEigen  &invM_Ltrans,
        const VectorEigen  &c,
        const std::set< unsigned int > &dofUsed,
        sofa::core::behavior::BaseMechanicalState* dofs) const
{

    //Correct Dof
    //    operation: M0^-1.L0^T.lambda -> A

    VectorEigen A;
    /* This seems very redundant, but the SparseTimeDenseProduct seems buggy when the dense rhs
    and sparse lhs uses different storage order. This is always the case with a sparse row major
    and a dense vector (col major otherwise compile time assert )*/
    //Eigen::SparseMatrix<SReal> invM_Ltrans_colMajor = Eigen::SparseMatrix<SReal>(invM_Ltrans);

    A.noalias() = invM_Ltrans*c;

    msg_info() << "M^-1.L^T " << "\n" << invM_Ltrans << msgendl
               << "Lambda " << dofs->getName() << "\n" << c << msgendl
               << "Correction " << dofs->getName() << "\n" << A ;

    const unsigned int dimensionDofs=dofs->getDerivDimension();

    //In case of position correction, we need to update the velocities
    if (order==core::ConstraintParams::POS)
    {
        //Detect Rigid Bodies
        if (dofs->getCoordDimension() == 7 && dofs->getDerivDimension() == 6)
        {
            VectorEigen Acorrection=VectorEigen::Zero(dofs->getSize()*(3+4));
            //We have to transform the Euler Rotations into a quaternion
            unsigned int offset=0;

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
            FullVector<SReal> v(Acorrection.data(), (Index)Acorrection.rows());

            msg_info() << "Lambda Corrected for Rigid " << "\n" << Acorrection ;

            dofs->addFromBaseVectorSameSize(id,&v,offset );

        }
        else
        {
            std::set< unsigned int >::const_iterator it;
            for (it=dofUsed.begin(); it!=dofUsed.end(); ++it)
            {
                unsigned int offset=(*it);
                FullVector<SReal> v(&(A.data()[offset*dimensionDofs]),dimensionDofs);
                dofs->addFromBaseVectorDifferentSize(id,&v,offset );
            }
        }

        if (isPositionChangesUpdateVelocity)
        {
            const double h=1.0/getContext()->getDt();

            std::set< unsigned int >::const_iterator it;
            for (it=dofUsed.begin(); it!=dofUsed.end(); ++it)
            {
                unsigned int offset=(*it);
                FullVector<SReal> v(&(A.data()[offset*dimensionDofs]),dimensionDofs);
                for (unsigned int i=0; i<dimensionDofs; ++i) v[i]*=h;
                dofs->addFromBaseVectorDifferentSize(id,&v,offset );
            }
        }

    }
    else
    {
        std::set< unsigned int >::const_iterator it;
        for (it=dofUsed.begin(); it!=dofUsed.end(); ++it)
        {
            unsigned int offset=(*it);
            FullVector<SReal> v(&(A.data()[offset*dimensionDofs]),dimensionDofs);
            dofs->addFromBaseVectorDifferentSize(id,&v,offset );
        }

    }
}

void LMConstraintSolver::computeKineticEnergy(MultiVecId id)
{
    helper::vector<double> &vError=(*graphKineticEnergy.beginEdit())["KineticEnergy"];

    core::ConstraintParams cparam;
    cparam.setOrder(core::ConstraintParams::VEL);

    applyCorrection(&cparam, id);

    double kineticEnergy=0;
    for (SetDof::const_iterator itDofs=setDofs.begin(); itDofs!=setDofs.end(); ++itDofs)
    {
        const sofa::core::behavior::BaseMechanicalState* dofs=*itDofs;
        const core::behavior::BaseMass *mass=dofs->getContext()->getMass();
        if (mass) kineticEnergy += mass->getKineticEnergy();
    }
    vError.push_back(kineticEnergy);
    graphKineticEnergy.endEdit();
}

void LMConstraintSolver::handleEvent(core::objectmodel::Event *e)
{
    if (sofa::simulation::AnimateBeginEvent::checkEventType(e))
    {
        graphGSError.beginEdit()->clear();
        graphGSError.endEdit();
    }
}



int LMConstraintSolverClass = core::RegisterObject("A Constraint Solver working specifically with LMConstraint based components")
        .add< LMConstraintSolver >();

SOFA_DECL_CLASS(LMConstraintSolver);


} // namespace constraintset

} // namespace component

} // namespace sofa
