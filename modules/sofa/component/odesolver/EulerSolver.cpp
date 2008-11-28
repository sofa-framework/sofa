/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
 *                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/helper/Quater.h>
#include <sofa/core/ObjectFactory.h>
#include <math.h>
#include <iostream>

#ifdef SOFA_HAVE_LAPACK
#include <sofa/component/linearsolver/LapackOperations.h>
#endif

#define SOFA_NO_VMULTIOP

using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace odesolver
{

using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace core::componentmodel::behavior;

int EulerSolverClass = core::RegisterObject("A simple explicit time integrator")
        .add< EulerSolver >()
        .addAlias("Euler")
        ;

SOFA_DECL_CLASS(Euler);

EulerSolver::EulerSolver()
    :
#ifdef SOFA_HAVE_LAPACK
    constraintAcc( initData( &constraintAcc, false, "constraintAcc", "Constraint the acceleration")),
    constraintVel( initData( &constraintVel, false, "constraintVel", "Constraint the velocity")),
    constraintPos( initData( &constraintPos, false, "constraintPos", "Constraint the position")),
    constraintResolution( initData( &constraintResolution, false, "constraintResolution", "Using Gauss-Seidel to solve the constraint.\nOtherwise, use direct LU resolution.")),
    numIterations( initData( &numIterations, (unsigned int)10, "numIterations", "Number of iterations for Gauss-Seidel")),
    maxError( initData( &maxError, 0.00001, "maxError", "Max error for Gauss-Seidel algorithm")),
#endif
    symplectic( initData( &symplectic, true, "symplectic", "If true, the velocities are updated before the positions and the method is symplectic (more robust). If false, the positions are updated before the velocities (standard Euler, less robust).") )
{
}

typedef simulation::Node::ctime_t ctime_t;
void EulerSolver::solve(double dt)
{
    MultiVector pos(this, VecId::position());
    MultiVector vel(this, VecId::velocity());
    MultiVector acc(this, VecId::dx());
    MultiVector f(this, VecId::force());

    //---------------------------------------------------------------
    //DEBUGGING TOOLS
    bool printLog = f_printLog.getValue();
    //---------------------------------------------------------------





    addSeparateGravity(dt);	// v += dt*g . Used if mass wants to added G separately from the other forces to v.
    computeForce(f);
    if( printLog )
    {
        cerr<<"EulerSolver, dt = "<< dt <<endl;
        cerr<<"EulerSolver, initial x = "<< pos <<endl;
        cerr<<"EulerSolver, initial v = "<< vel <<endl;
        cerr<<"EulerSolver, f = "<< f <<endl;
    }
    accFromF(acc, f);
    projectResponse(acc);
    if( printLog )
    {
        cerr<<"EulerSolver, a = "<< acc <<endl;
    }



#ifdef SOFA_HAVE_LAPACK
    if (constraintAcc.getValue())
    {
        solveConstraint(VecId::dx());
    }
#endif

    // update state
#ifdef SOFA_NO_VMULTIOP // unoptimized version
    if( symplectic.getValue() )
    {
        vel.peq(acc,dt);

#ifdef SOFA_HAVE_LAPACK
        if (constraintVel.getValue())
        {
            solveConstraint(VecId::velocity(), !constraintPos.getValue());
        }
#endif
        pos.peq(vel,dt);

#ifdef SOFA_HAVE_LAPACK
        if (constraintPos.getValue())
        {
            solveConstraint(VecId::position());
        }
#endif

    }
    else
    {
        pos.peq(vel,dt);

#ifdef SOFA_HAVE_LAPACK
        if (constraintPos.getValue())
        {
            solveConstraint(VecId::position());
        }
#endif

        vel.peq(acc,dt);

#ifdef SOFA_HAVE_LAPACK
        if (constraintVel.getValue())
        {
            solveConstraint(VecId::velocity(), !constraintPos.getValue());
        }
#endif
    }
#else // single-operation optimization
    {
        typedef core::componentmodel::behavior::BaseMechanicalState::VMultiOp VMultiOp;
        VMultiOp ops;
        ops.resize(2);
        // change order of operations depending on the sympletic flag
        int op_vel = (symplectic.getValue()?0:1);
        int op_pos = (symplectic.getValue()?1:0);
        ops[op_vel].first = (VecId)vel;
        ops[op_vel].second.push_back(std::make_pair((VecId)vel,1.0));
        ops[op_vel].second.push_back(std::make_pair((VecId)acc,dt));
        ops[op_pos].first = (VecId)pos;
        ops[op_pos].second.push_back(std::make_pair((VecId)pos,1.0));
        ops[op_pos].second.push_back(std::make_pair((VecId)vel,dt));
        simulation::MechanicalVMultiOpVisitor vmop(ops);
        vmop.execute(this->getContext());

#ifdef SOFA_HAVE_LAPACK
        if (constraintVel.getValue())
        {
            solveConstraint(VecId::velocity(), !constraintPos.getValue());
        }

        if (constraintPos.getValue())
        {
            solveConstraint(VecId::position());
        }
#endif


    }
#endif

    if( printLog )
    {
        cerr<<"EulerSolver, final x = "<< pos <<endl;
        cerr<<"EulerSolver, final v = "<< vel <<endl;
    }
}


#ifdef SOFA_HAVE_LAPACK
void EulerSolver::solveConstraint(VecId Id, bool propagateVelocityToPosition)
{
    if (this->f_printLog.getValue())
    {
        if (Id==VecId::dx())            std::cerr << "Applying the constraint on the acceleration\n";
        else if (Id==VecId::velocity()) std::cerr << "Applying the constraint on the velocity\n";
        else if (Id==VecId::position()) std::cerr << "Applying the constraint on the position\n";
    }

    //Get the matrices through mappings

    // mechanical action executed from root node to propagate the constraints
    simulation::MechanicalResetConstraintVisitor().execute(this->getContext());
    // calling writeConstraintEquations
    sofa::simulation::MechanicalAccumulateLMConstraint LMConstraintVisitor;
    LMConstraintVisitor.execute(this->getContext());


    using core::componentmodel::behavior::BaseLMConstraint ;
    BaseLMConstraint::ConstId idxState;
    if      (Id==VecId::dx())       idxState=BaseLMConstraint::ACC;
    else if (Id==VecId::velocity()) idxState=BaseLMConstraint::VEL;
    else                            idxState=BaseLMConstraint::POS;

    //************************************************************
    // Find the number of constraints                           //
    //************************************************************
    unsigned int numConstraint=0;
    for (unsigned int mat=0; mat<LMConstraintVisitor.numConstraintDatas(); ++mat)
    {
        ConstraintData& constraint=LMConstraintVisitor.getConstraint(mat);
        numConstraint += constraint.data->getNumConstraint(idxState);
    }
    if (numConstraint == 0) return; //Nothing to solve

    //Right Hand term
    FullVector<double>  c;
    c.resize(numConstraint);

    //Left Hand term: J0.M^-1.J0^T + J1.M1^-1.J1^T + ...
    FullMatrix<double>  A;
    A.resize(numConstraint,numConstraint);


    //Informations to build the matrices
    //Dofs to be constrained
    std::set< sofa::core::componentmodel::behavior::BaseMechanicalState* > setDofs;
    //Store the matrix M^-1.J^T for each dof in order to apply the modification of the state
    std::map< sofa::core::componentmodel::behavior::BaseMechanicalState*, FullMatrix<double>  > invM_Jtrans;
    //To Build J.M^-1.J^T, we need to know what line of the VecConst will be used
    std::map< sofa::core::componentmodel::behavior::BaseMechanicalState*, sofa::helper::vector< sofa::helper::vector< unsigned int > > > indicesUsedSystem;
    //To Build J.M^-1.J^T, we need to know to what system of contraint: the offset helps to write the matrix J
    std::map< sofa::core::componentmodel::behavior::BaseMechanicalState*, sofa::helper::vector< unsigned int > > offsetSystem;
    std::map< sofa::core::componentmodel::behavior::BaseMechanicalState*, sofa::helper::vector< double > > factorSystem;


    unsigned int offset;
    //************************************************************
    // Gather the information from all the constraint components
    //************************************************************
    unsigned constraintOffset=0;
    for (unsigned int mat=0; mat<LMConstraintVisitor.numConstraintDatas(); ++mat)
    {
        ConstraintData& constraint=LMConstraintVisitor.getConstraint(mat);

        sofa::helper::vector< unsigned int > indicesUsed[2];
        constraint.data->getIndicesUsed(idxState, indicesUsed[0], indicesUsed[1]);


        unsigned int currentNumConstraint=constraint.data->getNumConstraint(idxState);
        setDofs.insert(constraint.independentMState[0]);
        setDofs.insert(constraint.independentMState[1]);

        indicesUsedSystem[constraint.independentMState[0] ].push_back(indicesUsed[0]);
        indicesUsedSystem[constraint.independentMState[1] ].push_back(indicesUsed[1]);

        offsetSystem[constraint.independentMState[0] ].push_back(constraintOffset);
        offsetSystem[constraint.independentMState[1] ].push_back(constraintOffset);

        factorSystem[constraint.independentMState[0] ].push_back( 1.0);
        factorSystem[constraint.independentMState[1] ].push_back(-1.0);

        constraintOffset += currentNumConstraint;
    }


    //************************************************************
    // Build the Right Hand Term
    //************************************************************
    buildRightHandTerm(Id, LMConstraintVisitor, c);


    //************************************************************
    // Building A=J0.M0^-1.J0^T + J1.M1^-1.J1^T + ... and M^-1.J^T
    //************************************************************
    std::set< sofa::core::componentmodel::behavior::BaseMechanicalState* >::const_iterator itDofs;

    for (itDofs=setDofs.begin(); itDofs!=setDofs.end(); itDofs++)
    {
        sofa::core::componentmodel::behavior::BaseMechanicalState* dofs=*itDofs;
        core::componentmodel::behavior::BaseMass *mass=dynamic_cast< core::componentmodel::behavior::BaseMass *>(dofs->getContext()->getMass());
        FullMatrix<double>  &M=invM_Jtrans[dofs];

        if (mass)
        {
            //Apply Constraint on the inverse of the mass matrix: should maybe need a better interface
            FullVector<double>  FixedPoints(dofs->getSize());
            for (unsigned int i=0; i<FixedPoints.size(); ++i) FixedPoints.set(i,1.0);
            offset=0;
            sofa::helper::vector< core::componentmodel::behavior::BaseConstraint *> listConstraintComponent;
            dofs->getContext()->get<core::componentmodel::behavior::BaseConstraint>(&listConstraintComponent);
            for (unsigned int j=0; j<listConstraintComponent.size(); ++j) listConstraintComponent[j]->applyInvMassConstraint(&FixedPoints,offset);

            //Compute M^-1.J^T in M, and accumulate J.M^-1.J^T in A
            mass->buildSystemMatrix(M, A,
                    indicesUsedSystem[dofs],
                    factorSystem[dofs],
                    offsetSystem[dofs],
                    FixedPoints);
        }
        if (this->f_printLog.getValue())
        {
            std::cout << "Matrix M^-1.J^T " << dofs->getName() << " \n";
            printMatrix(invM_Jtrans[dofs]);
        }
    }


    if (this->f_printLog.getValue())
    {
        std::cerr << "A= J0.M0^-1.J0^T + J1.M1^-1.J1^T + ...: \n";
        printMatrix(A);
        std::cerr << "for a constraint: " << "\n";
        printVector(c);
    }

    //************************************************************
    // Solving the System using LAPACK
    //************************************************************

    if (constraintResolution.getValue())
    {
        if (this->f_printLog.getValue()) std::cerr << "Using Gauss-Seidel resolution\n";

        //-- Initialization of X, solution of the system
        FullVector<double>  X; X.resize(A.colSize());
        bool continueIteration=true;
        unsigned int iteration=0;
        for (; iteration < numIterations.getValue() && continueIteration; ++iteration)
        {
            unsigned int idxConstraint=0;
            FullVector<double>  var;
            FullVector<double>  previousIteration;
            continueIteration=false;
            //Iterate on all the Constraint components
            for (unsigned int componentConstraint=0; componentConstraint<LMConstraintVisitor.numConstraintDatas(); ++componentConstraint)
            {
                ConstraintData& constraint=LMConstraintVisitor.getConstraint(componentConstraint);
                //Get the vector containing all the constraint stored in one component
                std::vector< BaseLMConstraint::groupConstraint > constraintId;
                constraint.data->getConstraintsId(idxState, constraintId);
                for (unsigned int constraintEntry=0; constraintEntry<constraintId.size(); ++constraintEntry)
                {
                    //-------------------------------------
                    //Initialize the variables, and store X^(k-1) in previousIteration
                    unsigned int numConstraintToProcess=constraintId[constraintEntry].getNumConstraint();
                    var.resize(numConstraintToProcess); var.clear();
                    previousIteration.resize(numConstraintToProcess);
                    for (unsigned int i=0; i<numConstraintToProcess; ++i)
                    {
                        previousIteration.set(i,X.element(idxConstraint+i));
                        X.set(idxConstraint+i,0);
                    }
                    //    operation: A.X^k --> var
                    double alpha=1,beta=0;
                    applyLapackDGEMV( A.ptr()+(idxConstraint*numConstraint), false, X.ptr(), var.ptr(),
                            alpha, beta,
                            numConstraintToProcess, numConstraint);
                    double error=0;
                    for (unsigned int i=0; i<numConstraintToProcess; ++i)
                    {
                        //X^(k)= (c^(0)-A[c,c]*X^(k-1))/A[c,c]
                        X.set(idxConstraint+i,(c.element(idxConstraint+i) - var.element(i))/A.element(idxConstraint+i,idxConstraint+i));
                        error += pow(previousIteration.element(i)-X.element(idxConstraint+i),2);
                    }
                    error = sqrt(error);
                    //Update only if the error is higher than a threshold. If no "big changes" occured, we set: X[c]^(k) = X[c]^(k-1)
                    if (error < maxError.getValue())
                    {
                        for (unsigned int i=0; i<numConstraintToProcess; ++i)
                        {
                            X.set(idxConstraint+i, previousIteration.element(i));
                        }
                    }
                    else
                    {
                        continueIteration=true;
                    }
                    idxConstraint+=numConstraintToProcess;
                }
            }
        }
        //std::cerr << "Gauss-Seidel done in " << iteration << " iterations \n";
        //*********************************
        //Correct States
        //*********************************
        for (itDofs=setDofs.begin(); itDofs!=setDofs.end(); itDofs++)
        {
            sofa::core::componentmodel::behavior::BaseMechanicalState* dofs=*itDofs;
            constraintStateCorrection(Id, dofs, invM_Jtrans[dofs], X, propagateVelocityToPosition);
        }
        return;
    }
    else
    {
        if (this->f_printLog.getValue()) std::cerr << "Using direct LU resolution\n";
        //Third Operation: solve  (J0.M0^-1.J0^T + J1.M^-1.J1^T).F = C
        applyLapackDGESV( A[0],c.ptr(), numConstraint, 1 );

        if (this->f_printLog.getValue())
        {
            std::cerr << "Final Result lambda\n";
            printVector(c);


            std::cerr << "------------------------------------------------\n\n";
        }

        //************************************************************
        // Updating the state vectors
        // get the displacement. deltaState = M^-1.J^T.lambda : lambda being the solution of the system
        for (itDofs=setDofs.begin(); itDofs!=setDofs.end(); itDofs++)
        {
            sofa::core::componentmodel::behavior::BaseMechanicalState* dofs=*itDofs;
            constraintStateCorrection(Id, dofs, invM_Jtrans[dofs], c, propagateVelocityToPosition);
        }
    }
}






void EulerSolver::buildRightHandTerm(VecId &Id, sofa::simulation::MechanicalAccumulateLMConstraint &LMConstraintVisitor, FullVector<double>  &c)
{
    using core::componentmodel::behavior::BaseLMConstraint ;
    BaseLMConstraint::ConstId idxState;
    if      (Id==VecId::dx())       idxState=BaseLMConstraint::ACC;
    else if (Id==VecId::velocity()) idxState=BaseLMConstraint::VEL;
    else                            idxState=BaseLMConstraint::POS;

    unsigned constraintOffset=0;
    for (unsigned int mat=0; mat<LMConstraintVisitor.numConstraintDatas(); ++mat)
    {
        ConstraintData& constraint=LMConstraintVisitor.getConstraint(mat);

        sofa::helper::vector< unsigned int > indicesUsed[2];
        std::vector<double> expectedValues;
        std::vector<BaseLMConstraint::ValueId> expectedValuesType;
        constraint.data->getIndicesUsed(idxState, indicesUsed[0], indicesUsed[1]);
        constraint.data->getExpectedValues(idxState, expectedValues);
        constraint.data->getExpectedValuesType(idxState, expectedValuesType);

        unsigned int currentNumConstraint=expectedValues.size();

        //************************************************************
        //Building Right hand term
        FullVector<double>  V[2]; V[0].resize(c.size()); V[1].resize(c.size());
        //Different type of "expected values" exist:
        //the one called FINAL. It means, the value of the constraint (zero in velocity, ...). It needs to compute the current state of the dof, in order to build the right hand term.
        //Another is CORRECTION. It already knows the value of the right hand term. Often used in Constraint dealing with position. If we want to constrain two dofs to remain at a distance d, we put directly d in the right hand term.
        bool needComputeCurrentState=false;

        for (unsigned int i=0; i<currentNumConstraint; ++i)
        {
            if (expectedValuesType[i] == BaseLMConstraint::FINAL) { needComputeCurrentState=true; break;}
        }
        if (needComputeCurrentState)
        {
            constraint.independentMState[0]->computeConstraintProjection(indicesUsed[0], Id, V[0],constraintOffset);
            constraint.independentMState[1]->computeConstraintProjection(indicesUsed[1], Id, V[1],constraintOffset);
        }

        for (unsigned int i=constraintOffset; i<constraintOffset+currentNumConstraint; ++i)
        {
            unsigned int idxConstraint=i-constraintOffset;
            switch(expectedValuesType[idxConstraint])
            {
            case  BaseLMConstraint::FINAL:
                c.add(i,(V[1].element(i)-V[0].element(i)) - expectedValues[idxConstraint]);
                break;
            case  BaseLMConstraint::FACTOR:
                c.add(i,(V[1].element(i)-V[0].element(i))*(expectedValues[idxConstraint]));
                break;
            case  BaseLMConstraint::CORRECTION:
                c.add(i,(-expectedValues[idxConstraint]));
                break;
            }
        }

        constraintOffset += currentNumConstraint;
    }
}






void EulerSolver::constraintStateCorrection(VecId &Id, sofa::core::componentmodel::behavior::BaseMechanicalState* dofs,
        FullMatrix<double>  &invM_Jtrans, FullVector<double>  &c, bool propageVelocityChange)
{
    double alpha=1,beta=0;
    if (dofs->getContext()->getMass() == NULL) return;
    FullVector<double>  A; A.resize(invM_Jtrans.rowSize());
    //Correct Dof
    //    operation: M0^-1.J0^T.lambda -> A
    applyLapackDGEMV( invM_Jtrans, false, c, A,
            alpha, beta);


    //In case of position correction, we need to update the velocities
    if (Id==VecId::position())
    {
        //Detect Rigid Bodies
        if ((unsigned int)(dofs->getSize()*(3+3)) == A.size())
        {

            FullVector<double>  Acorrection; Acorrection.resize(dofs->getSize()*(3+4));
            //We have to transform the Euler Rotations into a quaternion
            unsigned int offset=0;
            for (unsigned int l=0; l<A.size(); l+=6)
            {
                offset=l/6;
                Acorrection.set(l+0+offset,A.element(l+0));
                Acorrection.set(l+1+offset,A.element(l+1));
                Acorrection.set(l+2+offset,A.element(l+2));

                Quaternion q=Quaternion::createQuaterFromEuler(Vector3(A.element(l+3),A.element(l+4),A.element(l+5)));

                Acorrection.set(l+3+offset,q[0]);
                Acorrection.set(l+4+offset,q[1]);
                Acorrection.set(l+5+offset,q[2]);
                Acorrection.set(l+6+offset,q[3]);
            }

            if (this->f_printLog.getValue())
            {
                std::cerr << "delta RigidState : " << dofs->getName() << "\n";  printVector(Acorrection);
            }
            offset=0;
            dofs->addBaseVectorToState(VecId::position(),&Acorrection,offset );
        }
        else
        {

            if (this->f_printLog.getValue())
            {
                std::cerr << "delta State : " << dofs->getName() << "\n";  printVector(A);
            }
            unsigned int offset=0;
            dofs->addBaseVectorToState(VecId::position(),&A,offset );
        }

        if (propageVelocityChange)
        {
            double h=1.0/getContext()->getDt();
            for (unsigned int i=0; i<A.size(); ++i) A[i]*=h;

            unsigned int offset=0;
            dofs->addBaseVectorToState(VecId::velocity(),&A,offset );
        }
    }
    else
    {

        if (this->f_printLog.getValue())
        {
            std::cerr << "delta State : " << dofs->getName() << "\n"; printVector(A);
        }

        unsigned int offset=0;
        dofs->addBaseVectorToState(Id,&A,offset );
    }
}
#endif

} // namespace odesolver

} // namespace component

} // namespace sofa

