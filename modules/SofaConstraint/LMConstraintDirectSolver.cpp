/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <SofaConstraint/LMConstraintDirectSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaConstraint/ContactDescription.h>
#include <sofa/core/ObjectFactory.h>
#include <Eigen/QR>
#include <Eigen/SVD>


namespace sofa
{

namespace component
{

namespace constraintset
{

LMConstraintDirectSolver::LMConstraintDirectSolver()
    : solverAlgorithm(initData(&solverAlgorithm, "solverAlgorithm", "Algorithm used to solve the system W.Lambda=c"))
{
    //Add here other algo
    sofa::helper::OptionsGroup algo(1,"SVD");
    solverAlgorithm.setValue(algo);
}


bool LMConstraintDirectSolver::buildSystem(const core::ConstraintParams* cParams, MultiVecId res1, MultiVecId res2)
{
    bool sucess = LMConstraintSolver::buildSystem(cParams, res1, res2);

    return sucess;
}

bool LMConstraintDirectSolver::solveSystem(const core::ConstraintParams* cParams, MultiVecId res1, MultiVecId res2)
{
    //First, do n iterations of Gauss Seidel
    bool success = LMConstraintSolver::solveSystem(cParams, res1, res2);


    if (cParams->constOrder() != core::ConstraintParams::VEL) return success;


    //Then process to a direct solution of the system

    //We need to find all the constraint related to contact
    // 1. extract the information about the state of the contact and build the new L, L^T matrices
    // 2. build the full system
    // 3. solve


    //------------------------------------------------------------------
    // extract the information about the state of the contact
    //------------------------------------------------------------------

    //************************************************************
#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::printNode("AnalyseConstraints");
#endif
    const helper::vector< sofa::core::behavior::BaseLMConstraint* > &LMConstraints=LMConstraintVisitor.getConstraints();

    JacobianRows rowsL ; rowsL.reserve(numConstraint);
    JacobianRows rowsLT; rowsLT.reserve(numConstraint);
    helper::vector< unsigned int > rightHandElements;

    analyseConstraints(LMConstraints, cParams->constOrder(),
            rowsL, rowsLT, rightHandElements);

#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::printCloseNode("AnalyseConstraints");
#endif
    if (rowsL.empty() || rowsLT.empty()) return success;





#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::printNode("BuildFullSystem");
#endif
    //------------------------------------------------------------------
    // build c: right hand term
    //------------------------------------------------------------------
    VectorEigen previousC(c);
    //TODO: change newC by c
    c=VectorEigen::Zero(rowsL.size());
    unsigned int idx=0;
    for (helper::vector<unsigned int >::const_iterator it=rightHandElements.begin(); it!=rightHandElements.end(); ++it)
        c[idx++]=previousC[*it];

    //------------------------------------------------------------------
    // build the L and LT matrices
    //------------------------------------------------------------------


    DofToMatrix LMatricesDirectSolver;
    DofToMatrix LTMatricesDirectSolver;
    for (DofToMatrix::iterator it=LMatrices.begin(); it!=LMatrices.end(); ++it)
    {
        //------------------------------------------------------------------
        const SparseMatrixEigen& matrix= it->second;
        //Init the manipulator with the full matrix
        linearsolver::LMatrixManipulator manip;
        manip.init(matrix);


        //------------------------------------------------------------------
        SparseMatrixEigen  L (rowsL.size(),  matrix.cols());
        L.reserve(rowsL.size()*matrix.cols());
        manip.buildLMatrix(rowsL ,L);
        L.finalize();
        LMatricesDirectSolver.insert (std::make_pair(it->first,L ));



        //------------------------------------------------------------------
        SparseMatrixEigen  LT(rowsLT.size(), matrix.cols());
        LT.reserve(rowsLT.size()*matrix.cols());
        manip.buildLMatrix(rowsLT,LT);
        LT.finalize();
        LTMatricesDirectSolver.insert(std::make_pair(it->first,LT));
    }



    //------------------------------------------------------------------
    // build the full system
    //------------------------------------------------------------------
    const  int rows=rowsL.size();
    const  int cols=rowsLT.size();
    SparseColMajorMatrixEigen Wsparse(rows,cols);
    buildLeftRectangularMatrix(invMassMatrix, LMatricesDirectSolver, LTMatricesDirectSolver, Wsparse,invMass_Ltrans);


    //------------------------------------------------------------------
    // conversion from sparse to dense matrix
    //------------------------------------------------------------------
    Lambda=VectorEigen::Zero(rows);

    W=MatrixEigen::Zero(rows,cols);



    SparseMatrixEigen Wresult(Wsparse);
    for (int k=0; k<Wresult.outerSize(); ++k)
        for (SparseMatrixEigen::InnerIterator it(Wresult,k); it; ++it) W(it.row(),it.col()) = it.value();



#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::printCloseNode("BuildFullSystem");
#endif

    //------------------------------------------------------------------
    // Solve the system
    //------------------------------------------------------------------
    const std::string &algo=solverAlgorithm.getValue().getSelectedItem() ;
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::TRACE_ARGUMENT arg1;
    arg1.push_back(std::make_pair("Algorithm", algo));
    arg1.push_back(std::make_pair("Dimension", printDimension(W)));
    sofa::simulation::Visitor::printNode("DirectSolveSystem", "",arg1);
#endif
    if(algo == "SVD")
    {
        Eigen::JacobiSVD< MatrixEigen > solverSVD(W);
        VectorEigen invSingularValues(solverSVD.singularValues());

        for (int i=0; i<invSingularValues.size(); ++i)
        {
            if (invSingularValues[i] < 1e-10) invSingularValues[i]=0;
            else invSingularValues[i]=1/invSingularValues[i];
        }
        Lambda.noalias() = solverSVD.matrixV()*invSingularValues.asDiagonal()*solverSVD.matrixU().transpose()*c;
    }

    msg_info() << "W" <<  printDimension(W) <<  "  Lambda" << printDimension(Lambda) << "  c" << printDimension(c) << msgendl
                << "W     ===============================================\n" << W << msgendl
                << "Lambda===============================================\n" << Lambda << msgendl
                << "c     ===============================================\n" << c ;

#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::printCloseNode("DirectSolveSystem");
#endif
    return success;

}


void LMConstraintDirectSolver::analyseConstraints(const helper::vector< sofa::core::behavior::BaseLMConstraint* > &LMConstraints, core::ConstraintParams::ConstOrder order,
        JacobianRows &rowsL,JacobianRows &rowsLT, helper::vector< unsigned int > &rightHandElements) const
{
    //Iterate among all the Sofa LMConstraint
    for (unsigned int componentConstraint=0; componentConstraint<LMConstraints.size(); ++componentConstraint)
    {
        sofa::core::behavior::BaseLMConstraint *constraint=LMConstraints[componentConstraint];
        //Find the constraint dealing with contact
        if (ContactDescriptionHandler* contactDescriptor=dynamic_cast<ContactDescriptionHandler*>(constraint))
        {
            const helper::vector< sofa::core::behavior::ConstraintGroup* > &constraintOrder=constraint->getConstraintsOrder(order);
            //Iterate among all the contacts
            for (helper::vector< sofa::core::behavior::ConstraintGroup* >::const_iterator itGroup=constraintOrder.begin(); itGroup!=constraintOrder.end(); ++itGroup)
            {
                const sofa::core::behavior::ConstraintGroup* group=*itGroup;
                const sofa::component::constraintset::ContactDescription& contact=contactDescriptor->getContactDescription(group);

                const unsigned int idxEquation=group->getConstraint(0).idx;

                switch(contact.state)
                {
                case VANISHING:
                {
                    //                    serr <<"Constraint " << idxEquation << " VANISHING" << sendl;
                    //0 equation
                    break;
                }
                case STICKING:
                {
                    //                    serr << "Constraint " <<idxEquation << " STICKING" << sendl;
                    const unsigned int i=rowsL.size();
                    rowsL.push_back(linearsolver::LLineManipulator().addCombination(idxEquation  ));
                    rowsL.push_back(linearsolver::LLineManipulator().addCombination(idxEquation+1));
                    rowsL.push_back(linearsolver::LLineManipulator().addCombination(idxEquation+2));

                    //3 equations
                    rowsLT.push_back(rowsL[i  ]);
                    rowsLT.push_back(rowsL[i+1]);
                    rowsLT.push_back(rowsL[i+2]);

                    rightHandElements.push_back(idxEquation  );
                    rightHandElements.push_back(idxEquation+1);
                    rightHandElements.push_back(idxEquation+2);
                    break;
                }
                case SLIDING:
                {
                    //                    serr << "Constraint " <<idxEquation << " SLIDING" << sendl;
                    rowsL.push_back(linearsolver::LLineManipulator().addCombination(idxEquation  ));
                    rowsL.push_back(linearsolver::LLineManipulator().addCombination(idxEquation+1));
                    rowsL.push_back(linearsolver::LLineManipulator().addCombination(idxEquation+2));


                    //1 equation with the response force along the Coulomb friction cone
                    rowsLT.push_back(linearsolver::LLineManipulator()
                            .addCombination(idxEquation  ,contact.coeff[0])
                            .addCombination(idxEquation+1,contact.coeff[1])
                            .addCombination(idxEquation+2,contact.coeff[2]));

                    rightHandElements.push_back(idxEquation  );
                    rightHandElements.push_back(idxEquation+1);
                    rightHandElements.push_back(idxEquation+2);
                    break;
                }
                }
            }
        }
        else
        {
            //Non contact constraints: we add all the equations
            const helper::vector< sofa::core::behavior::ConstraintGroup* > &constraintOrder=constraint->getConstraintsOrder(order);
            for (helper::vector< sofa::core::behavior::ConstraintGroup* >::const_iterator itGroup=constraintOrder.begin(); itGroup!=constraintOrder.end(); ++itGroup)
            {
                const sofa::core::behavior::ConstraintGroup* group=*itGroup;
                std::pair< sofa::core::behavior::ConstraintGroup::EquationConstIterator,sofa::core::behavior::ConstraintGroup::EquationConstIterator> range=group->data();
                for ( sofa::core::behavior::ConstraintGroup::EquationConstIterator it=range.first; it!=range.second; ++it)
                {
                    rowsL.push_back(linearsolver::LLineManipulator().addCombination(it->idx));
                    rowsLT.push_back(rowsL.back());
                    rightHandElements.push_back(it->idx);
                }
            }
        }
    }
}




void LMConstraintDirectSolver::buildLeftRectangularMatrix(const DofToMatrix& invMassMatrix,
        DofToMatrix& LMatrix, DofToMatrix& LTMatrix,
        SparseColMajorMatrixEigen &LeftMatrix, DofToMatrix &invMass_Ltrans) const
{
    invMass_Ltrans.clear();
    for (SetDof::const_iterator itDofs=setDofs.begin(); itDofs!=setDofs.end(); ++itDofs)
    {
        const sofa::core::behavior::BaseMechanicalState* dofs=*itDofs;
        const SparseMatrixEigen &invMass=invMassMatrix.find(dofs)->second;
        const SparseMatrixEigen &L =LMatrix[dofs];
        const SparseMatrixEigen &LT=LTMatrix[dofs];

        SparseMatrixEigen invMass_LT=invMass*LT.transpose();

        invMass_Ltrans.insert(std::make_pair(dofs, invMass_LT));
        //SparseColMajorMatrixEigen temp=L*invMass_LT;
        LeftMatrix += L*invMass_LT;
    }
}
int LMConstraintDirectSolverClass = core::RegisterObject("A Direct Constraint Solver working specifically with LMConstraint based components")
        .add< LMConstraintDirectSolver >();

SOFA_DECL_CLASS(LMConstraintDirectSolver);


} // namespace constraintset

} // namespace component

} // namespace sofa
