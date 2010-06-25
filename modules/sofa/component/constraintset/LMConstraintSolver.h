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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_LMCONSTRAINTSOLVER_H
#define SOFA_COMPONENT_CONSTRAINTSET_LMCONSTRAINTSOLVER_H

#include <sofa/core/behavior/ConstraintSolver.h>
#include <sofa/core/behavior/BaseLMConstraint.h>
#include <sofa/core/behavior/BaseConstraintCorrection.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/component/component.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
USING_PART_OF_NAMESPACE_EIGEN

namespace sofa
{

namespace component
{

namespace constraintset
{


using core::behavior::BaseLMConstraint;
class SOFA_COMPONENT_CONSTRAINTSET_API LMConstraintSolver : public sofa::core::behavior::ConstraintSolver
{
protected:
    typedef sofa::core::VecId VecId;
    typedef sofa::core::behavior::BaseLMConstraint::ConstOrder ConstOrder;
    typedef Matrix<SReal, Eigen::Dynamic, Eigen::Dynamic> MatrixEigen;
    typedef Matrix<SReal, Eigen::Dynamic, 1>              VectorEigen;
    typedef Eigen::SparseMatrix<SReal,Eigen::RowMajor>    SparseMatrixEigen;
    typedef Eigen::SparseVector<SReal,Eigen::RowMajor>    SparseVectorEigen;

    typedef helper::set< sofa::core::behavior::BaseMechanicalState* > SetDof;
    typedef std::map< const sofa::core::behavior::BaseMechanicalState *, SparseMatrixEigen > DofToMatrix;
    typedef std::map< const sofa::core::behavior::BaseMechanicalState *, helper::set<unsigned int> > DofToMask;
    typedef std::map< const sofa::core::behavior::BaseMechanicalState *, core::behavior::BaseConstraintCorrection* > DofToConstraintCorrection;







    struct LMatrixManipulator
    {
        void init(const SparseMatrixEigen& L)
        {
            const unsigned int numConstraint=L.rows();
            const unsigned int numDofs=L.cols();
            LMatrix.resize(numConstraint,SparseVectorEigen(numDofs));
            for (unsigned int i=0; i<LMatrix.size(); ++i) LMatrix[i].startFill(0.3*numDofs);
            for (int k=0; k<L.outerSize(); ++k)
            {
                for (SparseMatrixEigen::InnerIterator it(L,k); it; ++it)
                {
                    const unsigned int row=it.row();
                    const unsigned int col=it.col();
                    const SReal value=it.value();
                    LMatrix[row].fill(col)=value;
                }
            }

            for (unsigned int i=0; i<LMatrix.size(); ++i) LMatrix[i].endFill();
        }

        void buildLMatrix(const helper::vector<unsigned int> &lines, SparseMatrixEigen& matrix) const
        {
            matrix.startFill(LMatrix.size()*LMatrix.size());

            for (unsigned int l=0; l<lines.size(); ++l)
            {
                const SparseVectorEigen& line=LMatrix[lines[l]];

                for (SparseVectorEigen::InnerIterator it(line); it; ++it)
                {
                    matrix.fill(l,it.index())=it.value();
                }
            }
            matrix.endFill();
        }


        helper::vector< SparseVectorEigen > LMatrix;
    };





public:
    SOFA_CLASS(LMConstraintSolver, sofa::core::behavior::ConstraintSolver);
    LMConstraintSolver();

    virtual void init();
    virtual void reinit() {graphKineticEnergy.setDisplayed(traceKineticEnergy.getValue());};


    virtual bool prepareStates(double dt, VecId, core::behavior::BaseConstraintSet::ConstOrder);
    virtual bool buildSystem(double dt, VecId, core::behavior::BaseConstraintSet::ConstOrder);
    virtual bool solveSystem(double dt, VecId, core::behavior::BaseConstraintSet::ConstOrder);
    virtual bool applyCorrection(double dt, VecId, core::behavior::BaseConstraintSet::ConstOrder);

    virtual void handleEvent( core::objectmodel::Event *e);



    Data<bool> constraintAcc;
    Data<bool> constraintVel;
    Data<bool> constraintPos;
    Data<unsigned int> numIterations;
    Data<double> maxError;
    mutable Data<std::map < std::string, sofa::helper::vector<double> > > graphGSError;
    Data< bool > traceKineticEnergy;
    mutable Data<std::map < std::string, sofa::helper::vector<double> > > graphKineticEnergy;


protected:
    /// Explore the graph, looking for LMConstraints: each LMConstraint can tell if they need State Propagation in order to compute the right hand term of the system
    virtual bool needPriorStatePropagation();

    /// Construct the Right hand term of the system
    virtual void buildRightHandTerm      ( ConstOrder Order, const helper::vector< core::behavior::BaseLMConstraint* > &LMConstraints,
            VectorEigen &c) const;
    /// Construct the Inverse of the mass matrix for a set of Dofs
    virtual void buildInverseMassMatrices( const SetDof &setDofs,
            DofToMatrix& invMassMatrices);
    /// Construct the L matrices: write the constraint equations, and use dofUsed to remember the particles used in order to speed up the constraint correction
    virtual void buildLMatrices          ( ConstOrder Order, const helper::vector< core::behavior::BaseLMConstraint* > &LMConstraints,
            DofToMatrix &LMatrices, DofToMask &dofUsed) const;
    /// Construct the Left Matrix A=sum_i(L_i.M^{-1}_i.L_i^T), store the matrices M^{-1}_i.L_i^T in order to compute later the constraint correction to apply
    virtual void buildLeftMatrix         ( const DofToMatrix& invMassMatrices,
            DofToMatrix& LMatrices, SparseMatrixEigen &LeftMatrix, DofToMatrix &invMass_Ltrans) const;
    /// Solve the System using a projective Gauss-Seidel algorithm: compute the Lagrange Multipliers Lambda
    virtual bool solveConstraintSystemUsingGaussSeidel(VecId id, ConstOrder Order,
            const helper::vector< core::behavior::BaseLMConstraint* > &LMConstraints,
            const MatrixEigen &W,
            const VectorEigen &c,
            VectorEigen &Lambda);

    /// Compute Kinetic Energy
    virtual void computeKineticEnergy(VecId id);

    /** Apply the correction to the state corresponding
     * @param id nature of the constraint, and correction to apply
     * @param dof MechanicalState to correct
     * @param invM_Ltrans matrix M^-1.L^T to apply the correction from the independant dofs through the mapping
     * @param c correction vector
     * @param propageVelocityChange need to propagate the correction done to the velocity for the position
     **/
    virtual void constraintStateCorrection(VecId id, core::behavior::BaseConstraintSet::ConstOrder order,
            bool isPositionChangesUpdateVelocity,
            const SparseMatrixEigen  &invM_Ltrans,
            const VectorEigen  &Lambda,
            const sofa::helper::set< unsigned int > &dofUsed,
            sofa::core::behavior::BaseMechanicalState* dof) const;


    ///
    virtual void buildLMatrix          ( const sofa::core::behavior::BaseMechanicalState *dof,
            const std::list<unsigned int> &idxEquations,unsigned int constraintOffset,
            SparseMatrixEigen& L, sofa::helper::set< unsigned int > &dofUsed ) const;
    virtual void buildInverseMassMatrix( const sofa::core::behavior::BaseMechanicalState* mstate,
            const core::behavior::BaseConstraintCorrection* constraintCorrection,
            SparseMatrixEigen& matrix) const;
    virtual void buildInverseMassMatrix( const sofa::core::behavior::BaseMechanicalState* mstate,
            const core::behavior::BaseMass* mass,
            SparseMatrixEigen& matrix) const;

    core::behavior::BaseConstraintSet::ConstOrder orderState;
    unsigned int numConstraint;

    //Variables used to do the computation
    sofa::simulation::MechanicalWriteLMConstraint LMConstraintVisitor;
    SetDof setDofs;
    DofToMask dofUsed;
    DofToMatrix invMass_Ltrans;

    MatrixEigen W;
    VectorEigen c;
    VectorEigen Lambda;

    //Persitent datas
    DofToMatrix invMassMatrix;
    DofToConstraintCorrection constraintCorrections;

};

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif
