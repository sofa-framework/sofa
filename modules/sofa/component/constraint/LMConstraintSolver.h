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
#ifndef SOFA_COMPONENT_CONSTRAINT_LMCONSTRAINTSOLVER_H
#define SOFA_COMPONENT_CONSTRAINT_LMCONSTRAINTSOLVER_H

#include <sofa/core/componentmodel/behavior/ConstraintSolver.h>
#include <sofa/core/componentmodel/behavior/BaseLMConstraint.h>
#include <sofa/core/componentmodel/behavior/BaseConstraintCorrection.h>
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

namespace constraint
{

using core::componentmodel::behavior::BaseLMConstraint;
class SOFA_COMPONENT_CONSTRAINT_API LMConstraintSolver : public sofa::core::componentmodel::behavior::ConstraintSolver
{
    typedef sofa::core::VecId VecId;
    typedef sofa::core::componentmodel::behavior::BaseLMConstraint::ConstOrder ConstOrder;
    typedef Matrix<SReal, Eigen::Dynamic, Eigen::Dynamic> MatrixEigen;
    typedef Matrix<SReal, Eigen::Dynamic, 1>              VectorEigen;
    typedef Eigen::SparseMatrix<SReal,Eigen::RowMajor>    SparseMatrixEigen;

    typedef helper::set< sofa::core::componentmodel::behavior::BaseMechanicalState* > SetDof;
    typedef std::map< const sofa::core::componentmodel::behavior::BaseMechanicalState *, SparseMatrixEigen > DofToMatrix;
    typedef std::map< const sofa::core::componentmodel::behavior::BaseMechanicalState *, helper::set<unsigned int> > DofToMask;
    typedef std::map< const sofa::core::componentmodel::behavior::BaseMechanicalState *, core::componentmodel::behavior::BaseConstraintCorrection* > DofToConstraintCorrection;
public:
    SOFA_CLASS(LMConstraintSolver, sofa::core::componentmodel::behavior::ConstraintSolver);
    LMConstraintSolver();
    ~LMConstraintSolver();

    void init();
    void reinit() {graphKineticEnergy.setDisplayed(traceKineticEnergy.getValue());};


    bool prepareStates(double dt, VecId);
    bool buildSystem(double dt, VecId);
    bool solveSystem(double dt, VecId);
    bool applyCorrection(double dt, VecId);

    void handleEvent( core::objectmodel::Event *e);



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
    bool needPriorStatePropagation();

    /// Construct the Right hand term of the system
    void buildRightHandTerm      ( ConstOrder Order, const helper::vector< core::componentmodel::behavior::BaseLMConstraint* > &LMConstraints,
            VectorEigen &c) const;
    /// Construct the Inverse of the mass matrix for a set of Dofs
    void buildInverseMassMatrices( const SetDof &setDofs,
            DofToMatrix& invMassMatrices);
    /// Construct the L matrices: write the constraint equations, and use dofUsed to remember the particles used in order to speed up the constraint correction
    void buildLMatrices          ( ConstOrder Order, const helper::vector< core::componentmodel::behavior::BaseLMConstraint* > &LMConstraints,
            DofToMatrix &LMatrices, DofToMask &dofUsed) const;
    /// Construct the Left Matrix A=sum_i(L_i.M^{-1}_i.L_i^T), store the matrices M^{-1}_i.L_i^T in order to compute later the constraint correction to apply
    void buildLeftMatrix         ( const DofToMatrix& invMassMatrices,
            DofToMatrix& LMatrices, SparseMatrixEigen &LeftMatrix, DofToMatrix &invMass_Ltrans) const;
    /// Solve the System using a projective Gauss-Seidel algorithm: compute the Lagrange Multipliers Lambda
    bool solveConstraintSystemUsingGaussSeidel(ConstOrder Order,
            const helper::vector< core::componentmodel::behavior::BaseLMConstraint* > &LMConstraints,
            const MatrixEigen &A,
            VectorEigen  c,
            VectorEigen &Lambda);

    /// Compute Kinetic Energy
    void computeKineticEnergy();

    /** Apply the correction to the state corresponding
     * @param id nature of the constraint, and correction to apply
     * @param dof MechanicalState to correct
     * @param invM_Jtrans matrix M^-1.J^T to apply the correction from the independant dofs through the mapping
     * @param c correction vector
     * @param propageVelocityChange need to propagate the correction done to the velocity for the position
     **/
    void constraintStateCorrection(VecId id, bool isPositionChangesUpdateVelocity,
            const SparseMatrixEigen  &invM_Ltrans,
            const VectorEigen  &Lambda,
            const sofa::helper::set< unsigned int > &dofUsed,
            sofa::core::componentmodel::behavior::BaseMechanicalState* dof) const;


    ///
    void buildLMatrix          ( const sofa::core::componentmodel::behavior::BaseMechanicalState *dof,
            const std::list<unsigned int> &idxEquations,unsigned int constraintOffset,
            SparseMatrixEigen& L, sofa::helper::set< unsigned int > &dofUsed ) const;
    void buildInverseMassMatrix( const sofa::core::componentmodel::behavior::BaseMechanicalState* mstate,
            const core::componentmodel::behavior::BaseConstraintCorrection* constraintCorrection,
            SparseMatrixEigen& matrix) const;
    void buildInverseMassMatrix( const sofa::core::componentmodel::behavior::BaseMechanicalState* mstate,
            const core::componentmodel::behavior::BaseMass* mass,
            SparseMatrixEigen& matrix) const;

    core::componentmodel::behavior::BaseLMConstraint::ConstOrder orderState;
    unsigned int numConstraint;

    //Variables used to do the computation
    sofa::simulation::MechanicalWriteLMConstraint LMConstraintVisitor;
    SetDof setDofs;
    DofToMask dofUsed;
    DofToMatrix invMass_Ltrans;

    MatrixEigen *A;
    VectorEigen *c;
    VectorEigen *Lambda;


    //Persitent datas
    DofToMatrix invMassMatrix;
    DofToConstraintCorrection constraintCorrections;

};

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
