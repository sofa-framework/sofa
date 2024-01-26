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
#include <sofa/component/constraint/lagrangian/correction/config.h>

#include <sofa/core/behavior/ConstraintCorrection.h>

#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/behavior/LinearSolver.h>

#include <sofa/type/Mat.h>
#include <sofa/type/Vec.h>

#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/linearalgebra/FullMatrix.h>

namespace sofa::component::constraint::lagrangian::correction
{

/**
 *  \brief Component computing constrained forces within a simulated body using the compliance method.
 */
template<class TDataTypes>
class LinearSolverConstraintCorrection : public sofa::core::behavior::ConstraintCorrection< TDataTypes >
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(LinearSolverConstraintCorrection, TDataTypes), SOFA_TEMPLATE(sofa::core::behavior::ConstraintCorrection, TDataTypes));

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::MatrixDeriv::RowConstIterator MatrixDerivRowConstIterator;
    typedef typename DataTypes::MatrixDeriv::ColConstIterator MatrixDerivColConstIterator;
    typedef typename DataTypes::MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef typename DataTypes::MatrixDeriv::ColIterator MatrixDerivColIterator;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;

    typedef std::list<linearalgebra::BaseMatrix::Index> ListIndex;
    typedef sofa::core::behavior::ConstraintCorrection< TDataTypes > Inherit;
protected:
    LinearSolverConstraintCorrection(sofa::core::behavior::MechanicalState<DataTypes> *mm = nullptr);

    virtual ~LinearSolverConstraintCorrection();
public:
    void init() override;


    void addComplianceInConstraintSpace(const sofa::core::ConstraintParams *cparams, linearalgebra::BaseMatrix* W) override;

    void getComplianceMatrix(linearalgebra::BaseMatrix* ) const override;

    void computeMotionCorrection(const core::ConstraintParams* cparams, core::MultiVecDerivId dx, core::MultiVecDerivId f) override;

    void applyMotionCorrection(const core::ConstraintParams * cparams, Data< VecCoord > &x, Data< VecDeriv > &v, Data< VecDeriv > &dx, const Data< VecDeriv > &f) override;

    void applyPositionCorrection(const sofa::core::ConstraintParams *cparams, Data< VecCoord >& x, Data< VecDeriv>& dx, const Data< VecDeriv >& f) override;

    void applyVelocityCorrection(const sofa::core::ConstraintParams *cparams, Data< VecDeriv>& v, Data< VecDeriv>& dv, const Data< VecDeriv >& f) override;

    void rebuildSystem(SReal massFactor, SReal forceFactor) override;

    /// @name Deprecated API
    /// @{

    void applyContactForce(const linearalgebra::BaseVector *f) override;

    void resetContactForce() override;

    /// @}

    /// @name Unbuilt constraint system during resolution
    /// @{

    Data< bool > wire_optimization; ///< constraints are reordered along a wire-like topology (from tip to base)
    SingleLink<LinearSolverConstraintCorrection, sofa::core::behavior::LinearSolver, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_linearSolver; ///< Link towards the linear solver used to compute the compliance matrix, requiring the inverse of the linear system matrix
    SingleLink<LinearSolverConstraintCorrection, sofa::core::behavior::OdeSolver, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_ODESolver; ///< Link towards the ODE solver used to recover the integration factors


    SOFA_ATTRIBUTE_DISABLED__CONSTRAINTCORRECTION_EXPLICITLINK()
    core::objectmodel::lifecycle::RemovedData  solverName{this, "v22.12", "v23.06", "solverName", "replace \"solverName\" by using an explicit data link: \"linearSolver\" (PR #3152)}"};

    void verify_constraints();

    bool hasConstraintNumber(int index) override;  // virtual ???

    void resetForUnbuiltResolution(SReal* f, std::list<unsigned int>& renumbering) override;

    void addConstraintDisplacement(SReal*d, int begin,int end) override;

    void setConstraintDForce(SReal*df, int begin, int end, bool update) override;

    void getBlockDiagonalCompliance(linearalgebra::BaseMatrix* W, int begin, int end) override;

protected:
    linearalgebra::SparseMatrix<SReal> J; ///< constraint matrix
    linearalgebra::FullVector<SReal> F; ///< forces computed from the constraints

    /**
    * @brief Compute the compliance matrix
    */
    virtual void computeJ(sofa::linearalgebra::BaseMatrix* W, const MatrixDeriv& j);


    ////////////////////////// Inherited attributes ////////////////////////////
    /// https://gcc.gnu.org/onlinedocs/gcc/Name-lookup.html
    /// Bring inherited attributes and function in the current lookup context.
    /// otherwise any access to the base::attribute would require
    /// the "this->" approach.
    using Inherit::d_componentState ;
    using Inherit::mstate ;
    ////////////////////////////////////////////////////////////////////////////

private:
    // new :  for non building the constraint system during solving process //
    VecDeriv constraint_disp, constraint_force;
    std::list<int> constraint_dofs;		// list of indices of each point which is involve with constraint // TODO : verify if useful !!
    linearalgebra::BaseMatrix* systemMatrix_buf;
    linearalgebra::BaseVector* systemRHVector_buf;
    linearalgebra::BaseVector* systemLHVector_buf;
    linearalgebra::FullVector<Real>* systemLHVector_buf_fullvector { nullptr };

    // par un vecteur de listes precaclues pour chaque contrainte
    std::vector< ListIndex > Vec_I_list_dof;   // vecteur donnant la liste des indices des dofs par block de contrainte
    int last_force, last_disp; //last_force indice du dof le plus petit portant la force/le dpt qui a ?t? modifi? pour la derni?re fois (wire optimisation only?)
    bool _new_force; // if true, a "new" force was added in setConstraintDForce which is not yet integrated by a new computation in addConstraintDisplacements
};

#if !defined(SOFA_COMPONENT_CONSTRAINT_LINEARSOLVERCONSTRAINTCORRECTION_CPP)
extern template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_CORRECTION_API LinearSolverConstraintCorrection<sofa::defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_CORRECTION_API LinearSolverConstraintCorrection<sofa::defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_CORRECTION_API LinearSolverConstraintCorrection<sofa::defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_CORRECTION_API LinearSolverConstraintCorrection<sofa::defaulttype::Rigid3Types>;

#endif

} //namespace sofa::component::constraint::lagrangian::correction
