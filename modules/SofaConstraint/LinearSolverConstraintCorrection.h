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
#ifndef SOFA_CORE_COLLISION_LINEARSOLVERCONTACTCORRECTION_H
#define SOFA_CORE_COLLISION_LINEARSOLVERCONTACTCORRECTION_H
#include "config.h"

#include <sofa/core/behavior/ConstraintCorrection.h>

#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/behavior/LinearSolver.h>

#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>

#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <SofaBaseLinearSolver/FullMatrix.h>

namespace sofa
{

namespace component
{

namespace constraintset
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

    typedef std::list<int> ListIndex;
    typedef sofa::core::behavior::ConstraintCorrection< TDataTypes > Inherit;
protected:
    LinearSolverConstraintCorrection(sofa::core::behavior::MechanicalState<DataTypes> *mm = NULL);

    virtual ~LinearSolverConstraintCorrection();
public:
    virtual void init() override;

    virtual void addComplianceInConstraintSpace(const sofa::core::ConstraintParams *cparams, defaulttype::BaseMatrix* W) override;

    virtual void getComplianceMatrix(defaulttype::BaseMatrix* ) const override;

    virtual void computeAndApplyMotionCorrection(const sofa::core::ConstraintParams *cparams, sofa::core::MultiVecCoordId x, sofa::core::MultiVecDerivId v, sofa::core::MultiVecDerivId f, const sofa::defaulttype::BaseVector * lambda) override;

    virtual void computeAndApplyPositionCorrection(const sofa::core::ConstraintParams *cparams, sofa::core::MultiVecCoordId x, sofa::core::MultiVecDerivId f, const sofa::defaulttype::BaseVector *lambda) override;

    virtual void computeAndApplyVelocityCorrection(const sofa::core::ConstraintParams *cparams, sofa::core::MultiVecDerivId v, sofa::core::MultiVecDerivId f, const sofa::defaulttype::BaseVector *lambda) override;

    virtual void computeAndApplyMotionCorrection(const sofa::core::ConstraintParams * /*cparams*/, sofa::core::objectmodel::Data< VecCoord > &/*x*/, sofa::core::objectmodel::Data< VecDeriv > &/*v*/, sofa::core::objectmodel::Data< VecDeriv > &/*f*/, const sofa::defaulttype::BaseVector * /*lambda*/) override {};

    virtual void computeAndApplyPositionCorrection(const sofa::core::ConstraintParams * /*cparams*/, sofa::core::objectmodel::Data< VecCoord > &/*x*/, sofa::core::objectmodel::Data< VecDeriv > &/*f*/, const sofa::defaulttype::BaseVector * /*lambda*/) override {};

    virtual void computeAndApplyVelocityCorrection(const sofa::core::ConstraintParams * /*cparams*/, sofa::core::objectmodel::Data< VecDeriv > &/*v*/, sofa::core::objectmodel::Data< VecDeriv > &/*f*/, const sofa::defaulttype::BaseVector * /*lambda*/) override {};

    virtual void applyPredictiveConstraintForce(const sofa::core::ConstraintParams *cparams, sofa::core::objectmodel::Data< VecDeriv > &f, const defaulttype::BaseVector *lambda) override;

    virtual void rebuildSystem(double massFactor, double forceFactor) override;

    /// @name Deprecated API
    /// @{

    virtual void applyContactForce(const defaulttype::BaseVector *f) override;

    virtual void resetContactForce() override;

    /// @}

    /// @name Unbuilt constraint system during resolution
    /// @{

    Data< bool > wire_optimization; ///< constraints are reordered along a wire-like topology (from tip to base)
    Data< helper::vector< std::string > >  solverName; ///< name of the constraint solver

    void verify_constraints();

    virtual bool hasConstraintNumber(int index) override;  // virtual ???

    virtual void resetForUnbuiltResolution(double * f, std::list<unsigned int>& renumbering) override;

    virtual void addConstraintDisplacement(double *d, int begin,int end) override;

    virtual void setConstraintDForce(double *df, int begin, int end, bool update) override;

    virtual void getBlockDiagonalCompliance(defaulttype::BaseMatrix* W, int begin, int end) override;

    /// Pre-construction check method called by ObjectFactory.
#if 0
    template<class T>
    static bool canCreate(T*& obj, sofa::core::objectmodel::BaseContext* context, sofa::core::objectmodel::BaseObjectDescription* arg)
    {
        /*if (getOdeSolver(context) == NULL)
            return false;
        */
        return Inherit::canCreate(obj, context, arg);
    }
#endif //

protected:

    sofa::core::behavior::OdeSolver* odesolver;
    std::vector<sofa::core::behavior::LinearSolver*> linearsolvers;

    linearsolver::SparseMatrix<SReal> J; ///< constraint matrix
    linearsolver::FullVector<SReal> F; ///< forces computed from the constraints

    /**
    * @brief Compute the compliance matrix
    */
    virtual void computeJ(sofa::defaulttype::BaseMatrix* W);

    /**
     * @brief Compute dx correction from motion space force vector.
     */
    virtual void computeDx(sofa::core::MultiVecDerivId f);

    ////////////////////////// Inherited attributes ////////////////////////////
    /// https://gcc.gnu.org/onlinedocs/gcc/Name-lookup.html
    /// Bring inherited attributes and function in the current lookup context.
    /// otherwise any access to the base::attribute would require
    /// the "this->" approach.
    using Inherit::m_componentstate ;
    using Inherit::mstate ;
    ////////////////////////////////////////////////////////////////////////////

private:
    // new :  for non building the constraint system during solving process //
    VecDeriv constraint_disp, constraint_force;
    std::list<int> constraint_dofs;		// list of indices of each point which is involve with constraint // TODO : verify if useful !!
    defaulttype::BaseMatrix* systemMatrix_buf;
    defaulttype::BaseVector* systemRHVector_buf;
    defaulttype::BaseVector* systemLHVector_buf;


    // par un vecteur de listes precaclues pour chaque contrainte
    std::vector< ListIndex > Vec_I_list_dof;   // vecteur donnant la liste des indices des dofs par block de contrainte
    int last_force, last_disp; //last_force indice du dof le plus petit portant la force/le dpt qui a ?t? modifi? pour la derni?re fois (wire optimisation only?)
    bool _new_force; // if true, a "new" force was added in setConstraintDForce which is not yet integrated by a new computation in addConstraintDisplacements

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_CONSTRAINT_LINEARSOLVERCONSTRAINTCORRECTION_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_CONSTRAINT_API LinearSolverConstraintCorrection<sofa::defaulttype::Vec3dTypes>;
extern template class SOFA_CONSTRAINT_API LinearSolverConstraintCorrection<sofa::defaulttype::Vec2dTypes>;
extern template class SOFA_CONSTRAINT_API LinearSolverConstraintCorrection<sofa::defaulttype::Vec1dTypes>;
extern template class SOFA_CONSTRAINT_API LinearSolverConstraintCorrection<sofa::defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_CONSTRAINT_API LinearSolverConstraintCorrection<sofa::defaulttype::Vec3fTypes>;
extern template class SOFA_CONSTRAINT_API LinearSolverConstraintCorrection<sofa::defaulttype::Vec2fTypes>;
extern template class SOFA_CONSTRAINT_API LinearSolverConstraintCorrection<sofa::defaulttype::Vec1fTypes>;
extern template class SOFA_CONSTRAINT_API LinearSolverConstraintCorrection<sofa::defaulttype::Rigid3fTypes>;
#endif
#endif

} // namespace collision

} // namespace component

} // namespace sofa

#endif
