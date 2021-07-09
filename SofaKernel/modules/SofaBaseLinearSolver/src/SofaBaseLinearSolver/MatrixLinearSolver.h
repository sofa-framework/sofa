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
#include <SofaBaseLinearSolver/config.h>

#include <sofa/simulation/fwd.h>
#include <sofa/core/behavior/fwd.h>    // Because this is where is RotationMatrix
#include <SofaBaseLinearSolver/fwd.h>

#include <sofa/core/behavior/LinearSolver.h>
#include <SofaBaseLinearSolver/FullVector.h>

namespace sofa::component::linearsolver
{

class MatrixInvertData {};

template<class Matrix, class Vector>
class BaseMatrixLinearSolver : public sofa::core::behavior::LinearSolver
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(BaseMatrixLinearSolver,Matrix,Vector), sofa::core::behavior::LinearSolver);

    virtual void invert(Matrix& M) = 0;

    virtual void solve(Matrix& M, Vector& solution, Vector& rh) = 0;

    virtual Matrix * getSystemMatrix() = 0;

};

/// Empty class used for default solver implementation without multi-threading support
class NoThreadManager
{
public:
    static std::string Name() { return ""; }

    static bool isAsyncSolver()
    {
        return false;
    }
};

template<class TVector>
class MatrixLinearSolverInternalData;

template<class Matrix, class Vector, class ThreadManager = NoThreadManager>
class MatrixLinearSolver;

template<class Matrix, class Vector>
class MatrixLinearSolver<Matrix,Vector,NoThreadManager> : public BaseMatrixLinearSolver<Matrix, Vector>
{
public:
    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE3(MatrixLinearSolver,Matrix,Vector,NoThreadManager), SOFA_TEMPLATE2(BaseMatrixLinearSolver,Matrix,Vector));

    typedef BaseMatrixLinearSolver<Matrix, Vector> Inherit;
    typedef NoThreadManager ThreadManager;
    typedef std::list<Index> ListIndex;
    typedef typename Vector::Real Real;
    typedef SparseMatrix<Real> JMatrixType;
    typedef defaulttype::BaseMatrix ResMatrixType;

    MatrixLinearSolver();
    ~MatrixLinearSolver() override ;

    /// Reset the current linear system.
    void resetSystem() override;

    /// Reset the current linear system.
    void resizeSystem(Size n);

    /// Set the linear system matrix, combining the mechanical M,B,K matrices using the given coefficients
    ///
    /// Note that this automatically resizes the linear system to the number of active degrees of freedoms
    ///
    /// @todo Should we put this method in a specialized class for mechanical systems, or express it using more general terms (i.e. coefficients of the second order ODE to solve)
    void setSystemMBKMatrix(const core::MechanicalParams* mparams) override;

    /// Rebuild the system using a mass and force factor
    void rebuildSystem(double massFactor, double forceFactor) override;

    /// Set the linear system matrix (only use for bench)
    void setSystemMatrix(Matrix* matrix);

    /// Set the linear system right-hand term vector, from the values contained in the (Mechanical/Physical)State objects
    void setSystemRHVector(core::MultiVecDerivId v) override;

    /// Set the initial estimate of the linear system left-hand term vector, from the values contained in the (Mechanical/Physical)State objects
    /// This vector will be replaced by the solution of the system once solveSystem is called
    void setSystemLHVector(core::MultiVecDerivId v) override;

    /// Get the linear system matrix, or nullptr if this solver does not build it
    Matrix* getSystemMatrix() override;

    /// Get the linear system right-hand term vector, or nullptr if this solver does not build it
    Vector* getSystemRHVector();

    /// Get the linear system left-hand term vector, or nullptr if this solver does not build it
    Vector* getSystemLHVector();

    /// Get the linear system matrix, or nullptr if this solver does not build it
    defaulttype::BaseMatrix* getSystemBaseMatrix() override;

    /// Get the MultiMatrix view of the linear system, or nullptr if this solved does not build it
    const core::behavior::MultiMatrixAccessor* getSystemMultiMatrixAccessor() const override;

    /// Get the linear system right-hand term vector, or nullptr if this solver does not build it
    defaulttype::BaseVector* getSystemRHBaseVector() override;

    /// Get the linear system left-hand term vector, or nullptr if this solver does not build it
    defaulttype::BaseVector* getSystemLHBaseVector() override;

    /// Solve the system as constructed using the previous methods
    void solveSystem() override;

    /// Invert the system, this method is optional because it's call when solveSystem() is called for the first time
    void invertSystem() override;

    void prepareVisitor(simulation::Visitor* v);
    void prepareVisitor(simulation::BaseMechanicalVisitor* v);

    template<class T>
    void executeVisitor(T v)
    {
        prepareVisitor(&v);
        v.execute( this->getContext() );
    }

    template<class T>
    void executeVisitor(T* v)
    {
        prepareVisitor(v);
        v->execute( this->getContext() );
    }

    /// Implementing the GetCustomTemplateName is mandatory to have a custom template name paremters
    /// instead of the default one generated automatically by the SOFA_CLASS() macro.
    static std::string GetCustomTemplateName();

    bool isAsyncSolver() override
    {
        return ThreadManager::isAsyncSolver();
    }

    void invert(Matrix& /*M*/) override {}

    void solve(Matrix& M, Vector& solution, Vector& rh) override = 0;

    virtual bool addJMInvJtLocal(Matrix * /*M*/,ResMatrixType * result,const JMatrixType * J, double fact);

    virtual bool addMInvJtLocal(Matrix * /*M*/,ResMatrixType * result,const  JMatrixType * J, double fact);

    bool addJMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact) override;

    bool addMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact) override;

    bool buildComplianceMatrix(const core::ConstraintParams* cparams, defaulttype::BaseMatrix* result, double fact) override;

    void applyConstraintForce(const sofa::core::ConstraintParams* cparams, sofa::core::MultiVecDerivId dx, const defaulttype::BaseVector* f) override;

    void computeResidual(const core::ExecParams* params, defaulttype::BaseVector* f) override;

public:

    MatrixInvertData * getMatrixInvertData(defaulttype::BaseMatrix * m);

protected:

    using BaseMatrixLinearSolver<Matrix, Vector>::partial_solve;

    /// newPartially solve the system
    virtual void partial_solve(Matrix& /*M*/, Vector& /*partial_solution*/, Vector& /*sparse_rh*/, ListIndex& /* indices_solution*/, ListIndex& /* indices input */)
    {
        msg_info()<<" WARNING : partial_solve is not implemented for this solver";
    }

    class TempVectorContainer
    {
    public:
        MatrixLinearSolver<Matrix,Vector>* parent;
        TempVectorContainer(MatrixLinearSolver<Matrix,Vector>* p, const core::ExecParams* /*params*/, Matrix& /*M*/, Vector& /*x*/, Vector& /*b*/)
            : parent(p) {}
        Vector* createTempVector() { return parent->createPersistentVector(); }
        void deleteTempVector(Vector* v) { parent->deletePersistentVector(v); }
    };

    Vector* createPersistentVector();
    static void deletePersistentVector(Vector* v);

    Matrix* createMatrix();
    static void deleteMatrix(Matrix* v);

    std::unique_ptr<MatrixLinearSolverInternalData<Vector>> internalData;
    std::unique_ptr<MatrixInvertData> invertData;

    virtual MatrixInvertData * createInvertData();

    struct LinearSystemData;
    std::unique_ptr<LinearSystemData> linearSystem;

    double currentMFactor, currentBFactor, currentKFactor;
};

#if !defined(SOFA_COMPONENT_LINEARSOLVER_MATRIXLINEARSOLVER_CPP)
extern template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< FullMatrix<double>, FullVector<double>, NoThreadManager >;
extern template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< FullMatrix<float>, FullVector<float>, NoThreadManager >;
extern template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< SparseMatrix<double>, FullVector<double>, NoThreadManager >;
extern template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< SparseMatrix<float>, FullVector<float>, NoThreadManager >;
extern template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<double>, FullVector<double>, NoThreadManager >;
extern template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<float>, FullVector<float>, NoThreadManager >;
extern template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<2,2,double> >, FullVector<double>, NoThreadManager >;
extern template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<2,2,float> >, FullVector<float>, NoThreadManager >;
extern template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<3,3,double> >, FullVector<double>, NoThreadManager >;
extern template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<3,3,float> >, FullVector<float>, NoThreadManager >;
extern template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<4,4,double> >, FullVector<double>, NoThreadManager >;
extern template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<4,4,float> >, FullVector<float>, NoThreadManager >;
extern template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<6,6,double> >, FullVector<double>, NoThreadManager >;
extern template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<6,6,float> >, FullVector<float>, NoThreadManager >;
extern template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<8,8,double> >, FullVector<double>, NoThreadManager >;
extern template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<type::Mat<8,8,float> >, FullVector<float>, NoThreadManager >;
extern template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< DiagonalMatrix<double>, FullVector<double>, NoThreadManager >;
extern template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< DiagonalMatrix<float>, FullVector<float>, NoThreadManager >;
extern template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< BlockDiagonalMatrix<3,double>, FullVector<double>, NoThreadManager >;
extern template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< RotationMatrix<double>, FullVector<double>, NoThreadManager >;
extern template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< RotationMatrix<float>, FullVector<float>, NoThreadManager >;
#endif


} // namespace sofa::component::linearsolver
