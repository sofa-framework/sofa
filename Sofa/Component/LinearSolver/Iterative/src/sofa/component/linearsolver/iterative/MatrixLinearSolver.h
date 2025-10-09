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
#include <sofa/component/linearsolver/iterative/config.h>

#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/linearsolver/iterative/GraphScatteredTypes.h>
#include <sofa/linearalgebra/FullVector.h>
#include <sofa/linearalgebra/FullMatrix.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/linearalgebra/DiagonalMatrix.h>
#include <sofa/linearalgebra/BlockDiagonalMatrix.h>
#include <sofa/linearalgebra/RotationMatrix.h>
#include <sofa/component/linearsystem/TypedMatrixLinearSystem.h>
#include <sofa/component/linearsolver/iterative/MatrixLinearSystem[GraphScattered].h>
#include <sofa/type/trait/Rebind.h>

#if SOFA_CORE_ENABLE_CRSMULTIMATRIXACCESSOR
#include <sofa/core/behavior/CRSMultiMatrixAccessor.h>
#else
#include <sofa/core/behavior/DefaultMultiMatrixAccessor.h>
#endif // SOFA_CORE_ENABLE_CRSMULTIMATRIXACCESSOR

namespace sofa::component::linearsolver
{

class MatrixInvertData
{
public:
    virtual ~MatrixInvertData() = default;
};

template<class Matrix, class Vector>
class BaseMatrixLinearSolver : public sofa::core::behavior::LinearSolver
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(BaseMatrixLinearSolver,Matrix,Vector), sofa::core::behavior::LinearSolver);

    virtual void invert(Matrix& M) = 0;

    virtual void solve(Matrix& M, Vector& solution, Vector& rh) = 0;

    SOFA_ITERATIVE_SOLVER_ATTRIBUTE_REMOVE_ASSEMBLY_API()
    virtual Matrix * getSystemMatrix() final = delete;

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
class MatrixLinearSolverInternalData
{
public:
    typedef typename TVector::Real Real;
    typedef sofa::linearalgebra::SparseMatrix<Real> JMatrixType;
    typedef linearalgebra::BaseMatrix ResMatrixType;

    void projectForceInConstraintSpace(linearalgebra::BaseVector* r,const linearalgebra::BaseVector* f)
    {
        J_local.addMulTranspose(r, f);
    }

    JMatrixType * getLocalJ() {
        return &J_local;
    }

    /**
     * Returns a JMatrixType as a pointer to the input matrix (if the input
     * matrix type is a JMatrixType), or a copy of the input matrix as a pointer
     * to the class member @J_local.
     */
    JMatrixType * getLocalJ(linearalgebra::BaseMatrix * J)
    {
        if (JMatrixType * j = dynamic_cast<JMatrixType *>(J))
        {
            return j;
        }

        using OtherReal = std::conditional_t<std::is_same_v<Real, double>, float, double>;

        //in case the matrix J is not the same type as JMatrixType, it is
        //copied in the local variable. There are 2 cases:

        //Case 1: J can be rebound: the copy is optimized
        if (auto * j_d = dynamic_cast<sofa::type::rebind_to<JMatrixType, OtherReal> *>(J))
        {
            return convertMatrix(*j_d);
        }

        //Case 2: generic case: slow copy
        J_local.clear();
        J_local.resize(J->rowSize(),J->colSize());

        using Index = typename JMatrixType::Index;
        for (Index j = 0; j < J->rowSize(); ++j)
        {
            for (Index i = 0; i < J->colSize(); ++i)
            {
                J_local.set(j, i, J->element(j, i));
            }
        }

        return &J_local;
    }

    ResMatrixType * getLocalRes(linearalgebra::BaseMatrix * R)
    {
        return R;
    }


    void addLocalRes(linearalgebra::BaseMatrix * /*R*/)
    {}

protected:

    template<typename MReal>
    JMatrixType * convertMatrix(const linearalgebra::SparseMatrix<MReal>& inputMatrix)
    {
        J_local.clear();
        J_local.resize(inputMatrix.rowSize(), inputMatrix.colSize());

        for (auto jit1 = inputMatrix.begin(); jit1 != inputMatrix.end(); ++jit1)
        {
            const auto l = jit1->first;
            for (auto i1 = jit1->second.begin(); i1 != jit1->second.end(); ++i1)
            {
                const auto c = i1->first;
                const MReal val = i1->second;
                J_local.set(l, c, val);
            }
        }
        return &J_local;
    }

private :
    JMatrixType J_local;
};

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
    typedef typename MatrixLinearSolverInternalData<Vector>::JMatrixType JMatrixType;
    typedef typename MatrixLinearSolverInternalData<Vector>::ResMatrixType ResMatrixType;

    MatrixLinearSolver();
    ~MatrixLinearSolver() override ;

    void init() override;

    /// Reset the current linear system.
    SOFA_ITERATIVE_SOLVER_ATTRIBUTE_REMOVE_ASSEMBLY_API()
    void resizeSystem(Size n) = delete;

    /// Get the linear system right-hand term vector, or nullptr if this solver does not build it
    SOFA_ITERATIVE_SOLVER_ATTRIBUTE_DEPRECATED_ASSEMBLY_API()
    Vector* getSystemRHVector() { return l_linearSystem ? l_linearSystem->getRHSVector() : nullptr; }

    /// Get the linear system left-hand term vector, or nullptr if this solver does not build it
    SOFA_ITERATIVE_SOLVER_ATTRIBUTE_DEPRECATED_ASSEMBLY_API()
    Vector* getSystemLHVector() { return l_linearSystem ? l_linearSystem->getSolutionVector() : nullptr; }

    /// Returns the linear system component associated to the linear solver
    sofa::component::linearsystem::TypedMatrixLinearSystem<Matrix, Vector>* getLinearSystem() const override { return l_linearSystem.get(); }

    /// Solve the system as constructed using the previous methods
    void solveSystem() override;

    /// Apply the solution of the system to all the objects
    SOFA_ITERATIVE_SOLVER_ATTRIBUTE_REMOVE_ASSEMBLY_API()
    void applySystemSolution() = delete;

    /// Invert the system, this method is optional because it's call when solveSystem() is called for the first time
    void invertSystem() override;

    /// Implementing the GetCustomTemplateName is mandatory to have a custom template name parameters
    /// instead of the default one generated automatically by the SOFA_CLASS() macro.
    static std::string GetCustomTemplateName()
    {
        return ThreadManager::Name()+Matrix::Name();
    }

    bool isAsyncSolver() override
    {
        return ThreadManager::isAsyncSolver();
    }

    virtual void invertIfInvalidated(Matrix& M) final;

    void invert(Matrix& /*M*/) override {}

    void solve(Matrix& M, Vector& solution, Vector& rh) override = 0;

    virtual bool addJMInvJtLocal(Matrix * /*M*/,ResMatrixType * result,const JMatrixType * J, SReal fact);

    virtual bool addMInvJtLocal(Matrix * /*M*/,ResMatrixType * result,const  JMatrixType * J, SReal fact);

    bool addJMInvJt(linearalgebra::BaseMatrix* result, linearalgebra::BaseMatrix* J, SReal fact) override;

    bool addMInvJt(linearalgebra::BaseMatrix* result, linearalgebra::BaseMatrix* J, SReal fact) override;

    bool buildComplianceMatrix(const core::ConstraintParams* cparams, linearalgebra::BaseMatrix* result, SReal fact, SReal regularizationTerm) override;

    void applyConstraintForce(const sofa::core::ConstraintParams* cparams, sofa::core::MultiVecDerivId dx, const linearalgebra::BaseVector* f) override;

    void computeResidual(const core::ExecParams* params, linearalgebra::BaseVector* f) override;

    ///< Parallelize the computation of the product J*M^{-1}*J^T where M is the
    ///< matrix of the linear system and J is any matrix with compatible dimensions
    Data<bool> d_parallelInverseProduct;

    SingleLink<
        MatrixLinearSolver<Matrix,Vector,NoThreadManager>,
        sofa::component::linearsystem::TypedMatrixLinearSystem<Matrix, Vector>,
        BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK
    > l_linearSystem;

    MatrixInvertData * getMatrixInvertData(linearalgebra::BaseMatrix * m);

protected:

    void prepareVisitor(simulation::Visitor* v)
    {
        v->setTags(this->getTags());
    }

    void prepareVisitor(simulation::BaseMechanicalVisitor* v)
    {
        prepareVisitor(static_cast<simulation::Visitor*>(v));
    }

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


    virtual void checkLinearSystem();

    /**
     * Check if compatible linear systems are available in the current context. Otherwise, a linear
     * system of type TLinearSystemType is created, with a warning to the user.
     *
     * @tparam TLinearSystemType Type of linear system created if no linear system found in the
     * current context.
     */
    template<class TLinearSystemType>
    void doCheckLinearSystem();

    template<class TLinearSystemType>
    void createDefaultLinearSystem();

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

    MatrixLinearSolverInternalData<Vector> internalData;
    std::unique_ptr<MatrixInvertData> invertData;

    virtual MatrixInvertData * createInvertData();

    SOFA_ITERATIVE_SOLVER_ATTRIBUTE_REMOVE_ASSEMBLY_API()
    DeprecatedAndRemoved linearSystem;

    SOFA_ITERATIVE_SOLVER_ATTRIBUTE_REMOVE_ASSEMBLY_API()
    DeprecatedAndRemoved currentMFactor, currentBFactor, currentKFactor;

    bool singleThreadAddJMInvJtLocal(Matrix * /*M*/,ResMatrixType * result,const JMatrixType * J, SReal fact);

    Data<bool> d_factorizationInvalidation;
};

//////////////////////////////////////////////////////////////
//Specialization for GraphScatteredTypes
//////////////////////////////////////////////////////////////
template<>
class MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::TempVectorContainer
{
public:
    MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>* parent;
    simulation::common::VectorOperations vops;
    simulation::common::MechanicalOperations mops;
    GraphScatteredMatrix* matrix;
    TempVectorContainer(MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>* p, const core::ExecParams* params, GraphScatteredMatrix& M, GraphScatteredVector& x, GraphScatteredVector& b)
        : parent(p), vops(params, p->getContext()), mops(M.mparams.setExecParams(params), p->getContext()), matrix(&M)
    {
        x.setOps( &vops );
        b.setOps( &vops );
        M.parent = &mops;
    }
    GraphScatteredVector* createTempVector() { return new GraphScatteredVector(&vops); }
    void deleteTempVector(GraphScatteredVector* v) { delete v; }
};

template<> SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API
GraphScatteredVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::createPersistentVector();

template<> SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::applyConstraintForce(const sofa::core::ConstraintParams* /*cparams*/, sofa::core::MultiVecDerivId /*dx*/, const linearalgebra::BaseVector* /*f*/);

template<> SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::computeResidual(const core::ExecParams* params,linearalgebra::BaseVector* f);

template<> SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::checkLinearSystem();

#if !defined(SOFA_COMPONENT_LINEARSOLVER_MATRIXLINEARSOLVER_CPP)
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver< GraphScatteredMatrix, GraphScatteredVector, NoThreadManager >;
/// Extern template declarations don't prevent implicit instantiation in the case
/// of explicitly specialized classes.  (See section 14.3.7 of the C++ standard
/// [temp.expl.spec]). We have to declare non-specialized member functions by
/// hand to prevent MSVC from complaining that it doesn't find their definition.
extern template SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::MatrixLinearSolver();
extern template SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::~MatrixLinearSolver();
extern template SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::invertSystem();
extern template SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API bool MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::addJMInvJt(linearalgebra::BaseMatrix*, linearalgebra::BaseMatrix*, SReal);
extern template SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API bool MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::addMInvJt(linearalgebra::BaseMatrix*, linearalgebra::BaseMatrix*, SReal);
extern template SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API bool MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::addJMInvJtLocal(GraphScatteredMatrix*, ResMatrixType*, const JMatrixType*, SReal);
extern template SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API bool MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::addMInvJtLocal(GraphScatteredMatrix*, ResMatrixType*, const  JMatrixType*, SReal);
extern template SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API bool MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::buildComplianceMatrix(const core::ConstraintParams*, linearalgebra::BaseMatrix*, SReal, SReal);
extern template SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixInvertData* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::getMatrixInvertData(linearalgebra::BaseMatrix * m);
extern template SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixInvertData* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::createInvertData();

extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver< linearalgebra::FullMatrix<SReal>, linearalgebra::FullVector<SReal>, NoThreadManager >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver< linearalgebra::SparseMatrix<SReal>, linearalgebra::FullVector<SReal>, NoThreadManager >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver< linearalgebra::CompressedRowSparseMatrix<SReal>, linearalgebra::FullVector<SReal>, NoThreadManager >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver< linearalgebra::CompressedRowSparseMatrix<type::Mat<2,2,SReal> >, linearalgebra::FullVector<SReal>, NoThreadManager >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver< linearalgebra::CompressedRowSparseMatrix<type::Mat<3,3,SReal> >, linearalgebra::FullVector<SReal>, NoThreadManager >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver< linearalgebra::CompressedRowSparseMatrix<type::Mat<4,4,SReal> >, linearalgebra::FullVector<SReal>, NoThreadManager >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver< linearalgebra::CompressedRowSparseMatrix<type::Mat<6,6,SReal> >, linearalgebra::FullVector<SReal>, NoThreadManager >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver< linearalgebra::CompressedRowSparseMatrix<type::Mat<8,8,SReal> >, linearalgebra::FullVector<SReal>, NoThreadManager >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver< linearalgebra::DiagonalMatrix<SReal>, linearalgebra::FullVector<SReal>, NoThreadManager >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver< linearalgebra::BlockDiagonalMatrix<3,SReal>, linearalgebra::FullVector<SReal>, NoThreadManager >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixLinearSolver< linearalgebra::RotationMatrix<SReal>, linearalgebra::FullVector<SReal>, NoThreadManager >;
#endif


} // namespace sofa::component::linearsolver
