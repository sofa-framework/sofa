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
#ifndef SOFA_COMPONENT_LINEARSOLVER_MATRIXLINEARSOLVER_H
#define SOFA_COMPONENT_LINEARSOLVER_MATRIXLINEARSOLVER_H
#include "config.h"

#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/MechanicalMatrixVisitor.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <SofaBaseLinearSolver/DefaultMultiMatrixAccessor.h>
#include <SofaBaseLinearSolver/GraphScatteredTypes.h>
#include <SofaBaseLinearSolver/FullMatrix.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>
#include <SofaBaseLinearSolver/DiagonalMatrix.h>
#include <sofa/core/behavior/RotationMatrix.h>

namespace sofa
{

namespace component
{

namespace linearsolver
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
class MatrixLinearSolverInternalData
{
public:
    typedef typename TVector::Real Real;
    typedef SparseMatrix<Real> JMatrixType;
    typedef defaulttype::BaseMatrix ResMatrixType;

    template<typename MReal>
    JMatrixType * copyJmatrix(SparseMatrix<MReal> * J)
    {
        J_local.clear();
        J_local.resize(J->rowSize(),J->colSize());

        for (typename sofa::component::linearsolver::SparseMatrix<MReal>::LineConstIterator jit1 = J->begin(); jit1 != J->end(); jit1++)
        {
            int l = jit1->first;
            for (typename sofa::component::linearsolver::SparseMatrix<MReal>::LElementConstIterator i1 = jit1->second.begin(); i1 != jit1->second.end(); i1++)
            {
                int c = i1->first;
                MReal val = i1->second;
                J_local.set(l,c,val);
            }
        }
        return &J_local;
    }

    void projectForceInConstraintSpace(defaulttype::BaseVector* r,const defaulttype::BaseVector* f) {
        for (typename SparseMatrix<Real>::LineConstIterator jit = J_local.begin(), jitend = J_local.end(); jit != jitend; ++jit) {
            int row = jit->first;
            double force = f->element(row);
            for (typename SparseMatrix<Real>::LElementConstIterator i2 = jit->second.begin(), i2end = jit->second.end(); i2 != i2end; ++i2) {
                int col = i2->first;
                double val = i2->second;
                r->add(col,val * force);
            }
        }
    }

    JMatrixType * getLocalJ() {
        return &J_local;
    }

    JMatrixType * getLocalJ(defaulttype::BaseMatrix * J)
    {
        if (JMatrixType * j = dynamic_cast<JMatrixType *>(J))
        {
            return j;
        }
        else if (SparseMatrix<double> * j = dynamic_cast<SparseMatrix<double> *>(J))
        {
            return copyJmatrix(j);
        }
        else if (SparseMatrix<float> * j = dynamic_cast<SparseMatrix<float> *>(J))
        {
            return copyJmatrix(j);
        }
        else
        {
            J_local.clear();
            J_local.resize(J->rowSize(),J->colSize());

            for (typename JMatrixType::Index j=0; j<J->rowSize(); j++)
            {
                for (typename JMatrixType::Index i=0; i<J->colSize(); i++)
                {
                    J_local.set(j,i,J->element(j,i));
                }
            }

            return &J_local;
        }
    }

    ResMatrixType * getLocalRes(defaulttype::BaseMatrix * R)
    {
        return R;
    }


    void addLocalRes(defaulttype::BaseMatrix * /*R*/)
    {
        return ;
    }

private :
    JMatrixType J_local;
//    ResMatrixType res_data;
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
    typedef std::list<int> ListIndex;
    typedef typename Vector::Real Real;
    typedef typename MatrixLinearSolverInternalData<Vector>::JMatrixType JMatrixType;
    typedef typename MatrixLinearSolverInternalData<Vector>::ResMatrixType ResMatrixType;

    Data<bool> multiGroup;

    MatrixLinearSolver();
    virtual ~MatrixLinearSolver();

    /// Reset the current linear system.
    void resetSystem() override;

    /// Reset the current linear system.
    void resizeSystem(int n);

    /// Set the linear system matrix, combining the mechanical M,B,K matrices using the given coefficients
    ///
    /// Note that this automatically resizes the linear system to the number of active degrees of freedoms
    ///
    /// @todo Should we put this method in a specialized class for mechanical systems, or express it using more general terms (i.e. coefficients of the second order ODE to solve)
    void setSystemMBKMatrix(const core::MechanicalParams* mparams) override;

    /// Rebuild the system using a mass and force factor
    virtual void rebuildSystem(double massFactor, double forceFactor) override;

    /// Set the linear system matrix (only use for bench)
    void setSystemMatrix(Matrix* matrix);

    unsigned getSystemSize() {
        return currentGroup->systemSize;
    }

    /// Set the linear system right-hand term vector, from the values contained in the (Mechanical/Physical)State objects
    void setSystemRHVector(core::MultiVecDerivId v) override;

    /// Set the initial estimate of the linear system left-hand term vector, from the values contained in the (Mechanical/Physical)State objects
    /// This vector will be replaced by the solution of the system once solveSystem is called
    void setSystemLHVector(core::MultiVecDerivId v) override;

    /// Get the linear system matrix, or NULL if this solver does not build it
    Matrix* getSystemMatrix() override { return currentGroup->systemMatrix; }

    /// Get the linear system right-hand term vector, or NULL if this solver does not build it
    Vector* getSystemRHVector() { return currentGroup->systemRHVector; }

    /// Get the linear system left-hand term vector, or NULL if this solver does not build it
    Vector* getSystemLHVector() { return currentGroup->systemLHVector; }

    /// Get the linear system matrix, or NULL if this solver does not build it
    defaulttype::BaseMatrix* getSystemBaseMatrix() override { return currentGroup->systemMatrix; }

    /// Get the linear system right-hand term vector, or NULL if this solver does not build it
    defaulttype::BaseVector* getSystemRHBaseVector() override { return currentGroup->systemRHVector; }

    /// Get the linear system left-hand term vector, or NULL if this solver does not build it
    defaulttype::BaseVector* getSystemLHBaseVector() override { return currentGroup->systemLHVector; }

    /// Solve the system as constructed using the previous methods
    virtual void solveSystem() override;

    /// Invert the system, this method is optional because it's call when solveSystem() is called for the first time
    virtual void invertSystem() override;

    void prepareVisitor(simulation::Visitor* v)
    {
        v->setTags(this->getTags());
    }

    void prepareVisitor(simulation::BaseMechanicalVisitor* v)
    {
        prepareVisitor((simulation::Visitor*)v);
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

    static std::string templateName(const MatrixLinearSolver<Matrix,Vector,ThreadManager>* = NULL)
    {
        return ThreadManager::Name()+Matrix::Name();
    }

    virtual bool isAsyncSolver() override
    {
        return ThreadManager::isAsyncSolver();
    }

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }


    virtual void invert(Matrix& /*M*/) override {}

    virtual void solve(Matrix& M, Vector& solution, Vector& rh) override = 0;

    virtual bool addJMInvJtLocal(Matrix * /*M*/,ResMatrixType * result,const JMatrixType * J, double fact);

    virtual bool addMInvJtLocal(Matrix * /*M*/,ResMatrixType * result,const  JMatrixType * J, double fact);

    virtual bool addJMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact) override;

    virtual bool addMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact) override;

    virtual bool buildComplianceMatrix(defaulttype::BaseMatrix* result, double fact) override;

    virtual void applyContactForce(const defaulttype::BaseVector* f,double positionFactor,double velocityFactor) override;

    virtual void computeResidual(const core::ExecParams* params, defaulttype::BaseVector* f) override;

public :
    bool isMultiGroup() const override
    {
        return multiGroup.getValue();
    }

    virtual void createGroups(const core::MechanicalParams* mparams);

    int getNbGroups() const
    {
        if (isMultiGroup())
            return (int)this->groups.size();
        else
            return 1;
    }

    void setGroup(int i)
    {
        //serr << "setGroup("<<i<<")" << sendl;
        if (isMultiGroup() && (unsigned)i < this->groups.size())
        {
            currentNode = groups[i];
            currentGroup = &(gData[currentNode]);
        }
        else
        {
            currentNode = dynamic_cast<simulation::Node*>(this->getContext());
            currentGroup = &defaultGroup;
        }
    }

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

    MatrixLinearSolverInternalData<Vector> internalData;
    MatrixInvertData * invertData;

    virtual MatrixInvertData * createInvertData();

    class GroupData
    {
    public:
        int systemSize;
        bool needInvert;
        Matrix* systemMatrix;
        Vector* systemRHVector;
        Vector* systemLHVector;
        core::MultiVecDerivId solutionVecId;

#ifdef SOFA_SUPPORT_CRS_MATRIX
        CRSMultiMatrixAccessor matrixAccessor;
#else
        DefaultMultiMatrixAccessor matrixAccessor;
#endif
        GroupData()
            : systemSize(0), needInvert(true), systemMatrix(NULL), systemRHVector(NULL), systemLHVector(NULL), solutionVecId(core::MultiVecDerivId::null())
        {}
        ~GroupData()
        {
            if (systemMatrix) deleteMatrix(systemMatrix);
            if (systemRHVector) deletePersistentVector(systemRHVector);
            if (systemLHVector) deletePersistentVector(systemLHVector);
        }
    };

    typedef std::map<simulation::Node*,GroupData> GroupDataMap;
    typedef typename GroupDataMap::iterator GroupDataMapIter;
    simulation::Node* currentNode;
    GroupData* currentGroup;
    std::vector<simulation::Node*> groups;
    GroupDataMap gData;
    GroupData defaultGroup;

    double currentMFactor, currentBFactor, currentKFactor;

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

template<> SOFA_BASE_LINEAR_SOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::resetSystem();

template<> SOFA_BASE_LINEAR_SOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::resizeSystem(int);

template<> SOFA_BASE_LINEAR_SOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::setSystemMBKMatrix(const core::MechanicalParams* mparams);

template<> SOFA_BASE_LINEAR_SOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::rebuildSystem(double massFactor, double forceFactor);

template<> SOFA_BASE_LINEAR_SOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::setSystemRHVector(core::MultiVecDerivId v);

template<> SOFA_BASE_LINEAR_SOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::setSystemLHVector(core::MultiVecDerivId v);

template<> SOFA_BASE_LINEAR_SOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::solveSystem();

template<> SOFA_BASE_LINEAR_SOLVER_API
GraphScatteredVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::createPersistentVector();

template<> SOFA_BASE_LINEAR_SOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::deletePersistentVector(GraphScatteredVector* v);

template<> SOFA_BASE_LINEAR_SOLVER_API
GraphScatteredMatrix* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::createMatrix();

template<> SOFA_BASE_LINEAR_SOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::deleteMatrix(GraphScatteredMatrix* v);

template<> SOFA_BASE_LINEAR_SOLVER_API
defaulttype::BaseMatrix* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::getSystemBaseMatrix();

template<> SOFA_BASE_LINEAR_SOLVER_API
defaulttype::BaseVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::getSystemRHBaseVector();

template<> SOFA_BASE_LINEAR_SOLVER_API
defaulttype::BaseVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::getSystemLHBaseVector();

template<> SOFA_BASE_LINEAR_SOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::setSystemMatrix(GraphScatteredMatrix * matrix);

template<> SOFA_BASE_LINEAR_SOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::applyContactForce(const defaulttype::BaseVector* f,double positionFactor,double velocityFactor);

template<> SOFA_BASE_LINEAR_SOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::computeResidual(const core::ExecParams* params,defaulttype::BaseVector* f);

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_LINEARSOLVER_MATRIXLINEARSOLVER_CPP)
extern template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< GraphScatteredMatrix, GraphScatteredVector, NoThreadManager >;
// Extern template declarations don't prevent implicit instanciation in the case
// of explicitely specialized classes.  (See section 14.3.7 of the C++ standard
// [temp.expl.spec]). We have to declare non-specialized member functions by
// hand to prevent MSVC from complaining that it doesn't find their definition.
extern template SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::MatrixLinearSolver();
extern template SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::~MatrixLinearSolver();
extern template SOFA_BASE_LINEAR_SOLVER_API void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::invertSystem();
extern template SOFA_BASE_LINEAR_SOLVER_API bool MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::addJMInvJt(defaulttype::BaseMatrix*, defaulttype::BaseMatrix*, double);
extern template SOFA_BASE_LINEAR_SOLVER_API bool MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::addMInvJt(defaulttype::BaseMatrix*, defaulttype::BaseMatrix*, double);
extern template SOFA_BASE_LINEAR_SOLVER_API bool MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::addJMInvJtLocal(GraphScatteredMatrix*, ResMatrixType*, const JMatrixType*, double);
extern template SOFA_BASE_LINEAR_SOLVER_API bool MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::addMInvJtLocal(GraphScatteredMatrix*, ResMatrixType*, const  JMatrixType*, double);
extern template SOFA_BASE_LINEAR_SOLVER_API bool MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::buildComplianceMatrix(defaulttype::BaseMatrix*, double);
extern template SOFA_BASE_LINEAR_SOLVER_API void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::createGroups(const core::MechanicalParams* mparams);
extern template SOFA_BASE_LINEAR_SOLVER_API MatrixInvertData* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::getMatrixInvertData(defaulttype::BaseMatrix * m);
extern template SOFA_BASE_LINEAR_SOLVER_API MatrixInvertData* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::createInvertData();

extern template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< FullMatrix<double>, FullVector<double>, NoThreadManager >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< FullMatrix<float>, FullVector<float>, NoThreadManager >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< SparseMatrix<double>, FullVector<double>, NoThreadManager >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< SparseMatrix<float>, FullVector<float>, NoThreadManager >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<double>, FullVector<double>, NoThreadManager >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<float>, FullVector<float>, NoThreadManager >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<2,2,double> >, FullVector<double>, NoThreadManager >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<2,2,float> >, FullVector<float>, NoThreadManager >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<3,3,double> >, FullVector<double>, NoThreadManager >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<3,3,float> >, FullVector<float>, NoThreadManager >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<4,4,double> >, FullVector<double>, NoThreadManager >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<4,4,float> >, FullVector<float>, NoThreadManager >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<6,6,double> >, FullVector<double>, NoThreadManager >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<6,6,float> >, FullVector<float>, NoThreadManager >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<8,8,double> >, FullVector<double>, NoThreadManager >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< CompressedRowSparseMatrix<defaulttype::Mat<8,8,float> >, FullVector<float>, NoThreadManager >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< DiagonalMatrix<double>, FullVector<double>, NoThreadManager >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< DiagonalMatrix<float>, FullVector<float>, NoThreadManager >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< BlockDiagonalMatrix<3,double>, FullVector<double>, NoThreadManager >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< RotationMatrix<double>, FullVector<double>, NoThreadManager >;
extern template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< RotationMatrix<float>, FullVector<float>, NoThreadManager >;
#endif


} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
