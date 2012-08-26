/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_MATRIXLINEARSOLVER_H
#define SOFA_COMPONENT_LINEARSOLVER_MATRIXLINEARSOLVER_H

#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/MechanicalMatrixVisitor.h>
#include <sofa/simulation/common/MechanicalOperations.h>
#include <sofa/simulation/common/VectorOperations.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/component.h>
#include <sofa/component/linearsolver/DefaultMultiMatrixAccessor.h>
#include <sofa/component/linearsolver/GraphScatteredTypes.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/component/linearsolver/CompressedRowSparseMatrix.h>

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

            for (unsigned j=0; j<J->rowSize(); j++)
            {
                for (unsigned i=0; i<J->colSize(); i++)
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

#ifdef WARPING_SOLVER
    typedef RotationMatrix<Real> TRotationMatrix;

    //compute J = J * R
    void opMulJ(TRotationMatrix * R,JMatrixType * J)
    {
        for (typename sofa::component::linearsolver::SparseMatrix<Real>::LineConstIterator jit1 = J->begin(); jit1 != J->end(); jit1++)
        {
            int l = jit1->first;
            for (typename sofa::component::linearsolver::SparseMatrix<Real>::LElementConstIterator i1 = jit1->second.begin(); i1 != jit1->second.end();)
            {
                int c = i1->first;
                Real v0 = (Real)i1->second; i1++; if (i1==jit1->second.end()) break;
                Real v1 = (Real)i1->second; i1++; if (i1==jit1->second.end()) break;
                Real v2 = (Real)i1->second; i1++;
                J->set(l,c+0,v0 * R->getVector()[(c+0)*3+0] + v1 * R->getVector()[(c+1)*3+0] + v2 * R->getVector()[(c+2)*3+0] );
                J->set(l,c+1,v0 * R->getVector()[(c+0)*3+1] + v1 * R->getVector()[(c+1)*3+1] + v2 * R->getVector()[(c+2)*3+1] );
                J->set(l,c+2,v0 * R->getVector()[(c+0)*3+2] + v1 * R->getVector()[(c+1)*3+2] + v2 * R->getVector()[(c+2)*3+2] );
            }
        }
    }
#endif

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

    typedef NoThreadManager ThreadManager;
    typedef std::list<int> ListIndex;
    typedef typename Vector::Real Real;
    typedef typename MatrixLinearSolverInternalData<Vector>::JMatrixType JMatrixType;
    typedef typename MatrixLinearSolverInternalData<Vector>::ResMatrixType ResMatrixType;

    Data<bool> multiGroup;

    MatrixLinearSolver();
    virtual ~MatrixLinearSolver();

    /// Reset the current linear system.
    void resetSystem();

    /// Reset the current linear system.
    void resizeSystem(int n);

    /// Set the linear system matrix, combining the mechanical M,B,K matrices using the given coefficients
    ///
    /// Note that this automatically resizes the linear system to the number of active degrees of freedoms
    ///
    /// @todo Should we put this method in a specialized class for mechanical systems, or express it using more general terms (i.e. coefficients of the second order ODE to solve)
    void setSystemMBKMatrix(const core::MechanicalParams* mparams);

    /// Set the linear system right-hand term vector, from the values contained in the (Mechanical/Physical)State objects
    void setSystemRHVector(core::MultiVecDerivId v);

    /// Set the initial estimate of the linear system left-hand term vector, from the values contained in the (Mechanical/Physical)State objects
    /// This vector will be replaced by the solution of the system once solveSystem is called
    void setSystemLHVector(core::MultiVecDerivId v);

    /// Get the linear system matrix, or NULL if this solver does not build it
    Matrix* getSystemMatrix() { return currentGroup->systemMatrix; }

    /// Get the linear system right-hand term vector, or NULL if this solver does not build it
    Vector* getSystemRHVector() { return currentGroup->systemRHVector; }

    /// Get the linear system left-hand term vector, or NULL if this solver does not build it
    Vector* getSystemLHVector() { return currentGroup->systemLHVector; }

    /// Get the linear system matrix, or NULL if this solver does not build it
    defaulttype::BaseMatrix* getSystemBaseMatrix() { return currentGroup->systemMatrix; }

    /// Get the linear system right-hand term vector, or NULL if this solver does not build it
    defaulttype::BaseVector* getSystemRHBaseVector() { return currentGroup->systemRHVector; }

    /// Get the linear system left-hand term vector, or NULL if this solver does not build it
    defaulttype::BaseVector* getSystemLHBaseVector() { return currentGroup->systemLHVector; }

    /// Solve the system as constructed using the previous methods
    virtual void solveSystem();

    /// Invert the system, this method is optional because it's call when solveSystem() is called for the first time
    virtual void invertSystem()
    {
        for (unsigned int g=0, nbg = getNbGroups(); g < nbg; ++g)
        {
            setGroup(g);
            if (currentGroup->needInvert)
            {
                this->invert(*currentGroup->systemMatrix);
                currentGroup->needInvert = false;
            }
        }
    }

    void prepareVisitor(Visitor* v)
    {
        v->setTags(this->getTags());
    }

    void prepareVisitor(simulation::BaseMechanicalVisitor* v)
    {
        prepareVisitor((Visitor*)v);
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

    virtual bool isAsyncSolver()
    {
        return ThreadManager::isAsyncSolver();
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }


    virtual void invert(Matrix& /*M*/) {}

    virtual void solve(Matrix& M, Vector& solution, Vector& rh) = 0;


    /// Default implementation of Multiply the inverse of the system matrix by the transpose of the given matrix, and multiply the result with the given matrix J
    ///
    /// TODO : put this implementation in MatrixLinearSolver class - fix problems with Scattered Matrix

    virtual bool addJMInvJtLocal(Matrix * /*M*/,ResMatrixType * result, JMatrixType * J, double fact)
    {
        for (unsigned row=0; row<J->rowSize(); row++)
        {
            // STEP 1 : put each line of matrix Jt in the right hand term of the system
            for (unsigned i=0; i<J->colSize(); i++) currentGroup->systemRHVector->set(i,J->element(row,i)); // currentGroup->systemMatrix->rowSize()

            // STEP 2 : solve the system :
            solveSystem();

            // STEP 3 : project the result using matrix J
            if (SparseMatrix<Real> * j = dynamic_cast<SparseMatrix<Real> * >(J))   // optimization for sparse matrix
            {
                const typename SparseMatrix<Real>::LineConstIterator jitend = j->end();
                for (typename SparseMatrix<Real>::LineConstIterator jit = j->begin(); jit != jitend; ++jit)
                {
                    int row2 = jit->first;
                    double acc = 0.0;
                    for (typename SparseMatrix<Real>::LElementConstIterator i2 = jit->second.begin(), i2end = jit->second.end(); i2 != i2end; ++i2)
                    {
                        int col2 = i2->first;
                        double val2 = i2->second;
                        acc += val2 * currentGroup->systemLHVector->element(col2);
                    }
                    acc *= fact;
                    result->add(row2,row,acc);
                }
            }
            else
            {
                std::cerr << "AsyncMatrixLinearSolver::addJMInvJt is only implemented for SparseMatrix<Real>" << std::endl;
                return false;
            }
        }

        return true;
    }

    virtual bool addMInvJtLocal(Matrix * /*M*/,ResMatrixType * result, JMatrixType * J, double fact)
    {
        for (unsigned row=0; row<J->rowSize(); row++)
        {
            // STEP 1 : put each line of matrix Jt in the right hand term of the system
            for (unsigned i=0; i<J->colSize(); i++) currentGroup->systemRHVector->set(i,J->element(row,i)); // currentGroup->systemMatrix->rowSize()

            // STEP 2 : solve the system :
            solveSystem();

            // STEP 3 : project the result using matrix J
            for (unsigned i=0; i<J->colSize(); i++) result->add(row,i,currentGroup->systemRHVector->element(i) * fact);
        }

        return true;
    }

    //// The following functions should not be overload
    //// only addMInvJtLocal and addJMInvJtLocal can be overload

    /// Default implementation of Multiply the inverse of the system matrix by the transpose of the given matrix, and multiply the result with the given matrix J
    ///
    /// @param result the variable where the result will be added
    /// @param J the matrix J to use
    /// @return false if the solver does not support this operation, of it the system matrix is not invertible
    virtual bool addJMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact)
    {
        if (J->rowSize()==0) return true;

        JMatrixType * j_local = internalData.getLocalJ(J);
        ResMatrixType * res_local = internalData.getLocalRes(result);
        bool res = addJMInvJtLocal(currentGroup->systemMatrix,res_local,j_local,fact);
        internalData.addLocalRes(result);
        return res;
    }

    virtual bool addMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact)
    {
        if (J->rowSize()==0) return true;

        JMatrixType * j_local = internalData.getLocalJ(J);
        ResMatrixType * res_local = internalData.getLocalRes(result);
        bool res = addMInvJtLocal(currentGroup->systemMatrix,res_local,j_local,fact);
        internalData.addLocalRes(result);
        return res;
    }
public :
    bool isMultiGroup() const
    {
        return multiGroup.getValue();
    }

    virtual void createGroups(const core::MechanicalParams* mparams);

    int getNbGroups() const
    {
        if (isMultiGroup()) return this->groups.size();
        else return 1;
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

    MatrixInvertData * getMatrixInvertData(Matrix * m);

protected:

    /// newPartially solve the system
    virtual void partial_solve(Matrix& /*M*/, Vector& /*partial_solution*/, Vector& /*sparse_rh*/, ListIndex& /* indices_solution*/, ListIndex& /* indices input */)
    {
        std::cerr<<" WARNING : partial_solve is not implemented for this solver"<<std::endl;
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

};

template<class Matrix, class Vector>
MatrixLinearSolver<Matrix,Vector>::MatrixLinearSolver()
    : multiGroup( initData( &multiGroup, false, "multiGroup", "activate multiple system solve, one for each child node" ) )
//, needInvert(true), systemMatrix(NULL), systemRHVector(NULL), systemLHVector(NULL)
    , currentGroup(&defaultGroup)
{
    invertData = NULL;
}

template<class Matrix, class Vector>
MatrixLinearSolver<Matrix,Vector>::~MatrixLinearSolver()
{
    //if (systemMatrix) deleteMatrix(systemMatrix);
    //if (systemRHVector) deleteVector(systemRHVector);
    //if (systemLHVector) deleteVector(systemLHVector);
    if (invertData) delete invertData;
    invertData = NULL;
}

template<class Matrix, class Vector>
MatrixInvertData * MatrixLinearSolver<Matrix,Vector>::getMatrixInvertData(Matrix * /*m*/)
{
    if (invertData==NULL) invertData=createInvertData();
    return invertData;
}

template<class Matrix, class Vector>
MatrixInvertData * MatrixLinearSolver<Matrix,Vector>::createInvertData()
{
    std::cerr << "ERROR the solver didn't implement MatrixLinearSolver::getMatrixInvertData this function is not available in MatrixLinearSolver, NULL is return" << std::endl;
    return NULL;
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::createGroups(const core::MechanicalParams* mparams)
{
    simulation::Node* root = dynamic_cast<simulation::Node*>(this->getContext());
    //defaultGroup.node = root;
    for (GroupDataMapIter it = gData.begin(), itend = gData.end(); it != itend; ++it)
        it->second.systemSize = 0;
    if (isMultiGroup())
    {
        double dim = 0;
        groups.clear();
        for (unsigned int g=0; g<root->child.size(); ++g)
        {
            simulation::Node* n = root->child[g].get();
            double gdim = 0;
            simulation::MechanicalGetDimensionVisitor(mparams /* PARAMS FIRST */, &gdim).execute(n);
            if (gdim <= 0) continue;
            groups.push_back(n);
            gData[n].systemSize = (int)gdim;
            dim += gdim;
        }
        defaultGroup.systemSize = (int)dim;
    }
    else
    {
        groups.clear();
        double dim = 0;
        simulation::MechanicalGetDimensionVisitor(mparams /* PARAMS FIRST */, &dim).execute(root);
        defaultGroup.systemSize = (int)dim;
    }
    currentNode = root;
    currentGroup = &defaultGroup;
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::resetSystem()
{
    for (unsigned int g=0, nbg = getNbGroups(); g < nbg; ++g)
    {
        setGroup(g);
        if (!this->frozen)
        {
            if (currentGroup->systemMatrix) currentGroup->systemMatrix->clear();
            currentGroup->needInvert = true;
        }
        if (currentGroup->systemRHVector) currentGroup->systemRHVector->clear();
        if (currentGroup->systemLHVector) currentGroup->systemLHVector->clear();
        currentGroup->solutionVecId = core::MultiVecDerivId::null();
    }
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::resizeSystem(int n)
{
    if (!this->frozen)
    {
        if (!currentGroup->systemMatrix) currentGroup->systemMatrix = createMatrix();
        currentGroup->systemMatrix->resize(n,n);
    }
    if (!currentGroup->systemRHVector) currentGroup->systemRHVector = createPersistentVector();
    currentGroup->systemRHVector->resize(n);
    if (!currentGroup->systemLHVector) currentGroup->systemLHVector = createPersistentVector();
    currentGroup->systemLHVector->resize(n);
    currentGroup->needInvert = true;
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::setSystemMBKMatrix(const core::MechanicalParams* mparams)
{
    createGroups(mparams);
    for (unsigned int g=0, nbg = getNbGroups(); g < nbg; ++g)
    {
        setGroup(g);
        if (!this->frozen)
        {
            simulation::common::MechanicalOperations mops(mparams /* PARAMS FIRST */, this->getContext());
            if (!currentGroup->systemMatrix) currentGroup->systemMatrix = createMatrix();
            currentGroup->matrixAccessor.setGlobalMatrix(currentGroup->systemMatrix);
            currentGroup->matrixAccessor.clear();
            //unsigned int nbRow=0, nbCol=0;
            //MechanicalGetMatrixDimensionVisitor(nbRow, nbCol).execute( getContext() );
            //this->getMatrixDimension(&nbRow, &nbCol);
            //resizeSystem(nbRow);
            mops.getMatrixDimension(&(currentGroup->matrixAccessor));
            currentGroup->matrixAccessor.setupMatrices();
            resizeSystem(currentGroup->matrixAccessor.getGlobalDimension());
            currentGroup->systemMatrix->clear();
            //unsigned int offset = 0;
            //MechanicalAddMBK_ToMatrixVisitor(currentGroup->systemMatrix, mFact, bFact, kFact, offset).execute( getContext() );
            mops.addMBK_ToMatrix(&(currentGroup->matrixAccessor), mparams->mFactor(), mparams->bFactor(), mparams->kFactor());
            //this->addMBK_ToMatrix(&(currentGroup->matrixAccessor), mFact, bFact, kFact);
            currentGroup->matrixAccessor.computeGlobalMatrix();
        }
    }
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::setSystemRHVector(core::MultiVecDerivId v)
{
    for (unsigned int g=0, nbg = getNbGroups(); g < nbg; ++g)
    {
        setGroup(g);
        //this->multiVector2BaseVector(v, currentGroup->systemRHVector, &(currentGroup->matrixAccessor));
        executeVisitor( simulation::MechanicalMultiVectorToBaseVectorVisitor(core::ExecParams::defaultInstance(), v, currentGroup->systemRHVector, &(currentGroup->matrixAccessor)) );
    }
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::setSystemLHVector(core::MultiVecDerivId v)
{
    for (unsigned int g=0, nbg = getNbGroups(); g < nbg; ++g)
    {
        setGroup(g);
        currentGroup->solutionVecId = v;
        //this->multiVector2BaseVector(v, currentGroup->systemLHVector, &(currentGroup->matrixAccessor));
        executeVisitor( simulation::MechanicalMultiVectorToBaseVectorVisitor( core::ExecParams::defaultInstance(), v, currentGroup->systemLHVector, &(currentGroup->matrixAccessor)) );
    }
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::solveSystem()
{
    for (unsigned int g=0, nbg = getNbGroups(); g < nbg; ++g)
    {
        setGroup(g);
        if (currentGroup->needInvert)
        {
            this->invert(*currentGroup->systemMatrix);
            currentGroup->needInvert = false;
        }
        this->solve(*currentGroup->systemMatrix, *currentGroup->systemLHVector, *currentGroup->systemRHVector);
        if (!currentGroup->solutionVecId.isNull())
        {
            //v_clear(currentGroup->solutionVecId);
            //multiVectorPeqBaseVector(currentGroup->solutionVecId, currentGroup->systemLHVector, &(currentGroup->matrixAccessor));
            executeVisitor( simulation::MechanicalMultiVectorFromBaseVectorVisitor(core::ExecParams::defaultInstance(), currentGroup->solutionVecId, currentGroup->systemLHVector, &(currentGroup->matrixAccessor)) );
        }
    }
}

/*

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::partialSolveSystem(VecIndex&  Iout, VecIndex&  Iin)
{
    if (needInvert)
    {
        this->invert(*systemMatrix);
        needInvert = false;
    }
    this->partial_solve(*systemMatrix, *systemLHVector, *systemRHVector, Iout, Iin);


	if (!solutionVecId.isNull())
    {
        unsigned int offset = 0;
        //MechanicalBaseVector2MultiVectorVisitor(systemLHVector, solutionVecId, offset).execute( getContext() );
        //MechanicalVOpVisitor(solutionVecId).execute( getContext() ); // clear solutionVecId
        //MechanicalMultiVectorPeqBaseVectorVisitor(solutionVecId, systemLHVector, offset).execute( getContext() );
        v_clear(solutionVecId);
        multiVectorPeqBaseVector(solutionVecId, systemLHVector, offset);
    }
}
*/

template<class Matrix, class Vector>
Vector* MatrixLinearSolver<Matrix,Vector>::createPersistentVector()
{
    return new Vector;
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::deletePersistentVector(Vector* v)
{
    delete v;
}

template<class Matrix, class Vector>
Matrix* MatrixLinearSolver<Matrix,Vector>::createMatrix()
{
    return new Matrix;
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::deleteMatrix(Matrix* v)
{
    delete v;
}

template<> SOFA_BASE_LINEAR_SOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::resetSystem();

template<> SOFA_BASE_LINEAR_SOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::resizeSystem(int);

template<> SOFA_BASE_LINEAR_SOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::setSystemMBKMatrix(const core::MechanicalParams* mparams);

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

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_LINEARSOLVER_MATRIXLINEARSOLVER_CPP)
extern template class SOFA_BASE_LINEAR_SOLVER_API MatrixLinearSolver< GraphScatteredMatrix, GraphScatteredVector, NoThreadManager >;
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
#endif

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
