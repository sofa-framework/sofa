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
#ifndef SOFA_COMPONENT_LINEARSOLVER_MATRIXLINEARSOLVER_H
#define SOFA_COMPONENT_LINEARSOLVER_MATRIXLINEARSOLVER_H

#include <sofa/simulation/common/SolverImpl.h>
#include <sofa/core/componentmodel/behavior/LinearSolver.h>
#include <sofa/component/component.h>
#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/component/linearsolver/FullMatrix.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{


template<class Matrix, class Vector>
class MatrixLinearSolverInternalData
{
public:
    MatrixLinearSolverInternalData(core::objectmodel::BaseObject*)
    {}
};

template<class Matrix, class Vector>
class SOFA_EXPORT_DYNAMIC_LIBRARY MatrixLinearSolver : public sofa::core::componentmodel::behavior::LinearSolver, public sofa::simulation::SolverImpl
{
public:
    typedef sofa::core::componentmodel::behavior::BaseMechanicalState::VecId VecId;
    typedef  std::list<int> ListIndex;

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
    void setSystemMBKMatrix(double mFact=0.0, double bFact=0.0, double kFact=0.0);

    /// Set the linear system right-hand term vector, from the values contained in the (Mechanical/Physical)State objects
    void setSystemRHVector(VecId v);

    /// Set the initial estimate of the linear system left-hand term vector, from the values contained in the (Mechanical/Physical)State objects
    /// This vector will be replaced by the solution of the system once solveSystem is called
    void setSystemLHVector(VecId v);

    /// Get the linear system matrix, or NULL if this solver does not build it
    Matrix* getSystemMatrix() { return systemMatrix; }

    /// Get the linear system right-hand term vector, or NULL if this solver does not build it
    Vector* getSystemRHVector() { return systemRHVector; }

    /// Get the linear system left-hand term vector, or NULL if this solver does not build it
    Vector* getSystemLHVector() { return systemLHVector; }

    /// Get the linear system matrix, or NULL if this solver does not build it
    defaulttype::BaseMatrix* getSystemBaseMatrix() { return systemMatrix; }

    /// Get the linear system right-hand term vector, or NULL if this solver does not build it
    defaulttype::BaseVector* getSystemRHBaseVector() { return systemRHVector; }

    /// Get the linear system left-hand term vector, or NULL if this solver does not build it
    defaulttype::BaseVector* getSystemLHBaseVector() { return systemLHVector; }

    /// Solve the system as constructed using the previous methods
    virtual void solveSystem();

    /// Invert the system, this method is optional because it's call when solveSystem() is called for the first time
    virtual void invertSystem()
    {
        if (needInvert)
        {
            this->invert(*systemMatrix);
            needInvert = false;
        }
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const MatrixLinearSolver<Matrix,Vector>* = NULL)
    {
        return Matrix::Name();
    }


    virtual void invert(Matrix& /*M*/) {}

    virtual void solve(Matrix& M, Vector& solution, Vector& rh) = 0;


    /// Default implementation of Multiply the inverse of the system matrix by the transpose of the given matrix, and multiply the result with the given matrix J
    ///
    /// TODO : put this implementation in MatrixLinearSolver class - fix problems mith Scattered Matrix

    template<class RMatrix, class JMatrix>
    bool addJMInvJt(RMatrix& result, JMatrix& J, double fact)
    {
        const unsigned int Jrows = J.rowSize();
        const unsigned int Jcols = J.colSize();
        if (Jcols != this->systemMatrix->rowSize())
        {
            serr << "LULinearSolver::addJMInvJt ERROR: incompatible J matrix size." << sendl;
            return false;
        }

        if (!Jrows) return false;

        const typename JMatrix::LineConstIterator jitend = J.end();
        // STEP 1 : put each line of matrix Jt in the right hand term of the system
        for (typename JMatrix::LineConstIterator jit1 = J.begin(); jit1 != jitend; ++jit1)
        {
            int row1 = jit1->first;
            // clear the right hand term:
            this->systemRHVector->clear(); // this->systemMatrix->rowSize()
            //double acc = 0.0;
            for (typename JMatrix::LElementConstIterator i1 = jit1->second.begin(), i1end = jit1->second.end(); i1 != i1end; ++i1)
            {
                this->systemRHVector->add(i1->first,i1->second);
            }

            // STEP 2 : solve the system :
            solveSystem();


            // STEP 3 : project the result using matrix J
            for (typename JMatrix::LineConstIterator jit2 = jit1; jit2 != jitend; ++jit2)
            {
                int row2 = jit2->first;
                double acc = 0.0;
                for (typename JMatrix::LElementConstIterator i2 = jit2->second.begin(), i2end = jit2->second.end(); i2 != i2end; ++i2)
                {
                    int col2 = i2->first;
                    double val2 = i2->second;
                    acc += val2 * this->systemLHVector->element(col2);
                }
                acc *= fact;
                //sout << "W("<<row1<<","<<row2<<") += "<<acc<<" * "<<fact<<sendl;
                result.add(row2,row1,acc);
                if (row1!=row2)
                    result.add(row1,row2,acc);
            }
        }
        return true;
    }





    /// Default implementation of Multiply the inverse of the system matrix by the transpose of the given matrix, and multiply the result with the given matrix J
    ///
    /// @param result the variable where the result will be added
    /// @param J the matrix J to use
    /// @return false if the solver does not support this operation, of it the system matrix is not invertible
    bool addJMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact)
    {
        if (FullMatrix<double>* r = dynamic_cast<FullMatrix<double>*>(result))
        {
            if (SparseMatrix<double>* j = dynamic_cast<SparseMatrix<double>*>(J))
            {
                return addJMInvJt(*r,*j,fact);
            }
            else if (SparseMatrix<float>* j = dynamic_cast<SparseMatrix<float>*>(J))
            {
                return addJMInvJt(*r,*j,fact);
            }
        }
        else if (FullMatrix<double>* r = dynamic_cast<FullMatrix<double>*>(result))
        {
            if (SparseMatrix<double>* j = dynamic_cast<SparseMatrix<double>*>(J))
            {
                return addJMInvJt(*r,*j,fact);
            }
            else if (SparseMatrix<float>* j = dynamic_cast<SparseMatrix<float>*>(J))
            {
                return addJMInvJt(*r,*j,fact);
            }
        }
        else if (defaulttype::BaseMatrix* r = result)
        {
            if (SparseMatrix<double>* j = dynamic_cast<SparseMatrix<double>*>(J))
            {
                return addJMInvJt(*r,*j,fact);
            }
            else if (SparseMatrix<float>* j = dynamic_cast<SparseMatrix<float>*>(J))
            {
                return addJMInvJt(*r,*j,fact);
            }
        }
        return false;
    }


protected:

    /// newPartially solve the system
    virtual void partial_solve(Matrix& /*M*/, Vector& /*partial_solution*/, Vector& /*sparse_rh*/, ListIndex& /* indices_solution*/, ListIndex& /* indices input */) {}

    Vector* createVector();
    void deleteVector(Vector* v);

    Matrix* createMatrix();
    void deleteMatrix(Matrix* v);

    MatrixLinearSolverInternalData<Matrix,Vector>* data;

    bool needInvert;
    Matrix* systemMatrix;
    Vector* systemRHVector;
    Vector* systemLHVector;
    VecId solutionVecId;
};

template<class Matrix, class Vector>
MatrixLinearSolver<Matrix,Vector>::MatrixLinearSolver()
    : needInvert(true), systemMatrix(NULL), systemRHVector(NULL), systemLHVector(NULL)
{
    data = new MatrixLinearSolverInternalData<Matrix,Vector>(this);
}

template<class Matrix, class Vector>
MatrixLinearSolver<Matrix,Vector>::~MatrixLinearSolver()
{
    if (systemMatrix) deleteMatrix(systemMatrix);
    if (systemRHVector) deleteVector(systemRHVector);
    if (systemLHVector) deleteVector(systemLHVector);
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::resetSystem()
{
    if (!frozen)
    {
        if (systemMatrix) systemMatrix->clear();
        needInvert = true;
    }
    if (systemRHVector) systemRHVector->clear();
    if (systemLHVector) systemLHVector->clear();
    solutionVecId = VecId();
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::resizeSystem(int n)
{
    if (!frozen)
    {
        if (!systemMatrix) systemMatrix = createMatrix();
        systemMatrix->resize(n,n);
    }
    if (!systemRHVector) systemRHVector = createVector();
    systemRHVector->resize(n);
    if (!systemLHVector) systemLHVector = createVector();
    systemLHVector->resize(n);
    needInvert = true;
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::setSystemMBKMatrix(double mFact, double bFact, double kFact)
{
    if (!this->frozen)
    {
        unsigned int nbRow=0, nbCol=0;
        //MechanicalGetMatrixDimensionVisitor(nbRow, nbCol).execute( getContext() );
        this->getMatrixDimension(&nbRow, &nbCol);
        resizeSystem(nbRow);
        systemMatrix->clear();
        unsigned int offset = 0;
        //MechanicalAddMBK_ToMatrixVisitor(systemMatrix, mFact, bFact, kFact, offset).execute( getContext() );
        this->addMBK_ToMatrix(systemMatrix, mFact, bFact, kFact, offset);
    }
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::setSystemRHVector(VecId v)
{
    unsigned int offset = 0;
    //MechanicalMultiVector2BaseVectorVisitor(v, systemRHVector, offset).execute( getContext() );
    this->multiVector2BaseVector(v, systemRHVector, offset);
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::setSystemLHVector(VecId v)
{
    solutionVecId = v;
    unsigned int offset = 0;
    //MechanicalMultiVector2BaseVectorVisitor(v, systemLHVector, offset).execute( getContext() );
    this->multiVector2BaseVector(v, systemLHVector, offset);
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::solveSystem()
{
    if (needInvert)
    {
        this->invert(*systemMatrix);
        needInvert = false;
    }
    this->solve(*systemMatrix, *systemLHVector, *systemRHVector);
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
Vector* MatrixLinearSolver<Matrix,Vector>::createVector()
{
    return new Vector;
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::deleteVector(Vector* v)
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


class GraphScatteredMatrix;
class GraphScatteredVector;
template <class T1, class T2>
class MultExpr
{
public:
    T1& a;
    T2& b;
    MultExpr(T1& a, T2& b) : a(a), b(b) {}
};

class SOFA_COMPONENT_LINEARSOLVER_API GraphScatteredMatrix
{
protected:
    simulation::SolverImpl* parent;
    double mFact, bFact, kFact;
public:
    GraphScatteredMatrix(simulation::SolverImpl* p)
        : parent(p), mFact(0.0), bFact(0.0), kFact(0.0)
    {
    }
    void setMBKFacts(double m, double b, double k)
    {
        mFact = m;
        bFact = b;
        kFact = k;
    }
    MultExpr<GraphScatteredMatrix,GraphScatteredVector> operator*(GraphScatteredVector& v)
    {
        return MultExpr<GraphScatteredMatrix,GraphScatteredVector>(*this, v);
    }
    void apply(GraphScatteredVector& res, GraphScatteredVector& x);

    // compatibility with baseMatrix
    unsigned int rowSize()
    {
        unsigned int nbRow=0, nbCol=0;
        this->parent->getMatrixDimension(&nbRow, &nbCol);
        return nbRow;

    }
    int colSize()
    {
        unsigned int nbRow=0, nbCol=0;
        this->parent->getMatrixDimension(&nbRow, &nbCol);
        return nbCol;
    }

    //void papply(GraphScatteredVector& res, GraphScatteredVector& x);

    static const char* Name() { return "GraphScattered"; }
};


class SOFA_COMPONENT_LINEARSOLVER_API GraphScatteredVector : public sofa::core::componentmodel::behavior::MultiVector<simulation::SolverImpl>
{
public:
    typedef sofa::core::componentmodel::behavior::MultiVector<simulation::SolverImpl> Inherit;
    GraphScatteredVector(simulation::SolverImpl* p, VecId id)
        : Inherit(p, id)
    {
    }
    GraphScatteredVector(simulation::SolverImpl* p, VecId::Type t = VecId::V_DERIV)
        : Inherit(p, t)
    {
    }
    void set(VecId id)
    {
        this->v = id;
    }
    void reset()
    {
        this->v = VecId();
    }

    // for compatibility with base vector
    void clear()
    {
        reset();
    }

    /// TO IMPLEMENT
    void add(int /*row*/, SReal /*v*/)
    {
        std::cerr<<"WARNING : add an element is not supported in MultiVector"<<std::endl;
    }

    SReal element(int /*i*/)
    {
        std::cerr<<"WARNING : get a single element is not supported in MultiVector"<<std::endl;
        return 0;
    }



    friend class GraphScatteredMatrix;

    void operator=(const MultExpr<GraphScatteredMatrix,GraphScatteredVector>& expr)
    {
        expr.a.apply(*this,expr.b);
    }

    //void operator+=(const MultExpr<GraphScatteredMatrix,GraphScatteredVector>& expr)
    //{
    //    expr.a.papply(*this,expr.b);
    //}

    static const char* Name() { return "GraphScattered"; }
};

#if defined(WIN32) && !defined(SOFA_COMPONENT_LINEARSOLVER_MATRIXLINEARSOLVER_CPP)
extern template class SOFA_COMPONENT_LINEARSOLVER_API MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>;
#endif

template<> SOFA_COMPONENT_LINEARSOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::resetSystem();

template<> SOFA_COMPONENT_LINEARSOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::resizeSystem(int);

template<> SOFA_COMPONENT_LINEARSOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::setSystemMBKMatrix(double mFact, double bFact, double kFact);

template<> SOFA_COMPONENT_LINEARSOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::setSystemRHVector(VecId v);

template<> SOFA_COMPONENT_LINEARSOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::setSystemLHVector(VecId v);

template<> SOFA_COMPONENT_LINEARSOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::solveSystem();

template<> SOFA_COMPONENT_LINEARSOLVER_API
GraphScatteredVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::createVector();

template<> SOFA_COMPONENT_LINEARSOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::deleteVector(GraphScatteredVector* v);

template<> SOFA_COMPONENT_LINEARSOLVER_API
GraphScatteredMatrix* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::createMatrix();

template<> SOFA_COMPONENT_LINEARSOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::deleteMatrix(GraphScatteredMatrix* v);

template<> SOFA_COMPONENT_LINEARSOLVER_API
defaulttype::BaseMatrix* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::getSystemBaseMatrix();

template<> SOFA_COMPONENT_LINEARSOLVER_API
defaulttype::BaseVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::getSystemRHBaseVector();

template<> SOFA_COMPONENT_LINEARSOLVER_API
defaulttype::BaseVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::getSystemLHBaseVector();

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
