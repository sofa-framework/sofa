/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_SIMULATION_MATRIXLINEARSOLVER_H
#define SOFA_SIMULATION_MATRIXLINEARSOLVER_H

#include <sofa/simulation/common/OdeSolverImpl.h>
#include <sofa/core/componentmodel/behavior/LinearSolver.h>

namespace sofa
{

namespace simulation
{


template<class Matrix, class Vector>
class MatrixLinearSolverInternalData
{
public:
    MatrixLinearSolverInternalData(core::objectmodel::BaseObject*)
    {}
};

template<class Matrix, class Vector>
class MatrixLinearSolver : public sofa::core::componentmodel::behavior::LinearSolver, public SolverImpl
{
public:
    typedef sofa::core::componentmodel::behavior::BaseMechanicalState::VecId VecId;

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

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const MatrixLinearSolver<Matrix,Vector>* = NULL)
    {
        return Matrix::Name();
    }


protected:

    virtual void invert(Matrix& /*M*/) {}

    virtual void solve(Matrix& M, Vector& solution, Vector& rh) = 0;

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
    if (systemMatrix) systemMatrix->clear();
    if (systemRHVector) systemRHVector->clear();
    if (systemLHVector) systemLHVector->clear();
    solutionVecId = VecId();
    needInvert = true;
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::resizeSystem(int n)
{
    if (!systemMatrix) systemMatrix = createMatrix();
    systemMatrix->resize(n,n);
    if (!systemRHVector) systemRHVector = createVector();
    systemRHVector->resize(n);
    if (!systemLHVector) systemLHVector = createVector();
    systemLHVector->resize(n);
    needInvert = true;
}

template<class Matrix, class Vector>
void MatrixLinearSolver<Matrix,Vector>::setSystemMBKMatrix(double mFact, double bFact, double kFact)
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

class GraphScatteredMatrix
{
protected:
    SolverImpl* parent;
    double mFact, bFact, kFact;
public:
    GraphScatteredMatrix(SolverImpl* p)
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
    //void papply(GraphScatteredVector& res, GraphScatteredVector& x);

    static const char* Name() { return "GraphScattered"; }
};


class GraphScatteredVector : public sofa::core::componentmodel::behavior::MultiVector<SolverImpl>
{
public:
    typedef sofa::core::componentmodel::behavior::MultiVector<SolverImpl> Inherit;
    GraphScatteredVector(SolverImpl* p, VecId id)
        : Inherit(p, id)
    {
    }
    GraphScatteredVector(SolverImpl* p, VecId::Type t = VecId::V_DERIV)
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

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::resetSystem();

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::resizeSystem(int);

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::setSystemMBKMatrix(double mFact, double bFact, double kFact);

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::setSystemRHVector(VecId v);

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::setSystemLHVector(VecId v);

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::solveSystem();

template<>
GraphScatteredVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::createVector();

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::deleteVector(GraphScatteredVector* v);

template<>
GraphScatteredMatrix* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::createMatrix();

template<>
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::deleteMatrix(GraphScatteredMatrix* v);

template<>
defaulttype::BaseMatrix* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::getSystemBaseMatrix();

template<>
defaulttype::BaseVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::getSystemRHBaseVector();

template<>
defaulttype::BaseVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector>::getSystemLHBaseVector();

} // namespace simulation

} // namespace sofa

#endif
