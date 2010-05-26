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
#ifndef SOFA_COMPONENT_LINEARSOLVER_GRAPHSCATTEREDTYPES_H
#define SOFA_COMPONENT_LINEARSOLVER_GRAPHSCATTEREDTYPES_H

#include <sofa/simulation/common/SolverImpl.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/component.h>
#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/component/linearsolver/FullMatrix.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

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


class SOFA_COMPONENT_LINEARSOLVER_API GraphScatteredVector : public sofa::core::behavior::MultiVector<simulation::SolverImpl>
{
public:
    typedef sofa::core::behavior::MultiVector<simulation::SolverImpl> Inherit;
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

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
