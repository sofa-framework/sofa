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

#include <sofa/simulation/fwd.h>
#include <sofa/core/behavior/MultiVec.h>
#include <sofa/core/MechanicalParams.h>

namespace sofa::component::linearsolver
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

class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API GraphScatteredMatrix
{
public:
    typedef SReal Real;
    core::MechanicalParams mparams;
    simulation::common::MechanicalOperations* parent;
public:
    GraphScatteredMatrix()
        : parent(nullptr)
    {
    }
    void setMBKFacts(const core::MechanicalParams* mparams){ this->mparams = *mparams;}
    MultExpr<GraphScatteredMatrix,GraphScatteredVector> operator*(GraphScatteredVector& v)
    {
        return MultExpr<GraphScatteredMatrix,GraphScatteredVector>(*this, v);
    }
    void apply(GraphScatteredVector& res, GraphScatteredVector& x);


    // compatibility with baseMatrix
    sofa::Size rowSize(); /// provides the number of rows of the Graph Scattered Matrix

    sofa::Size colSize();  /// provides the number of columns of the Graph Scattered Matrix

    static const char* Name() { return "GraphScattered"; }
};

class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API GraphScatteredVector : public sofa::core::behavior::MultiVecDeriv
{
public:
    typedef sofa::core::behavior::MultiVecDeriv Inherit;
    typedef SReal Real;
    GraphScatteredVector(core::behavior::BaseVectorOperations* p, core::VecDerivId id)
        : Inherit(p, id)
    {
    }
    GraphScatteredVector(core::behavior::BaseVectorOperations* p)
        : Inherit(p)
    {
    }
    void set(core::MultiVecDerivId id)
    {
        this->v = id;
    }
    void reset()
    {
        this->v = core::VecDerivId::null();
    }


    /// TO IMPLEMENT
    void add(int /*row*/, SReal /*v*/)
    {
        msg_warning("GraphScatterredType")<<"add an element is not supported in MultiVector";
    }

    /// TO IMPLEMENT
    void set(int /*row*/, SReal /*v*/)
    {
        msg_warning("GraphScatterredType")<<"set an element is not supported in MultiVector";
    }

    SReal element(int /*i*/)
    {
        msg_info("GraphScatterredType")<<"get a single element is not supported in MultiVector";
        return 0;
    }

    void resize( int ){
        msg_info("GraphScatterredType")<<"resize is not supported in MultiVector";
        assert(false);
    }

    friend class GraphScatteredMatrix;

    void operator=(const MultExpr<GraphScatteredMatrix,GraphScatteredVector>& expr);



    static const char* Name() { return "GraphScattered"; }
};


} // namespace sofa::component::linearsolver
