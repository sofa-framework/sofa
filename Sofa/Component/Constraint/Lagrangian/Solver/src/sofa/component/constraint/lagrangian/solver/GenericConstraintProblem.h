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
#include <sofa/component/constraint/lagrangian/solver/config.h>

#include <sofa/component/constraint/lagrangian/solver/ConstraintSolverImpl.h>
#include <sofa/linearalgebra/SparseMatrix.h>

namespace sofa::component::constraint::lagrangian::solver
{

class GenericConstraintSolver;

class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_SOLVER_API GenericConstraintProblem : public ConstraintProblem, public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(GenericConstraintProblem, BaseObject);

    typedef std::vector< core::behavior::BaseConstraintCorrection* > ConstraintCorrections;

    GenericConstraintProblem()
    : scaleTolerance(true)
    , allVerified(false)
    , sor(1.0)
    , currentError(0.0)
    , currentIterations(0)
    {}

    ~GenericConstraintProblem() override
    {
        freeConstraintResolutions();
    }

    void clear(int nbConstraints) override;
    void freeConstraintResolutions();
    int getNumConstraints();
    int getNumConstraintGroups();
    void result_output(GenericConstraintSolver* solver, SReal *force, SReal error, int iterCount, bool convergence);
    void solveTimed(SReal tol, int maxIt, SReal timeout) override;

    virtual void buildSystem( const core::ConstraintParams *cParams, unsigned int numConstraints, GenericConstraintSolver* solver = nullptr) = 0;
    virtual void solve( SReal timeout = 0.0, GenericConstraintSolver* solver = nullptr) = 0;

    static void addRegularization(linearalgebra::BaseMatrix& W, const SReal regularization);

    sofa::linearalgebra::FullVector<SReal> _d; //
    std::vector<core::behavior::ConstraintResolution*> constraintsResolutions; //
    bool scaleTolerance, allVerified; //
    SReal sor; /** GAUSS-SEIDEL **/
    SReal currentError; //
    int currentIterations; //

protected:
    sofa::linearalgebra::FullVector<SReal> m_lam;
    sofa::linearalgebra::FullVector<SReal> m_deltaF;
    sofa::linearalgebra::FullVector<SReal> m_deltaF_new;
    sofa::linearalgebra::FullVector<SReal> m_p;

};
}
