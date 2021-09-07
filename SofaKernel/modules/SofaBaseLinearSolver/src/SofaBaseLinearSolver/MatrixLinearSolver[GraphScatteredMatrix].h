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

#include <sofa/simulation/VectorOperations.h>
#include <sofa/simulation/MechanicalOperations.h>

#include <SofaBaseLinearSolver/MatrixLinearSolver[_].h>
#include <SofaBaseLinearSolver/GraphScatteredTypes.h>

namespace sofa::component::linearsolver
{
//////////////////////////////////////////////////////////////
///Specialization for GraphScatteredTypes
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

template<> SOFA_SOFABASELINEARSOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::resetSystem();

template<> SOFA_SOFABASELINEARSOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::resizeSystem(Size);

template<> SOFA_SOFABASELINEARSOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::setSystemMBKMatrix(const core::MechanicalParams* mparams);

template<> SOFA_SOFABASELINEARSOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::rebuildSystem(double massFactor, double forceFactor);

template<> SOFA_SOFABASELINEARSOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::setSystemRHVector(core::MultiVecDerivId v);

template<> SOFA_SOFABASELINEARSOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::setSystemLHVector(core::MultiVecDerivId v);

template<> SOFA_SOFABASELINEARSOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::solveSystem();

template<> SOFA_SOFABASELINEARSOLVER_API
GraphScatteredVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::createPersistentVector();

template<> SOFA_SOFABASELINEARSOLVER_API
defaulttype::BaseMatrix* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::getSystemBaseMatrix();

template<> SOFA_SOFABASELINEARSOLVER_API
const core::behavior::MultiMatrixAccessor* MatrixLinearSolver<GraphScatteredMatrix, GraphScatteredVector, NoThreadManager>::getSystemMultiMatrixAccessor() const;

template<> SOFA_SOFABASELINEARSOLVER_API
defaulttype::BaseVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::getSystemRHBaseVector();

template<> SOFA_SOFABASELINEARSOLVER_API
defaulttype::BaseVector* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::getSystemLHBaseVector();

template<> SOFA_SOFABASELINEARSOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::setSystemMatrix(GraphScatteredMatrix * matrix);

template<> SOFA_SOFABASELINEARSOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::applyConstraintForce(const sofa::core::ConstraintParams* /*cparams*/, sofa::core::MultiVecDerivId /*dx*/, const defaulttype::BaseVector* /*f*/);

template<> SOFA_SOFABASELINEARSOLVER_API
void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::computeResidual(const core::ExecParams* params,defaulttype::BaseVector* f);

#if !defined(SOFA_COMPONENT_LINEARSOLVER_MATRIXLINEARSOLVER_GRAPHSCATTERREDMATRIX_CPP)
extern template class SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver< GraphScatteredMatrix, GraphScatteredVector, NoThreadManager >;
/// Extern template declarations don't prevent implicit instanciation in the case
/// of explicitely specialized classes.  (See section 14.3.7 of the C++ standard
/// [temp.expl.spec]). We have to declare non-specialized member functions by
/// hand to prevent MSVC from complaining that it doesn't find their definition.
extern template SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::MatrixLinearSolver();
extern template SOFA_SOFABASELINEARSOLVER_API MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::~MatrixLinearSolver();
extern template SOFA_SOFABASELINEARSOLVER_API void MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::invertSystem();
extern template SOFA_SOFABASELINEARSOLVER_API bool MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::addJMInvJt(defaulttype::BaseMatrix*, defaulttype::BaseMatrix*, double);
extern template SOFA_SOFABASELINEARSOLVER_API bool MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::addMInvJt(defaulttype::BaseMatrix*, defaulttype::BaseMatrix*, double);
extern template SOFA_SOFABASELINEARSOLVER_API bool MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::addJMInvJtLocal(GraphScatteredMatrix*, ResMatrixType*, const JMatrixType*, double);
extern template SOFA_SOFABASELINEARSOLVER_API bool MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::addMInvJtLocal(GraphScatteredMatrix*, ResMatrixType*, const  JMatrixType*, double);
extern template SOFA_SOFABASELINEARSOLVER_API bool MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::buildComplianceMatrix(const core::ConstraintParams*, defaulttype::BaseMatrix*, double);
extern template SOFA_SOFABASELINEARSOLVER_API MatrixInvertData* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::getMatrixInvertData(defaulttype::BaseMatrix * m);
extern template SOFA_SOFABASELINEARSOLVER_API MatrixInvertData* MatrixLinearSolver<GraphScatteredMatrix,GraphScatteredVector,NoThreadManager>::createInvertData();
#endif


} // namespace sofa::component::linearsolver
