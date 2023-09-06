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
#define SOFA_MULTITHREADING_PARALLELCGLINEARSOLVER_CPP
#include <MultiThreading/component/linearsolver/iterative/ParallelCGLinearSolver.h>
#include <MultiThreading/config.h>
#include <sofa/component/linearsolver/iterative/CGLinearSolver.inl>
#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.inl>
#include <sofa/component/linearsystem/MatrixLinearSystem.inl>

namespace multithreading::component::linearsolver::iterative
{

template class SOFA_MULTITHREADING_PLUGIN_API ParallelCompressedRowSparseMatrixMechanical<SReal>;

template class SOFA_MULTITHREADING_PLUGIN_API
sofa::component::linearsystem::TypedMatrixLinearSystem< ParallelCompressedRowSparseMatrixMechanical<SReal>, sofa::linearalgebra::FullVector<SReal> >;
template class SOFA_MULTITHREADING_PLUGIN_API
sofa::component::linearsystem::MatrixLinearSystem< ParallelCompressedRowSparseMatrixMechanical<SReal>, sofa::linearalgebra::FullVector<SReal> >;
template class SOFA_MULTITHREADING_PLUGIN_API
sofa::component::linearsolver::MatrixLinearSolver< ParallelCompressedRowSparseMatrixMechanical<SReal>, sofa::linearalgebra::FullVector<SReal> >;
template class SOFA_MULTITHREADING_PLUGIN_API
sofa::component::linearsolver::iterative::CGLinearSolver< ParallelCompressedRowSparseMatrixMechanical<SReal>, sofa::linearalgebra::FullVector<SReal> >;


}
