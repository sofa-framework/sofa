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
#define SOFA_COMPONENT_LINEARSOLVER_MATRIXFREESYSTEM_GRAPHSCATTERED_CPP
#include <sofa/component/linearsolver/iterative/MatrixFreeSystem[GraphScattered].h>
#include <sofa/component/linearsystem/CompositeLinearSystem.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::linearsystem
{

using sofa::component::linearsolver::GraphScatteredMatrix;
using sofa::component::linearsolver::GraphScatteredVector;

template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API MatrixFreeSystem<linearsolver::GraphScatteredMatrix, GraphScatteredVector>;
template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API CompositeLinearSystem<linearsolver::GraphScatteredMatrix, GraphScatteredVector>;

void registerMatrixFreeSystemGraphScattered(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Matrix-free (unbuilt) linear system.")
        .add< MatrixFreeSystem<GraphScatteredMatrix, GraphScatteredVector> >(true));
    factory->registerObjects(core::ObjectRegistrationData("Matrix-free (unbuilt) linear system.")
        .add< CompositeLinearSystem<GraphScatteredMatrix, GraphScatteredVector> >());
}

} //namespace sofa::component::linearsystem
