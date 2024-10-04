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
#define SOFA_COMPONENT_LINEARSOLVER_CHOLESKYSOLVER_CPP
#include <sofa/component/linearsolver/direct/CholeskySolver.inl>

#include <sofa/core/ObjectFactory.h>

namespace sofa::component::linearsolver::direct
{

using namespace sofa::defaulttype;
using namespace sofa::linearalgebra;

void registerCholeskySolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Direct linear solver based on Cholesky factorization, for dense matrices.")
        .add< CholeskySolver< SparseMatrix<SReal>, FullVector<SReal> > >(true)
        .add< CholeskySolver< FullMatrix<SReal>, FullVector<SReal> > >());
}

template class SOFA_COMPONENT_LINEARSOLVER_DIRECT_API CholeskySolver< SparseMatrix<SReal>, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_DIRECT_API CholeskySolver< FullMatrix<SReal>, FullVector<SReal> >;


} //namespace sofa::component::linearsolver::direct
