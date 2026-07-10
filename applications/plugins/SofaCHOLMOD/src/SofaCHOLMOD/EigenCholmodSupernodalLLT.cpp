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
#define SOFACHOLMOD_EIGENCHOLMODSUPERNODALLLT_CPP

#include <SofaCHOLMOD/config.h>

#include <Eigen/CholmodSupport>
#include <SofaCHOLMOD/EigenCholmodSupernodalLLT.h>
#include <sofa/component/linearsolver/direct/EigenDirectSparseSolver.inl>

#include <sofa/core/ObjectFactory.h>

namespace sofacholmod
{

// CHOLMOD only supports double precision
static_assert(std::is_same_v<SReal, double>, "EigenCholmodSupernodalLLT requires double precision (SReal must be double)");

template class SOFACHOLMOD_API EigenCholmodSupernodalLLT< SReal >;
template class SOFACHOLMOD_API EigenCholmodSupernodalLLT< sofa::type::Mat<3,3,SReal> >;

MainCholmodSupernodalLLTFactory::~MainCholmodSupernodalLLTFactory() = default;

void registerEigenCholmodSupernodalLLT(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Direct linear solver using a supernodal sparse LL^T Cholesky factorization from CHOLMOD (SuiteSparse).")
        .add< EigenCholmodSupernodalLLT< SReal > >()
        .add< EigenCholmodSupernodalLLT< sofa::type::Mat<3, 3, SReal> > >());
}

} // namespace sofacholmod
