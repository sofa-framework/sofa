/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_COMPONENT_ENGINE_DifferenceEngine_CPP
#include "DifferenceEngine.inl"
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(DifferenceEngine)

int DifferenceEngineClass = core::RegisterObject("Computing the difference between two vector of dofs")
#ifndef SOFA_FLOAT
        .add< DifferenceEngine<Vec1d> >()
        .add< DifferenceEngine<Vec3d> >(true) // default template
#endif
#ifndef SOFA_DOUBLE
        .add< DifferenceEngine<Vec1f> >()
        .add< DifferenceEngine<Vec3f> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_GENERAL_ENGINE_API DifferenceEngine<Vec1d>;
template class SOFA_GENERAL_ENGINE_API DifferenceEngine<Vec3d>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_GENERAL_ENGINE_API DifferenceEngine<Vec1f>;
template class SOFA_GENERAL_ENGINE_API DifferenceEngine<Vec3f>;
#endif

} // namespace engine

} // namespace component

} // namespace sofa


