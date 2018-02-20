/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_ENGINE_DILATEENGINE_CPP
#define SOFA_COMPONENT_ENGINE_DILATEENGINE_CPP
#include <SofaGeneralEngine/DilateEngine.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace defaulttype;

SOFA_DECL_CLASS(DilateEngine)

int DilateEngineClass = core::RegisterObject("Dilates a given mesh by moving vertices along their normal.")
// TriangleOctree implemented with double only
//#ifdef SOFA_WITH_FLOAT
//.add< DilateEngine<Vec3fTypes>>(true)
//#endif //SOFA_FLOAT
#ifdef SOFA_WITH_DOUBLE
.add< DilateEngine<Vec3dTypes>>(true) // default template
#endif //SOFA_DOUBLE
        ;

// TriangleOctree implemented with double only
//#ifdef SOFA_WITH_FLOAT
//template class SOFA_GENERAL_ENGINE_API DilateEngine<Vec3fTypes>;
//#endif //SOFA_WITH_FLOAT
#ifdef SOFA_WITH_DOUBLE
template class SOFA_GENERAL_ENGINE_API DilateEngine<Vec3dTypes>;
#endif //SOFA_WITH_DOUBLE

} // namespace engine

} // namespace component

} // namespace sofa

#endif

