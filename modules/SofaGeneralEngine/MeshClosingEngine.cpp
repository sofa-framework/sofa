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
#define SOFA_COMPONENT_ENGINE_MeshClosingEngine_CPP
#include <SofaGeneralEngine/MeshClosingEngine.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

SOFA_DECL_CLASS(MeshClosingEngine)

int MeshClosingEngineClass = core::RegisterObject("Merge several meshes")
#ifdef SOFA_FLOAT
        .add< MeshClosingEngine<defaulttype::Vec3fTypes> >(true) // default template
#else
        .add< MeshClosingEngine<defaulttype::Vec3dTypes> >(true) // default template
#ifndef SOFA_DOUBLE
        .add< MeshClosingEngine<defaulttype::Vec3fTypes> >()
#endif
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_GENERAL_ENGINE_API MeshClosingEngine<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_GENERAL_ENGINE_API MeshClosingEngine<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE


} // namespace constraint

} // namespace component

} // namespace sofa

