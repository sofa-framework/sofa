/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define SOFA_COMPONENT_ENGINE_NORMENGINE_CPP
#include <SofaEngine/NormEngine.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

SOFA_DECL_CLASS(NormEngine)

int NormEngineClass = core::RegisterObject("Convert Vec in Real")
#ifndef SOFA_FLOAT
        .add< NormEngine<defaulttype::Vec3d> >(true) // default template
#endif
#ifndef SOFA_DOUBLE
        .add< NormEngine<defaulttype::Vec3f> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_ENGINE_API NormEngine<defaulttype::Vec3d>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_ENGINE_API NormEngine<defaulttype::Vec3f>;
#endif

} // namespace engine

} // namespace component

} // namespace sofa


