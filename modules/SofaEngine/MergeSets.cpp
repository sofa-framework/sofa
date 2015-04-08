/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#define SOFA_COMPONENT_ENGINE_MERGESETS_CPP
#include <SofaEngine/MergeSets.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

SOFA_DECL_CLASS(MergeSets)

int MergeSetsClass = core::RegisterObject("Merge two sets of indices using specified boolean operation")
        .add< MergeSets<int> >(true)
        .add< MergeSets<unsigned int> >()
//.add< MergeSets<long long> >()
//.add< MergeSets<unsigned long long> >()
        ;

template class SOFA_ENGINE_API MergeSets<int>;
template class SOFA_ENGINE_API MergeSets<unsigned int>;
//template class SOFA_ENGINE_API MergeSets<long long>;
//template class SOFA_ENGINE_API MergeSets<unsigned long long>;

} // namespace constraint

} // namespace component

} // namespace sofa
