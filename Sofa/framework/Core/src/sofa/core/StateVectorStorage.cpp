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
#define SOFA_CORE_STATEVECTORSTORAGE_CPP
#include <sofa/core/StateVectorStorage.inl>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::core
{

template class SOFA_CORE_API StateVectorStorage<sofa::defaulttype::Vec3dTypes>;
template class SOFA_CORE_API StateVectorStorage<sofa::defaulttype::Vec2Types>;
template class SOFA_CORE_API StateVectorStorage<sofa::defaulttype::Vec1Types>;
template class SOFA_CORE_API StateVectorStorage<sofa::defaulttype::Vec6Types>;
template class SOFA_CORE_API StateVectorStorage<sofa::defaulttype::Rigid3Types>;
template class SOFA_CORE_API StateVectorStorage<sofa::defaulttype::Rigid2Types>;
template class SOFA_CORE_API StateVectorStorage<sofa::defaulttype::Vec3fTypes>;

}
