/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#define INDEXVALUEMAPPER_CPP_

#include "IndexValueMapper.inl"
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace sofa;
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(IndexValueMapper)

int IndexValueMapperClass = core::RegisterObject("?")
#ifndef SOFA_FLOAT
        .add< IndexValueMapper<Vec3dTypes> >(true)
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< IndexValueMapper<Vec3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_GENERAL_ENGINE_API IndexValueMapper<Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_GENERAL_ENGINE_API IndexValueMapper<Vec3fTypes>;
#endif //SOFA_DOUBLE


} // namespace engine

} // namespace component

} // namespace sofa
