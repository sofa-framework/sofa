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
#define SOFA_COMPONENT_ENGINE_SHAPEMATCHING_CPP

#include <SofaGeneralEngine/ShapeMatching.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace engine
{

SOFA_DECL_CLASS(ShapeMatching)

using namespace defaulttype;

int ShapeMatchingClass = core::RegisterObject("Compute target positions using shape matching deformation method by Mueller et al.")
#ifndef SOFA_FLOAT
        .add< ShapeMatching<Vec3dTypes> >()
        .add< ShapeMatching<Rigid3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< ShapeMatching<Vec3fTypes> >()
        .add< ShapeMatching<Rigid3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_GENERAL_ENGINE_API ShapeMatching<Vec3dTypes>;
template class SOFA_GENERAL_ENGINE_API ShapeMatching<Rigid3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_GENERAL_ENGINE_API ShapeMatching<Vec3fTypes>;
template class SOFA_GENERAL_ENGINE_API ShapeMatching<Rigid3fTypes>;
#endif //SOFA_DOUBLE


// specialization for rigids

#ifndef SOFA_FLOAT
template <>
void ShapeMatching<Rigid3dTypes>::update()
{
    // TO DO: shape matching for rigids as in [Muller11]
}

#endif

#ifndef SOFA_DOUBLE
template <>
void ShapeMatching<Rigid3fTypes>::update()
{
    // TO DO: shape matching for rigids as in [Muller11]
}

#endif


} //
} // namespace component

} // namespace sofa

