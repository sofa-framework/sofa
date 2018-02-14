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
#define SOFA_COMPONENT_ENGINE_MATHOP_CPP
#include <SofaGeneralEngine/MathOp.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

SOFA_DECL_CLASS(MathOp)

int MathOpClass = core::RegisterObject("Apply a math operation to combine several inputs")
#if defined(SOFA_DOUBLE)
    .add< MathOp< helper::vector<double> > >(true)
#elif defined(SOFA_FLOAT)
    .add< MathOp< helper::vector<float> > >(true)
#else
    .add< MathOp< helper::vector<double> > >(true)
    .add< MathOp< helper::vector<float> > >()
#endif
    .add< MathOp< helper::vector<int> > >()
    .add< MathOp< helper::vector<bool> > >()
#ifndef SOFA_FLOAT
    .add< MathOp< helper::vector<defaulttype::Vec2d> > >()
    .add< MathOp< helper::vector<defaulttype::Vec3d> > >()
    .add< MathOp< defaulttype::Rigid2dTypes::VecCoord > >()
    .add< MathOp< defaulttype::Rigid2dTypes::VecDeriv > >()
    .add< MathOp< defaulttype::Rigid3dTypes::VecCoord > >()
    .add< MathOp< defaulttype::Rigid3dTypes::VecDeriv > >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
    .add< MathOp< helper::vector<defaulttype::Vec2f> > >()
    .add< MathOp< helper::vector<defaulttype::Vec3f> > >()
    .add< MathOp< defaulttype::Rigid2fTypes::VecCoord > >()
    .add< MathOp< defaulttype::Rigid2fTypes::VecDeriv > >()
    .add< MathOp< defaulttype::Rigid3fTypes::VecCoord > >()
    .add< MathOp< defaulttype::Rigid3fTypes::VecDeriv > >()
#endif //SOFA_DOUBLE
        ;

template class SOFA_GENERAL_ENGINE_API MathOp< helper::vector<int> >;
template class SOFA_GENERAL_ENGINE_API MathOp< helper::vector<bool> >;

#ifndef SOFA_FLOAT
template class SOFA_GENERAL_ENGINE_API MathOp< helper::vector<double> >;
template class SOFA_GENERAL_ENGINE_API MathOp< helper::vector<defaulttype::Vec2d> >;
template class SOFA_GENERAL_ENGINE_API MathOp< helper::vector<defaulttype::Vec3d> >;
template class SOFA_GENERAL_ENGINE_API MathOp< defaulttype::Rigid2dTypes::VecCoord >;
template class SOFA_GENERAL_ENGINE_API MathOp< defaulttype::Rigid2dTypes::VecDeriv >;
template class SOFA_GENERAL_ENGINE_API MathOp< defaulttype::Rigid3dTypes::VecCoord >;
template class SOFA_GENERAL_ENGINE_API MathOp< defaulttype::Rigid3dTypes::VecDeriv >;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_GENERAL_ENGINE_API MathOp< helper::vector<float> >;
template class SOFA_GENERAL_ENGINE_API MathOp< helper::vector<defaulttype::Vec2f> >;
template class SOFA_GENERAL_ENGINE_API MathOp< helper::vector<defaulttype::Vec3f> >;
template class SOFA_GENERAL_ENGINE_API MathOp< defaulttype::Rigid2fTypes::VecCoord >;
template class SOFA_GENERAL_ENGINE_API MathOp< defaulttype::Rigid2fTypes::VecDeriv >;
template class SOFA_GENERAL_ENGINE_API MathOp< defaulttype::Rigid3fTypes::VecCoord >;
template class SOFA_GENERAL_ENGINE_API MathOp< defaulttype::Rigid3fTypes::VecDeriv >;
#endif //SOFA_DOUBLE


} // namespace constraint

} // namespace component

} // namespace sofa

