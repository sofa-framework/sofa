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
#define SOFA_COMPONENT_CONSTRAINT_LINEARSOLVERCONSTRAINTCORRECTION_CPP
#include "LinearSolverConstraintCorrection.inl"
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace constraintset
{
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(LinearSolverConstraintCorrection)

int LinearSolverContactCorrectionClass = core::RegisterObject("")
#ifndef SOFA_FLOAT
        .add< LinearSolverConstraintCorrection<Vec3dTypes> >()
        .add< LinearSolverConstraintCorrection<Vec2dTypes> >()
        .add< LinearSolverConstraintCorrection<Vec1dTypes> >()
        .add< LinearSolverConstraintCorrection<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< LinearSolverConstraintCorrection<Vec3fTypes> >()
        .add< LinearSolverConstraintCorrection<Vec2fTypes> >()
        .add< LinearSolverConstraintCorrection<Vec1fTypes> >()
        .add< LinearSolverConstraintCorrection<Rigid3fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class SOFA_CONSTRAINT_API LinearSolverConstraintCorrection<Vec3dTypes>;
template class SOFA_CONSTRAINT_API LinearSolverConstraintCorrection<Vec2dTypes>;
template class SOFA_CONSTRAINT_API LinearSolverConstraintCorrection<Vec1dTypes>;
template class SOFA_CONSTRAINT_API LinearSolverConstraintCorrection<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_CONSTRAINT_API LinearSolverConstraintCorrection<Vec3fTypes>;
template class SOFA_CONSTRAINT_API LinearSolverConstraintCorrection<Vec2fTypes>;
template class SOFA_CONSTRAINT_API LinearSolverConstraintCorrection<Vec1fTypes>;
template class SOFA_CONSTRAINT_API LinearSolverConstraintCorrection<Rigid3fTypes>;
#endif


} // namespace collision

} // namespace component

} // namespace sofa
