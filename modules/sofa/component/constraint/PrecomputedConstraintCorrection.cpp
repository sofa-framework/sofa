/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include "PrecomputedConstraintCorrection.inl"
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace constraint
{
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(PrecomputedConstraintCorrection)

int ContactCorrectionClass = core::RegisterObject("Component computing contact forces within a simulated body using the compliance method.")
#ifndef SOFA_FLOAT
        .add< PrecomputedConstraintCorrection<Vec3dTypes> >()
//     .add< PrecomputedConstraintCorrection<Vec2dTypes> >()
        .add< PrecomputedConstraintCorrection<Vec1dTypes> >()
        .add< PrecomputedConstraintCorrection<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< PrecomputedConstraintCorrection<Vec3fTypes> >()
//     .add< PrecomputedConstraintCorrection<Vec2fTypes> >()
        .add< PrecomputedConstraintCorrection<Vec1fTypes> >()
        .add< PrecomputedConstraintCorrection<Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class PrecomputedConstraintCorrection<Vec3dTypes>;
//     template class PrecomputedConstraintCorrection<Vec2dTypes>;
template class PrecomputedConstraintCorrection<Vec1dTypes>;
template class PrecomputedConstraintCorrection<Rigid3dTypes>;
//     template class PrecomputedConstraintCorrection<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class PrecomputedConstraintCorrection<Vec3fTypes>;
//     template class PrecomputedConstraintCorrection<Vec2fTypes>;
template class PrecomputedConstraintCorrection<Vec1fTypes>;
template class PrecomputedConstraintCorrection<Rigid3fTypes>;
//     template class PrecomputedConstraintCorrection<Rigid2fTypes>;
#endif


} // namespace collision

} // namespace component

} // namespace sofa
