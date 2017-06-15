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
#include <sofa/helper/system/config.h>
#include <SofaBoundaryCondition/initBoundaryCondition.h>


namespace sofa
{

namespace component
{


void initBoundaryCondition()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

SOFA_LINK_CLASS(BuoyantForceField)
SOFA_LINK_CLASS(ConicalForceField)
SOFA_LINK_CLASS(ConstantForceField)
#ifdef TODOTOPO
SOFA_LINK_CLASS(EdgePressureForceField)
#endif
SOFA_LINK_CLASS(EllipsoidForceField)
SOFA_LINK_CLASS(LinearForceField)
#ifdef TODOTOPO
SOFA_LINK_CLASS(OscillatingTorsionPressureForceField)
#endif
SOFA_LINK_CLASS(PlaneForceField)
SOFA_LINK_CLASS(SphereForceField)
SOFA_LINK_CLASS(SurfacePressureForceField)
SOFA_LINK_CLASS(TaitSurfacePressureForceField)
#ifdef TODOTOPO
SOFA_LINK_CLASS(TrianglePressureForceField)
#endif
SOFA_LINK_CLASS(VaccumSphereForceField)
SOFA_LINK_CLASS(FixedConstraint)
SOFA_LINK_CLASS(FixedPlaneConstraint)
SOFA_LINK_CLASS(FixedRotationConstraint)
SOFA_LINK_CLASS(FixedTranslationConstraint)
SOFA_LINK_CLASS(HermiteSplineConstraint)
SOFA_LINK_CLASS(LinearMovementConstraint)
SOFA_LINK_CLASS(LinearVelocityConstraint)
SOFA_LINK_CLASS(OscillatorConstraint)
SOFA_LINK_CLASS(ParabolicConstraint)
SOFA_LINK_CLASS(PartialFixedConstraint)
SOFA_LINK_CLASS(PartialLinearMovementConstraint)
SOFA_LINK_CLASS(PositionBasedDynamicsConstraint)

} // namespace component

} // namespace sofa
