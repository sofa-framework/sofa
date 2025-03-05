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
#define SOFA_COMPONENT_FORCEFIELD_TAITSURFACEPRESSUREFORCEFIELD_CPP
#include <sofa/component/mechanicalload/TaitSurfacePressureForceField.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::mechanicalload
{

using namespace sofa::defaulttype;

void registerTaitSurfacePressureForceField(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("\
This component computes the volume enclosed by a surface mesh \
and apply a pressure force following Tait's equation: $P = P_0 - B((V/V_0)^\\gamma - 1)$.\n\
This ForceField can be used to apply :\n\
 * a constant pressure (set $B=0$ and use $P_0$)\n\
 * an ideal gas pressure (set $\\gamma=1$ and use $B$)\n\
 * a pressure from water (set $\\gamma=7$ and use $B$)")
        .add< TaitSurfacePressureForceField<Vec3Types> >());
}
template class SOFA_COMPONENT_MECHANICALLOAD_API TaitSurfacePressureForceField<Vec3Types>;

} // namespace sofa::component::mechanicalload
