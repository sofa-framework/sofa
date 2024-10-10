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
#define SOFA_COMPONENT_FORCEFIELD_RESTSHAPESPRINGSFORCEFIELD_CPP

#include <sofa/component/solidmechanics/spring/RestShapeSpringsForceField.inl>

#include <sofa/helper/visual/DrawTool.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>


namespace sofa::component::solidmechanics::spring
{

using namespace sofa::defaulttype;

void registerRestShapeSpringsForceField(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Elastic springs generating forces on degrees of freedom between their current and rest shape position.")
        .add< RestShapeSpringsForceField<Vec3Types> >()
        .add< RestShapeSpringsForceField<Vec1Types> >()
        .add< RestShapeSpringsForceField<Rigid3Types> >());
}

template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API RestShapeSpringsForceField<Vec3Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API RestShapeSpringsForceField<Vec1Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API RestShapeSpringsForceField<Rigid3Types>;

} // namespace sofa::component::solidmechanics::spring
