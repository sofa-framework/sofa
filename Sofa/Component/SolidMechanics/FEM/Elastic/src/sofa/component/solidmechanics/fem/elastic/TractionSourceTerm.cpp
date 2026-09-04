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
#define SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_TRACTION_SOURCE_TERM_CPP

#include <sofa/component/solidmechanics/fem/elastic/TractionSourceTerm.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/geometry/Quad.h>
#include <sofa/geometry/Triangle.h>

namespace sofa::component::solidmechanics::fem::elastic
{

void registerTractionSourceTerm(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Pressure load following the current-configuration normal direction, on a Triangle or Quad boundary")
        .add< TractionSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle> >()
        .add< TractionSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Quad> >()
    );
}

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API TractionSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API TractionSourceTerm<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;

}  // namespace sofa::component::solidmechanics::fem::elastic
