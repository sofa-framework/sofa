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
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/component/mass/ElementFEMMass.inl>

namespace sofa::component::mass
{

template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
template class SOFA_COMPONENT_MASS_API ElementFEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;

void registerFEMMass(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Finite-element mass (inertia and body force) defined on edges")
        .add< ElementFEMMass<sofa::defaulttype::Vec1Types, sofa::geometry::Edge> >()
        .add< ElementFEMMass<sofa::defaulttype::Vec2Types, sofa::geometry::Edge> >()
        .add< ElementFEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Edge> >(true));

    factory->registerObjects(sofa::core::ObjectRegistrationData("Finite-element mass (inertia and body force) defined on triangles")
        .add< ElementFEMMass<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle> >()
        .add< ElementFEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle> >(true));

    factory->registerObjects(sofa::core::ObjectRegistrationData("Finite-element mass (inertia and body force) defined on quads")
        .add< ElementFEMMass<sofa::defaulttype::Vec2Types, sofa::geometry::Quad> >()
        .add< ElementFEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Quad> >(true));

    factory->registerObjects(sofa::core::ObjectRegistrationData("Finite-element mass (inertia and body force) defined on tetrahedra")
        .add< ElementFEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron> >(true));

    factory->registerObjects(sofa::core::ObjectRegistrationData("Finite-element mass (inertia and body force) defined on hexahedra")
        .add< ElementFEMMass<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron> >(true));

}

}
