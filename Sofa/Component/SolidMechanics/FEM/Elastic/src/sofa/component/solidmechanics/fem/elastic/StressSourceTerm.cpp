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
#define SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_STRESS_SOURCE_TERM_CPP

#include <sofa/component/solidmechanics/fem/elastic/StressSourceTerm.inl>

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/fem/FiniteElement[all].h>

namespace sofa::component::solidmechanics::fem::elastic
{

using sofa::defaulttype::Vec2Types;
using sofa::defaulttype::Vec3Types;

void registerNodalStress(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Definition of a nodal stress tensor (one tensor per dof).")
        .add< NodalStress<Vec2Types> >()
        .add< NodalStress<Vec3Types> >()
    );
}

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NodalStress<Vec2Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NodalStress<Vec3Types>;

void registerNodalSymmetricStress(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Definition of a nodal symmetric stress tensor (one tensor per dof).")
        .add< NodalSymmetricStress<Vec2Types> >()
        .add< NodalSymmetricStress<Vec3Types> >()
    );
}

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NodalSymmetricStress<Vec2Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NodalSymmetricStress<Vec3Types>;

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseStressSourceTerm<Vec2Types, sofa::geometry::Edge, NodalStress<Vec2Types>>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseStressSourceTerm<Vec3Types, sofa::geometry::Triangle, NodalStress<Vec3Types>>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseStressSourceTerm<Vec3Types, sofa::geometry::Quad, NodalStress<Vec3Types>>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseStressSourceTerm<Vec2Types, sofa::geometry::Edge, NodalSymmetricStress<Vec2Types>>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseStressSourceTerm<Vec3Types, sofa::geometry::Triangle, NodalSymmetricStress<Vec3Types>>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseStressSourceTerm<Vec3Types, sofa::geometry::Quad, NodalSymmetricStress<Vec3Types>>;

void registerStressSourceTerm(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Traction obtained from a stress tensor prescribed at the nodes, contracted with the normal of the element")
        .add< StressSourceTerm<Vec2Types, sofa::geometry::Edge> >()
        .add< StressSourceTerm<Vec3Types, sofa::geometry::Triangle> >()
        .add< StressSourceTerm<Vec3Types, sofa::geometry::Quad> >()
    );
}

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API StressSourceTerm<Vec2Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API StressSourceTerm<Vec3Types, sofa::geometry::Triangle>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API StressSourceTerm<Vec3Types, sofa::geometry::Quad>;

void registerSymmetricStressSourceTerm(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Traction obtained from a symmetric stress tensor prescribed at the nodes, contracted with the normal of the element")
        .add< SymmetricStressSourceTerm<Vec2Types, sofa::geometry::Edge> >()
        .add< SymmetricStressSourceTerm<Vec3Types, sofa::geometry::Triangle> >()
        .add< SymmetricStressSourceTerm<Vec3Types, sofa::geometry::Quad> >()
    );
}

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API SymmetricStressSourceTerm<Vec2Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API SymmetricStressSourceTerm<Vec3Types, sofa::geometry::Triangle>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API SymmetricStressSourceTerm<Vec3Types, sofa::geometry::Quad>;

}  // namespace sofa::component::solidmechanics::fem::elastic
