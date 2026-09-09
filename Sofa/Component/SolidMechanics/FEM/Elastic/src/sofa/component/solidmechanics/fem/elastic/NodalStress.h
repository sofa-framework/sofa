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
#pragma once

#include <sofa/component/solidmechanics/fem/elastic/config.h>
#include <sofa/core/BaseNodalProperty.h>
#include <sofa/core/trait/DataTypes.h>
#include <sofa/type/MatSym.h>
#include <sofa/type/Vec.h>

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_NODAL_STRESS_CPP)
#include <sofa/defaulttype/VecTypes.h>
#endif

namespace sofa::component::solidmechanics::fem::elastic
{

/// The independent components of a symmetric stress tensor, in the storage order of MatSym.
template <class DataTypes>
using StressComponents = sofa::type::Vec<
    sofa::type::NumberOfIndependentElements<DataTypes::spatial_dimensions>,
    sofa::Real_t<DataTypes>>;

/**
 * @class NodalStress
 * @brief A symmetric stress tensor prescribed at the nodes, one tensor per node.
 *
 * A tensor is written as its independent components in the storage order of MatSym, which is not
 * the standard Voigt one: xx xy yy xz yz zz in 3D, xx xy yy in 2D.
 *
 * @tparam TDataTypes The data types used for positions, velocities, etc. (e.g., Vec3Types).
 */
template <class TDataTypes>
class NodalStress : public sofa::core::BaseNodalProperty<StressComponents<TDataTypes>>
{
public:
    using DataTypes = TDataTypes;
    using Components = StressComponents<DataTypes>;

    SOFA_CLASS(SOFA_TEMPLATE(NodalStress, DataTypes),
        SOFA_TEMPLATE(sofa::core::BaseNodalProperty, StressComponents<DataTypes>));

protected:

    NodalStress() : sofa::core::BaseNodalProperty<Components>(Components{}) {}
};

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_NODAL_STRESS_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NodalStress<sofa::defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NodalStress<sofa::defaulttype::Vec3Types>;
#endif

}  // namespace sofa::component::solidmechanics::fem::elastic
