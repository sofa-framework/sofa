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

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_NODAL_SOURCE_DENSITY_CPP)
#include <sofa/defaulttype/VecTypes.h>
#endif

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * @class NodalSourceDensity
 * @brief A source density prescribed at the nodes, one vector per node.
 *
 * @tparam TDataTypes The data types used for positions, velocities, etc. (e.g., Vec3Types).
 */
template <class TDataTypes>
class NodalSourceDensity : public sofa::core::BaseNodalProperty<sofa::Deriv_t<TDataTypes>>
{
public:
    using DataTypes = TDataTypes;
    using Deriv = sofa::Deriv_t<DataTypes>;

    SOFA_CLASS(SOFA_TEMPLATE(NodalSourceDensity, DataTypes),
        SOFA_TEMPLATE(sofa::core::BaseNodalProperty, sofa::Deriv_t<DataTypes>));

protected:

    NodalSourceDensity() : sofa::core::BaseNodalProperty<Deriv>(Deriv{}) {}
};

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_NODAL_SOURCE_DENSITY_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NodalSourceDensity<sofa::defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NodalSourceDensity<sofa::defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NodalSourceDensity<sofa::defaulttype::Vec3Types>;
#endif

}  // namespace sofa::component::solidmechanics::fem::elastic
