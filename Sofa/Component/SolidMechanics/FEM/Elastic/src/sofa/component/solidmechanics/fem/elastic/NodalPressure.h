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

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_NODAL_PRESSURE_CPP)
#include <sofa/defaulttype/VecTypes.h>
#endif

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * @class NodalPressure
 * @brief A pressure prescribed at the nodes, one scalar per node.
 *
 * @tparam TDataTypes The data types used for positions, velocities, etc. (e.g., Vec3Types).
 */
template <class TDataTypes>
class NodalPressure : public sofa::core::BaseNodalProperty<sofa::Real_t<TDataTypes>>
{
public:
    using DataTypes = TDataTypes;
    using Real = sofa::Real_t<DataTypes>;

    SOFA_CLASS(SOFA_TEMPLATE(NodalPressure, DataTypes),
        SOFA_TEMPLATE(sofa::core::BaseNodalProperty, sofa::Real_t<DataTypes>));

protected:

    NodalPressure() : sofa::core::BaseNodalProperty<Real>(Real{}) {}
};

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_NODAL_PRESSURE_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NodalPressure<sofa::defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API NodalPressure<sofa::defaulttype::Vec3Types>;
#endif

}  // namespace sofa::component::solidmechanics::fem::elastic
