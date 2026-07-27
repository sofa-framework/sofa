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
#include <sofa/component/solidmechanics/fem/elastic/impl/trait.h>
#include <sofa/core/behavior/TopologyAccessor.h>
#include <sofa/core/behavior/SingleStateAccessor.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes, class ElementType>
class VonMisesStress : public core::behavior::TopologyAccessor, public core::behavior::SingleStateAccessor<DataTypes>
{
public:
    SOFA_CLASS2(
        SOFA_TEMPLATE2(VonMisesStress, DataTypes, ElementType),
        core::behavior::TopologyAccessor,
        core::behavior::SingleStateAccessor<DataTypes>);

    void init() override;

    Data<sofa::type::vector<sofa::Real_t<DataTypes> > > d_nodalStress;

protected:

    using trait = sofa::component::solidmechanics::fem::elastic::trait<DataTypes, ElementType>;

    VonMisesStress();

    void handleEvent(core::objectmodel::Event*) override;

};

}  // namespace sofa::component::solidmechanics::fem::elastic
