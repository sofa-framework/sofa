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

#include <sofa/core/topology/TopologicalMapping.h>
#include <sofa/component/topology/mapping/config.h>

namespace sofa::component::topology::mapping
{

class SOFA_COMPONENT_TOPOLOGY_MAPPING_API Hexa2PrismTopologicalMapping : public sofa::core::topology::TopologicalMapping
{
public:
    SOFA_CLASS(Hexa2PrismTopologicalMapping, sofa::core::topology::TopologicalMapping);

    virtual void init() override;
    virtual Index getFromIndex(Index ind) override;
    virtual void updateTopologicalMappingTopDown() override;

protected:
    Hexa2PrismTopologicalMapping();

    void convertHexaToPrisms();
};

} // namespace sofa::component::topology::mapping
