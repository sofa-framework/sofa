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
#include <sofa/component/topology/mapping/CenterPointTopologicalMapping.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/component/topology/container/dynamic/PointSetTopologyModifier.h>

#include <sofa/type/Vec.h>
#include <map>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::topology::mapping
{
using namespace sofa::defaulttype;
using namespace sofa::component::topology::mapping;
using namespace sofa::core::topology;

// Register in the Factory
int CenterPointTopologicalMappingClass = core::RegisterObject ( "" )
        .add< CenterPointTopologicalMapping >()
        ;

// Implementation
CenterPointTopologicalMapping::CenterPointTopologicalMapping ()
{
}

void CenterPointTopologicalMapping::init()
{
    this->d_componentState.setValue(core::objectmodel::ComponentState::Invalid);
    if(fromModel && toModel)
    {
        const auto nbHexa = fromModel->getNbHexahedra();
        toModel->setNbPoints(nbHexa);

        sofa::core::behavior::BaseMechanicalState::SPtr state{};
        toModel->getContext()->get(state);
        if (state)
        {
            state->resize(nbHexa);
            this->d_componentState.setValue(core::objectmodel::ComponentState::Valid);
        }
    }
}

void CenterPointTopologicalMapping::updateTopologicalMappingTopDown()
{
    if(this->d_componentState.getValue() == core::objectmodel::ComponentState::Valid)
    {
        std::list<const TopologyChange *>::const_iterator changeIt = fromModel->beginChange();
        const std::list<const TopologyChange *>::const_iterator itEnd = fromModel->endChange();

        container::dynamic::PointSetTopologyModifier *to_pstm;
        toModel->getContext()->get(to_pstm);

        while( changeIt != itEnd )
        {
            const TopologyChangeType changeType = (*changeIt)->getChangeType();

            switch( changeType )
            {
            case core::topology::HEXAHEDRAADDED:
            {
                const size_t nbHexaAdded = ( static_cast< const HexahedraAdded *>( *changeIt ) )->getNbAddedHexahedra();
                to_pstm->addPoints(nbHexaAdded, true);
                break;
            }
            case core::topology::HEXAHEDRAREMOVED:
            {
                auto tab = ( static_cast< const HexahedraRemoved *>( *changeIt ) )->getArray();
                to_pstm->removePoints(tab, true);
                break;
            }
            default:
                break;
            }
            ++changeIt;
        }
    }
}

} //namespace sofa::component::topology::mapping
