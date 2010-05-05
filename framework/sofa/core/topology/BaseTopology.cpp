/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/topology/BaseTopology.h>

namespace sofa
{

namespace core
{

namespace topology
{
// GeometryAlgorithms implementation

void GeometryAlgorithms::init()
{
}

// TopologyAlgorithms implementation

void TopologyAlgorithms::init()
{
    this->getContext()->get(m_topologyContainer);
}

void TopologyAlgorithms::addTopologyChange(const TopologyChange *topologyChange)
{
    m_topologyContainer->addTopologyChange(topologyChange);
}

// TopologyModifier implementation

void TopologyModifier::init()
{
    this->getContext()->get(m_topologyContainer);
}

void TopologyModifier::addTopologyChange(const TopologyChange *topologyChange)
{
    m_topologyContainer->addTopologyChange(topologyChange);
}

void TopologyModifier::addStateChange(const TopologyChange *topologyChange)
{
    m_topologyContainer->addStateChange(topologyChange);
}

void TopologyModifier::propagateStateChanges() {}
void TopologyModifier::propagateTopologicalChanges() {}
void TopologyModifier::notifyEndingEvent() {}
void TopologyModifier::removeItems(sofa::helper::vector< unsigned int >& /*items*/) {}

// TopologyContainer implementation

void TopologyContainer::init()
{
    core::topology::BaseMeshTopology::init();
    core::topology::BaseTopologyObject::init();
}

std::list<const TopologyChange *>::const_iterator TopologyContainer::lastChange() const
{
    return m_changeList.end();
}

std::list<const TopologyChange *>::const_iterator TopologyContainer::firstChange() const
{
    return m_changeList.begin();
}

std::list<const TopologyChange *>::const_iterator TopologyContainer::lastStateChange() const
{
    return m_stateChangeList.end();
}

std::list<const TopologyChange *>::const_iterator TopologyContainer::firstStateChange() const
{
    return m_stateChangeList.begin();
}

void TopologyContainer::resetTopologyChangeList()
{
    for (std::list<const TopologyChange *>::iterator it=m_changeList.begin();
            it!=m_changeList.end(); ++it)
    {
        delete (*it);
    }

    m_changeList.clear();
}

void TopologyContainer::resetStateChangeList()
{
    for (std::list<const TopologyChange *>::iterator it=m_stateChangeList.begin();
            it!=m_stateChangeList.end(); ++it)
    {
        delete (*it);
    }

    m_stateChangeList.clear();
}

} // namespace topology

} // namespace core

} // namespace sofa

