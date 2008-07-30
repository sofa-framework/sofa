/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/core/componentmodel/topology/BaseTopology.h>

namespace sofa
{
namespace core
{
namespace componentmodel
{
namespace topology
{
// BaseTopology implementation

BaseTopology::BaseTopology(bool isMainTopology)
    : m_topologyContainer(NULL),
      m_topologyModifier(NULL),
      m_topologyAlgorithms(NULL),
      m_geometryAlgorithms(NULL),
      m_mainTopology(isMainTopology),
      revisionCounter(0)
{}

BaseTopology::~BaseTopology()
{
}

std::list<const TopologyChange *>::const_iterator BaseTopology::lastChange() const
{
    return m_topologyContainer->lastChange();
}

std::list<const TopologyChange *>::const_iterator BaseTopology::firstChange() const
{
    return m_topologyContainer->firstChange();
}

std::list<const TopologyChange *>::const_iterator BaseTopology::lastStateChange() const
{
    return m_topologyContainer->lastStateChange();
}

std::list<const TopologyChange *>::const_iterator BaseTopology::firstStateChange() const
{
    return m_topologyContainer->firstStateChange();
}

void BaseTopology::propagateTopologicalChanges()
{
    m_topologyContainer->propagateTopologicalChanges();

    ++revisionCounter;
}

void BaseTopology::propagateStateChanges()
{
    m_topologyContainer->propagateStateChanges();
}

void BaseTopology::resetTopologyChangeList() const
{
    m_topologyContainer->resetTopologyChangeList();
}

void BaseTopology::resetStateChangeList() const
{
    m_topologyContainer->resetStateChangeList();
}

// TopologyAlgorithms implementation

void TopologyAlgorithms::addTopologyChange(const TopologyChange *topologyChange)
{
    m_basicTopology->getTopologyContainer()->addTopologyChange(topologyChange);
}

// TopologyModifier implementation

void TopologyModifier::addTopologyChange(const TopologyChange *topologyChange)
{
    m_topologyContainer->addTopologyChange(topologyChange);
}

void TopologyModifier::addStateChange(const TopologyChange *topologyChange)
{
    m_topologyContainer->addStateChange(topologyChange);
}

// TopologyContainer implementation


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

} // namespace componentmodel

} // namespace core

} // namespace sofa

