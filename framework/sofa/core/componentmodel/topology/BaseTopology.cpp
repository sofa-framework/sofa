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
      filename(initData(&filename, "filename", "Filename of the object"))
{}

BaseTopology::~BaseTopology()
{
    if (m_topologyContainer)
        delete m_topologyContainer;
    if (m_topologyModifier)
        delete m_topologyModifier;
    if (m_topologyAlgorithms)
        delete m_topologyAlgorithms;
    if (m_geometryAlgorithms)
        delete m_geometryAlgorithms;
}

std::list<const TopologyChange *>::const_iterator BaseTopology::lastChange() const
{
    return m_topologyContainer->getChangeList().end();
}

std::list<const TopologyChange *>::const_iterator BaseTopology::firstChange() const
{
    return m_topologyContainer->getChangeList().begin();
}

std::list<const TopologyChange *>::const_iterator BaseTopology::lastStateChange() const
{
    return m_topologyContainer->getStateChangeList().end();
}

std::list<const TopologyChange *>::const_iterator BaseTopology::firstStateChange() const
{
    return m_topologyContainer->getStateChangeList().begin();
}

void BaseTopology::resetTopologyChangeList() const
{
    getTopologyContainer()->resetTopologyChangeList();
}

void BaseTopology::resetStateChangeList() const
{
    getTopologyContainer()->resetStateChangeList();
}

// TopologyAlgorithms implementation

void TopologyAlgorithms::addTopologyChange(const TopologyChange *topologyChange)
{
    m_basicTopology->getTopologyContainer()->addTopologyChange(topologyChange);
}

// TopologyModifier implementation

void TopologyModifier::addTopologyChange(const TopologyChange *topologyChange)
{
    m_basicTopology->getTopologyContainer()->addTopologyChange(topologyChange);
}

void TopologyModifier::addStateChange(const TopologyChange *topologyChange)
{
    m_basicTopology->getTopologyContainer()->addStateChange(topologyChange);
}

// TopologyContainer implementation

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
    for (std::list<const TopologyChange *>::iterator it=m_StateChangeList.begin();
            it!=m_StateChangeList.end(); ++it)
    {
        delete (*it);
    }

    m_StateChangeList.clear();
}

} // namespace topology

} // namespace componentmodel

} // namespace core

} // namespace sofa

