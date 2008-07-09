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

/** Question : shouldn't this be virtual, given this class has some virtual members?
         */
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

void TopologyContainer::resetTopologyChangeList()
{
    std::list<const TopologyChange *>::iterator it=m_changeList.begin();
    for (; it!=m_changeList.end(); ++it)
    {
        delete (*it);
    }
    m_changeList.erase(m_changeList.begin(),m_changeList.end());
}

void BaseTopology::resetStateChangeList() const
{
    getTopologyContainer()->resetStateChangeList();
}

void TopologyContainer::resetStateChangeList()
{
    std::list<const TopologyChange *>::iterator it=m_StateChangeList.begin();
    for (; it!=m_StateChangeList.end(); ++it)
    {
        delete (*it);
    }
    m_StateChangeList.erase(m_StateChangeList.begin(),m_StateChangeList.end());
}

} // namespace topology

} // namespace componentmodel

} // namespace core

} // namespace sofa

