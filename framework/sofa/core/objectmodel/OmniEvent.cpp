/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/core/objectmodel/OmniEvent.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{


OmniEvent::OmniEvent(State state, double posX, double posY, double posZ)
    : sofa::core::objectmodel::Event()
    , m_state(state)
    , m_posX(posX)
    , m_posY(posY)
    , m_posZ(posZ)
{

}

OmniEvent::OmniEvent(State state, Vector3 position, Quat orientation)
    : sofa::core::objectmodel::Event()
    , m_state(state)
    , m_position(position)
    , m_orientation(orientation)
{
    m_posX = position[0];
    m_posY = position[1];
    m_posZ = position[2];
}

OmniEvent::~OmniEvent()
{

}

} // namespace tree

} // namespace simulation

} // namespace sofa
