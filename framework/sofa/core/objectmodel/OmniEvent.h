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
#ifndef SOFA_CORE_OBJECTMODEL_OMNIEVENT_H
#define SOFA_CORE_OBJECTMODEL_OMNIEVENT_H

#include <sofa/core/objectmodel/Event.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

/**
 * @brief OmniEvent Class
 *
 * Implements an Event that notifies about a Mouse Interaction.
 */
class OmniEvent : public sofa::core::objectmodel::Event
{
public:

    /**
     * @brief Defines possible Mouse states.
     */
    typedef enum
    {
        Button1=0,
        Button2
    } State;


    /**
     * @brief Default constructor.
     */
    OmniEvent(State state, double posX, double posY, double posZ);

    /**
     * @brief Default destructor.
     */
    virtual ~OmniEvent();

    /**
     * @name Accessors
     */
    //@{
    double getPosX(void) const {return m_posX;};
    double getPosY(void) const {return m_posY;};
    double getPosZ(void) const {return m_posZ;};
    State getState(void) const {return m_state;};
    //}@

private:

    State m_state; ///< Mouse State on the event propagation.
    double m_posX, m_posY, m_posZ; ///< Mouse coordinates.
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_OBJECTMODEL_MOUSEEVENT_H
