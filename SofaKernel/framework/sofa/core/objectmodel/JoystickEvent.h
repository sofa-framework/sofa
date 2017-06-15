/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

// Author: Pierre-Jean Bensoussan, Digital Trainers (2008)

// This file is a work based on the open source release of VGSDK.
//
// VGSDK - Copyright (C) 2008, Nicolas Papier.
// Distributed under the terms of the GNU Library General Public License (LGPL)
// as published by the Free Software Foundation.
// Author Nicolas Papier
// Author Guillaume Brocker

#ifndef SOFA_CORE_OBJECTMODEL_JOYSTICKEVENT_H
#define SOFA_CORE_OBJECTMODEL_JOYSTICKEVENT_H

#include <sofa/core/objectmodel/Event.h>
#include <vector>

namespace sofa
{

namespace core
{

namespace objectmodel
{

/**
* @brief JoystickEvent Class
*
*
*/
class SOFA_CORE_API JoystickEvent : public sofa::core::objectmodel::Event
{
public:

    SOFA_EVENT_H( JoystickEvent )

    /**
     * @brief	Implements an event that notifies about axis positions (like analog controls of a joystick).
     * 			The axis position is normalized so values are always in the range [-1, 1].
     */
    class SOFA_CORE_API AxisEvent
    {
    public:
        /**
         * @brief	Constructor
         *
         * @param	index			axis' index
         * @param	value			axis' value (must be in the range [-1, 1])
         *
         * @pre		(value >= -1.f) && (value <= 1.f)
         */
        AxisEvent( const int /*index*/, const float /*value*/ );

        /**
         * @brief Default destructor.
         */
        virtual ~AxisEvent() {};

        /**
         * @name	Accessors
         */
        //@{
        /**
         * @brief	Retrieves the index of the axis.
         *
         * @return	the axis' index
         */
        int getIndex() const;

        /**
         * @brief	Retrieves the value of the axis.
         *
         * @remark	Values are always in the range [-1, 1].
         *
         * @return	the axis' value
         */
        float getValue() const;
        //@}

        virtual const char* getClassName() const { return "AxisEvent"; }
    private:
        const int	m_index;	///< The index of the axis
        const float	m_value;	///< The value of the xais
    };



    /**
     * @brief Implements the button event for joysticks
     */
    class SOFA_CORE_API ButtonEvent
    {
    public:
        /**
         * @brief	Default constructor
         */
        ButtonEvent( const int  buttonStates = 0 );

        /**
         * @brief	Default destructor
         */
        virtual ~ButtonEvent() {};

        /**
         * @brief
         */
        void setButtons(const int);

        /**
         * @brief
         */
        bool getButton(const int) const;

        virtual const char* getClassName() const { return "ButtonEvent"; }

    private:
        bool m_buttons[32]; ///< Current State of the whole Joystick Buttons
    };



    /**
     * @brief Implements an event notifiying changes about a directionnal hat on a device (like a joystick).
     */
    class SOFA_CORE_API HatEvent
    {
    public:
        /**
         * @brief	Defines possible hat states.
         */
        typedef enum
        {
            CENTERED	= 0,
            UP			= 1 << 0,
            RIGHT		= 1 << 1,
            DOWN		= 1 << 2,
            LEFT		= 1 << 3,
            UP_RIGHT	= UP|RIGHT,
            DOWN_RIGHT	= DOWN|RIGHT,
            DOWN_LEFT	= DOWN|LEFT,
            UP_LEFT		= UP|LEFT
        } State;


        /**
         * @brief	Constructor
         *
         * @param	index			the hat's index
         * @param	state			the hat's state
         */
        HatEvent( const int /*index*/, const State /*state*/ );

        /**
         * @brief Default destructor.
         */
        virtual ~HatEvent() {};

        /**
         * @name	Accessors
         */
        //@{
        /**
         * @brief	Retrieves the index of the hat.
         *
         * @return	the hat's index
         */
        int getIndex() const;

        /**
         * @brief	Retrieves the state of the hat.
         *
         * @return	the hat's state
         */
        State getState() const;
        //@}

        virtual const char* getClassName() const { return "HatEvent"; }
    private:

        const int	m_index;	///< The index of the hat
        const State	m_state;	///< The state of the hat
    };

    /**
     * @name	Accessors
     */
    //@{

    /**
     * @name Axis
     */
    //@{

    const AxisEvent *getAxisEvent(const int /*index*/) const;

    const std::vector<AxisEvent*> &getAxisEvents(void) const;

    unsigned int getAxisEventsSize(void) const;

    void addAxisEvent( AxisEvent * );

    //@}

    /**
     * @name	Button
     */
    //@{

    ButtonEvent *getButtonEvent(void) const;

    bool getButton(unsigned int /*index*/) const;

    void setButtonEvent( ButtonEvent * );

    //@}

    /**
     * @name Hat
     */
    //@{

    const HatEvent *getHatEvent(const int /*index*/) const;

    const std::vector<HatEvent*> &getHatEvents(void) const;

    unsigned int getHatEventsSize(void) const;

    void addHatEvent( HatEvent * );

    //@}

    //@}

    /**
     * @brief default constructor.
     */
    JoystickEvent();

    /**
     * @brief default destructor.
     */
    virtual ~JoystickEvent();

    virtual const char* getClassName() const { return "JoystickEvent"; }
protected:

    std::vector< AxisEvent* > axisEvents; ///< State of the Analogic Pad
    ButtonEvent *buttonEvent; ///< State of the Joystick Buttons
    std::vector< HatEvent* > hatEvents; ///< State of the directional cross

private:

};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_OBJECTMODEL_JOYSTICKEVENT_H
