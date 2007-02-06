//
// C++ Interface: KeypressedEvent
//
// Description:
//
//
// Author: Francois Faure, INRIA/UJF-Grenoble, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef Sofa_ComponentsKeypressedEvent_h
#define Sofa_ComponentsKeypressedEvent_h

#include <Sofa-old/Abstract/Event.h>

namespace Sofa
{

namespace Components
{

/**
	@author Francois Faure
*/
class KeypressedEvent : public Sofa::Abstract::Event
{
public:
    KeypressedEvent( char );

    ~KeypressedEvent();

    char getKey() const;

protected:
    char m_char;

};

}

}

#endif
