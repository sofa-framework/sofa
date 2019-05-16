/***************************************************************************
                          Load.cpp  -  description
                             -------------------
    begin                : mar fï¿½v 11 2003
    copyright            : (C) 2003 by Emmanuel Promayon
    email                : Emmanuel.Promayon@imag.fr

 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/



#include "Load.h"

#include "Translation.h"
#include "Rotation.h"
#include "Force.h"
#include "Pressure.h"

//--------- static factory -------------
Load * Load::LoadFactory ( std::string type )
{
	Load * newOne = NULL;

	// instanciate depending on the load type
	if ( type=="Translation" )
	{
		newOne = new Translation();
	}
	// case rotation
	else if ( type=="Rotation" )
	{
		newOne = new Rotation();
	}
	// case force
	else if ( type=="Force" )
	{
		newOne = new Force();
	}
	// case pressure
	else if ( type=="Pressure" )
	{
		newOne = new Pressure();
	}

	return newOne;
}

//--------- constructor -------------
Load::Load()
{
	typeString = "unknown";
}

//--------- detructor -------------
Load::~Load()
{
	deleteEventList();
}


//--------- deleteEventList -------------
void Load::deleteEventList()
{
	std::vector<ValueEvent *>::iterator currentE;
	for ( currentE = eventList.begin(); currentE != eventList.end(); currentE++ )
	{
		delete ( *currentE );
	}
	eventList.clear();
}

//--------- setAllEvents -------------
void Load::setAllEvents ( std::vector<ValueEvent *>& newList )
{
	deleteEventList();
	std::vector<ValueEvent *>::iterator currentE;
	for ( currentE = newList.begin(); currentE != newList.end(); currentE++ )
	{
		addEvent ( *currentE );
	}
}

// --------------- isActive ---------------
bool Load::isActive ( const SReal t )
{
	std::vector<ValueEvent *>::iterator currentE;

	currentE = eventList.begin();
	while ( currentE!=eventList.end() )
	{
		// current event is active
		if ( ( *currentE )->isActive ( t ) )
			return true;
		currentE++;
	}

	return false;
}

// --------------- getValue ---------------
// the current norm value at time t
SReal Load::getValue ( const SReal t )
{
	std::vector<ValueEvent *>::iterator currentE;
	std::vector<ValueEvent *>::iterator nextE;

	// search the first active event
	currentE = eventList.begin();
	bool foundLastActive = false;
	bool isLast = false;
	while ( currentE!=eventList.end() && !foundLastActive )
	{
		// current event is active
		if ( ( *currentE )->isActive ( t ) )
		{
			// check if this is the last event
			nextE = currentE + 1;
			if ( nextE == eventList.end() )
			{
				// there is none, so currentE is the last active
				foundLastActive = true;
				isLast = true;
			}
			else
			{
				// if there is another event in the list, then check if it is active
				if ( ( *nextE )->isActive ( t ) )
				{
					// it is active, we need to continue (at least one more step)
					currentE++;
				}
				else
				{
					// it is not active: currentE is the last active
					foundLastActive = true;
				}
			}
		}
		else
		{
			// the current event is not active, check the next one
			currentE++;
		}
	}

	if ( !foundLastActive )
	{
		// not active
		return 0.0;
	}

	// if
	if ( isLast )
	{
		return ( *currentE )->getValue ( t );
	}
	else
	{
		return ( *currentE )->getValue ( t, ( *nextE ) );
	}
}

#ifdef WIN32
namespace std
{
	bool greater ( const ValueEvent* lhs, const ValueEvent* rhs )
	{
		return lhs->getDate() < rhs->getDate();
	}
}
#else
// ------------- sorting overloaded function ---------
// (saw some comments that say it is better to have
// this kind of overloads where you need them - compiler efficiency?
// so it is here cause needed in the addEvent method for the sort)
namespace std
{
	template <>
	struct greater<ValueEvent*>
	{
	 	bool operator() ( const ValueEvent* lhs, const ValueEvent* rhs ) const
		{
			return lhs->getDate() < rhs->getDate();
		}
	};
}
#endif

// --------------- addEvent ---------------
void Load::addEvent ( ValueEvent * ve )
{
	// insert the event
	eventList.push_back ( ve );

	// !TODO : SORT the <value event *> by date
	//* sort the list
#ifdef WIN32
	std::sort ( eventList.begin(), eventList.end(), std::greater<ValueEvent *>() ); // use the greater() method (see above)
#else
	std::sort ( eventList.begin(), eventList.end(), std::greater<ValueEvent *>() ); // use the greater() method (see above)
#endif

}

// --------------- addValueEvent ---------------
void Load::addValueEvent ( const SReal v, const SReal d )
{
	addEvent ( new ValueEvent ( v,d ) );
}


// --------------- addTarget ---------------
void Load::addTarget ( unsigned int target )
{
	targetList.add ( target );
}

// --------------- addTarget ---------------
void Load::addTarget ( std::string l )
{
	targetList.add ( l );
}

// --------------- addTarget ---------------
TargetList Load::getTargetList() const
{
	return targetList;
}

// --------------- addTarget ---------------
void Load::setTargetList ( const TargetList & t )
{
	targetList = t;
}

// --------------- numberOfTargets ---------------
unsigned int Load::numberOfTargets() const
{
	return targetList.getNumberOfTargets();
}

// --------------- getTarget ---------------
int Load::getTarget ( const unsigned int targetIndex ) const
{
	return targetList.getIndexedTarget ( targetIndex );
}


// --------------- getDirection ---------------
void Load::getDirection ( SReal &x, SReal &y, SReal &z ) const
{
	x = dir.getX();
	y = dir.getY();
	z = dir.getZ();
}

Direction Load::getDirection() const
{
	return dir;
}

// --------------- setDirection ---------------
void Load::setDirection ( SReal x, SReal y, SReal z )
{
	dir.set ( x ,y , z );
}

void Load::setDirection ( const Direction &d )
{
	dir = d;
}


// --------------- getValueEvent ---------------
ValueEvent * Load::getValueEvent ( const unsigned int i ) const
{
	if ( i<eventList.size() )
		return eventList[i];
	else
		return NULL;
}

// --------------- numberOfValueEvents ---------------
unsigned int Load::numberOfValueEvents() const
{
	return eventList.size();
}


// --------------- getUnit ---------------
Unit Load::getUnit() const
{
	return unit;
}

// --------------- setUnit ---------------
void Load::setUnit ( const Unit u )
{
	unit = u;
}


// --------------- getType --------------
std::string Load::getType() const
{
	return typeString;
}


// --------------- xmlPrint --------------
//Write the xml borns in the xml file
void  Load::xmlPrint ( std::ostream & o ) const
{
	// the Load tag
	o << "<load xsi:type=\"" << getType() << "\">" << std::endl;

	unsigned int i;
	/*  o << "\t<appliedTo>";
	  for (i = 0; i < numberOfTargets(); i++) {
	        if (i>0)
	            o << ",";
	         o << getTarget(i);
	    }
	    o << "</appliedTo>" << std::endl;
	*/
	o << "\t<appliedTo>" << targetList.toString() << "</appliedTo>" << std::endl;

	// the event tags
	ValueEvent *ve;
	for ( i=0; i<numberOfValueEvents(); i++ )
	{
		ve = getValueEvent ( i );
		ve->xmlPrint ( o );
	}

	dir.xmlPrint ( o );

	o << "\t<unit>" << getUnit().getUnitName() << "</unit>" << std::endl;

	o << "</load>" << std::endl;

}

// --------------- ansysPrint --------------
void  Load::ansysPrint ( std::ostream & o ) const
{
	// selection of points
	o << targetList.toAnsys() << std::endl;
}

// --------------- operator << ---------------
std::ostream & operator << ( std::ostream &o, Load )
{
	// the Load tag
	o << "<load xsi:type=\"" << "not implemented yet, USE xmlPrint() instead" /*<< ld.getType()*/ << "\">" << std::endl;

	o << "</load>" << std::endl;

	return o;
}
