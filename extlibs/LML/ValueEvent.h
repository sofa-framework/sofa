/***************************************************************************
                          ValueEvent.h  -  description
                             -------------------
    begin                : mar f�v 4 2003
    copyright            : (C) 2003 by TIMC
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

#ifndef VALUEEVENT_H
#define VALUEEVENT_H

#include "xmlio.h"
#include <sofa/helper/system/config.h>

/** a valueEvent of a load is composed of a value and a date
 *
 * $Revision: 44 $
 */
class ValueEvent {

public:
    /// constructor with initial values
    ValueEvent(const double v, const double d):value(v),date(d) {};

    /// default constructor
    ValueEvent():value(0.0), date(0.0) {};
    
    /// destructor
    ~ValueEvent() {};

    /// return true if the event is active at time t
    bool isActive(const SReal t) const;

    /// return the scalar value of the event
    double getValue() ;
    /// return the scalar value of the event at time t
    double getValue(const double t) ;
    /// return the scalar value of the event at time t, knowing that next event is nextE
    double getValue(const double t, ValueEvent * nextE) ;

    /// set the value event date
    void setDate(const double);
    /// set the value event value
    void setValue(const double);
    
    /// double get start time
    double getDate() const;

    /** print to an output stream in XML format.
     * @see load.xsd the loadML XML schema
     */
    friend std::ostream & operator << (std::ostream &, ValueEvent);
 
    /// Print to an ostream
    void xmlPrint(std::ostream &);

private:
    /// actual value
    double value;
    /// actual date
    double date;

};

#endif //VALUEEVENT_H
