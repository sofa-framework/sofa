/***************************************************************************
                          ValueEvent.cpp  -  description
                             -------------------
    begin                : mar mar 4 2003
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

///a valueEvent of a load is composed of a value and a date

#include "ValueEvent.h"

//--------- isActive ----------
bool ValueEvent::isActive(const SReal t) const {
    return (t>=date);
}

//--------- getValue ----------
double ValueEvent::getValue()  {
    return value;
}

double ValueEvent::getValue(const double t)  {
    if (isActive(t))
        return getValue();
    else
        return 0.0;
}

double ValueEvent::getValue(const double t, ValueEvent * nextEvent)  {
    if (isActive(t)) {
        return (getValue()
                + (t-getDate())
                *((nextEvent->getValue() - getValue())/(nextEvent->getDate()-getDate())) );
    } else
        return 0.0; // Value is not active
}


//--------- getDate ----------
double ValueEvent::getDate() const {
    return date;
}

// --------------- xmlPrint ---------------
void ValueEvent::xmlPrint(std::ostream & o) {
    o << "\t" << "<valueEvent date=\"" << date
    << "\" value=\"" << value
    << "\"/>" << std::endl;
}

// --------------- operator << ---------------
std::ostream & operator << (std::ostream &o,  ValueEvent e) {

    o << "\t" << "<valueEvent date=\"" << e.getDate()
    << "\" value=\"" << e.getValue()
    << "\"/>" << std::endl;

    return o;
}

// --------------- setDate ---------------
void ValueEvent::setDate(const double d) {
    date = d;
}

// --------------- setValue ---------------
void ValueEvent::setValue(const double v) {
    value = v;
}


