/***************************************************************************
                          Unit.h  -  description
                             -------------------
    begin                : mar f�v 4 2003
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

#ifndef UNIT_H
#define UNIT_H

#include "xmlio.h"

class Unit;

/** Class that defines the unit of the Load 
  *
  * All sub classes implement the (nice) type-safe design pattern.
  *
  * $Revision: 44 $
  */
class Unit {
public:
  std::string getUnitName() { return unitString; };
protected:
  std::string unitString;
};

#endif //UNIT_H
