/***************************************************************************
                          PressureUnit.h  -  description
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

#ifndef PRESSUREUNIT_H
#define PRESSUREUNIT_H

#include "Unit.h"

/** Class that defines the different units of a Load 'Pressure' 
 *
 * This class implements the type-safe design pattern.
 *
 * $Revision: 44 $
 */
class PressureUnit : public Unit {

public:
  /// kiloPascal
  static PressureUnit KPA; 
  /// Millimeters of mercure
  static PressureUnit MMHG;
 
private:
  PressureUnit(char * n) {unitString = n;}
};


#endif //PRESSUREUNIT_H
