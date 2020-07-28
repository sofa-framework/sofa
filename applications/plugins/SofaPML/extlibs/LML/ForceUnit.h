/***************************************************************************
                          ForceUnit.h  -  description
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

#ifndef FORCEUNIT_H
#define FORCEUNIT_H

#include "Unit.h"

/** Class that defines the different units of a Load 'Force' 
 *
 * This class implements the type-safe design pattern.
 *
 * $Revision: 44 $
 */
class ForceUnit : public Unit {

public:
  /// picoNewtons
  static ForceUnit PN; 
  /// Newtons
  static ForceUnit N;
  /// KiloNewtons
  static ForceUnit KN; 

private:
  /// private constructor
  ForceUnit(const char * n) {unitString = n;}
};


#endif //FORCEUNIT_H
