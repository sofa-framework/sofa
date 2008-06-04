/***************************************************************************
                          TranslationUnit.h  -  description
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

#ifndef TRANSLATIONUNIT_H
#define TRANSLATIONUNIT_H

#include "Unit.h"

/** TranslationUnit model the different values that can be taken by the unit
  * field of a translation.
  *
  * This class implements the type-safe design pattern.
  *
  * $Revision: 44 $
  */
class TranslationUnit : public Unit {

public:
  /// millimeters
  static TranslationUnit MM; 
  /// micro meters
  static TranslationUnit MICRO_M;
  /// nano meters
  static TranslationUnit NM;
  
private:
  TranslationUnit(char * n) {unitString = n;}
};


#endif //TRANSLATIONUNIT_H
