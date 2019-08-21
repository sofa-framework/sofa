/***************************************************************************
                          Translation.cpp  -  description
                             -------------------
    begin                : mar fév 4 2003
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

#include "Translation.h"
#include "TranslationUnit.h"

Translation::Translation() {
  typeString = "Translation";
  unit = TranslationUnit::MM;
}
     
void Translation::ansysPrint(std::ostream &o) const {
  Load::ansysPrint(o);
  // print the translation in ansys format
//  o << "!-- BEWARE: only null displacement is implemented!" <<
  if (dir.getX() == 0 && dir.getY() == 0 && dir.getZ() == 0) {
    o << "D, ALL, ALL" << std::endl;
  }
  else {
    // normalize the direction
    // multiply by the value at time t
    // scalar product with Ox, Oy and Oz to get the three value to
    // be printed here
  }
}


