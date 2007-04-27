/***************************************************************************
                          Force.cpp  -  description
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

#include "Force.h"
#include "ForceUnit.h"

Force::Force() {
    typeString="Force";
    unit = ForceUnit::N;
}

void Force::ansysPrint(std::ostream &o) const {
  Load::ansysPrint(o);
  // print the force in ansys format
  // normalize the direction
  // multiply by the value at time t
  // scalar product with Ox, Oy and Oz to get the three value to
  // be printed here
}

  
