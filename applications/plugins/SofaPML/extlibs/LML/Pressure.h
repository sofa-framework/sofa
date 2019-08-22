/***************************************************************************
                          Pressure.h  -  description
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
 

#ifndef PRESSURE_H
#define PRESSURE_H
#include "PressureUnit.h"
#include "Load.h"

/** Class that defines the type of Load 'Pressure' 
 *
 * $Revision: 44 $
 */
class Pressure : public Load {

public:
    Pressure() { typeString="Pressure";unit=PressureUnit::KPA;}
};

#endif //PRESSURE_H

