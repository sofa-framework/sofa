/***************************************************************************
                          Rotation.h  -  description
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

#ifndef ROTATION_H
#define ROTATION_H
#include "RotationUnit.h"
#include "Load.h"

/** Class that defines the type of Load 'Rotation' 
 *
 * $Revision: 44 $
 */
class Rotation : public Load {

public:
  Rotation() {typeString="Rotation"; unit=RotationUnit::DEG;}
        
};

#endif //ROTATION_H

