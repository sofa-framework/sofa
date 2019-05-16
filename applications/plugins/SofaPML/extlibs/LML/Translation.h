/***************************************************************************
                          Translation.h  -  description
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

#ifndef TRANSLATION_H
#define TRANSLATION_H

#include "TranslationUnit.h"
#include "Load.h"

/** Class that defines the type of Load 'Translation'
  *
  * $Revision: 44 $
  */
class Translation : public Load {
public:
  Translation();

  /** Redefinition of ansysPrint to print in ansys format 
   * \warning not everything is implemented (yet!)
   */
  virtual void ansysPrint(std::ostream &) const;

};

#endif //TRANSLATION_H
