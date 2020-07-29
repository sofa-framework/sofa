/***************************************************************************
                          TargetList.h  -  description
                             -------------------
    begin                : Sun Mar 30 2003
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

#ifndef TARGETLIST_H
#define TARGETLIST_H

#include "xmlio.h"

/**
 * Manage a list of targets, can be created/modified using either an integer,
 * a name, a list of integers given in a string (e.g. "1,3,5,10-15"), or
 * a list of names given in a string (e.g. "componentA,componentB").
 *
 * Mixed (indexed/named) are not supported (yet?)
 *
 * @author Emmanuel Promayon
 *
 * $Revision: 44 $
 */
class TargetList {

public:
  /// default constructor, the target list is empty
  TargetList();
  
  /** create a target list using initial list of targets.
   *  The list of targets can be either a indexed list (eg. "5-7,10,12-15")
   *  or a named list (eg. "componentA,componentB")
   */
  TargetList(const std::string);
  
  /// create a target list using another one
  TargetList(const TargetList &);

  /// add a load using an integer
  void add(const unsigned int);

  /// add a load using a list (either an indexed list or a named list)
  void add(const std::string);

  /// get the nr of indevidual targets
  unsigned int getNumberOfTargets() const;

  /** get an indexed target
    * @return -1 if index out of bound or if targets are not indexed (i.e targets are named)
    */
  int getIndexedTarget(const unsigned int) const;
  
  /** get a named target
    * @return "" if index out of bound or if targets are not named (i.e targets are indexed)
    */
  std::string getNamedTarget(const unsigned int) const;
  
  /// clear the list
  void clear();

  /// return the list in a handy/compact format (compact interval, i.e. 4,5,6 becomes 4-6, ...)
  std::string toString() const;

  /// return the ANSYS command to select the list of target (only work for indexed targets)
  std::string toAnsys() const;
  
  /// return true only if the list of target are indexes
  bool indexedTargets() const;
  
  /** return true only if this is the list of target are indexes and the given index is in the list
   *  or if the name of the entities is star!
   */
  bool isIn(unsigned int) const;

private:
  /// list of indexed target = index of the entities
  std::vector <unsigned int> indexedTargetList;
  
  /// list of named target = name of the entities
  std::vector <std::string> namedTargetList;
};

#endif
