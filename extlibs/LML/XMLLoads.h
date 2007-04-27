/***************************************************************************
                          xmlloads.h  -  description
                             -------------------
    begin                : mar mar 4 2003
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

#ifndef XMLLOADS_H
#define XMLLOADS_H

#include "Loads.h"

#include <libxml/parser.h>
#include <libxml/tree.h>


/** Allows to read loads from an XML file (LML)
 
Usage example:
<pre>
main() {
  XMLLoads data;
  data.xmlRead("toto.lml");
  Loads *l = data.getLoad();
  ...
  cout << l;
}
 
main() {
  Loads *l= new Loads();
  Translation *t = new Translation();
  t->setUnit(..)
  ...
  cout << l;
}
</pre>
 
  *@author Emmanuel Promayon
  *
  * $Revision: 1.7 $
  */
class XMLLoads {
public:

    /// create a list of loads from an IML file
    XMLLoads(std::string);
    /// create an empty list of loads
    XMLLoads();
    /// destructor (delete all loads)
    ~XMLLoads();

    /// get the list of loads
    Loads * getLoads();

    /// add a new load to the list (creates one if no loads yet)
    void addLoad(Load *);
    
    /** Read the loads from an LML file.
     *  Uses libxml2.
     */
    void xmlRead(std::string);

protected:

	/** Read a load xml element, from lml file
	 *  Uses xmlNodePtr from libxml2
     */
    bool parseElement(xmlNodePtr elem);

	/** Read a load AppliedTo property, from lml file
	 *  Uses xmlNodePtr from libxml2
	 *  AppliedTo contains the list of atoms on which the load is applied
     */
    void readLoadAppliedTo(xmlNodePtr elem, Load *currentLoad);

	/** Read a load Direction property, from lml file
	 *  Uses xmlNodePtr from libxml2
	 *  Direction contains a 3D vector
     */
    void readLoadDirection(xmlNodePtr elem, Load *currentLoad);

	/** Read a load valueEvent property, from lml file
	 *  Uses xmlNodePtr from libxml2
	 *  valueEvent contains a pair of date/value
     */
    void readLoadValueEvent(xmlNodePtr elem, Load *currentLoad);

	/** Read a load Unit property, from lml file
	 *  Uses xmlNodePtr from libxml2
	 *  Unit contains the unit value of the load
     */
    void readLoadUnit(xmlNodePtr elem, Load *currentLoad);


private:
    Loads *l;

	const char*xmlFile;
};

#endif
