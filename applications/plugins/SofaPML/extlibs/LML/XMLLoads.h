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

/* old doc
Usage example:
<pre>
// reading
main() {
  XMLLoads data;
  data.xmlRead("toto.lml");
  Loads *l = data.getLoad();
  ...
  cout << l;
}
 
// writing
main() {
  Loads *l= new Loads();
  Translation *t = new Translation();
  t->setUnit(..)
  ...
  cout << l;
}
</pre>
*/

/** (obsolete) Allows to read loads from an XML file (LML)/
 *   
 *  Please use now Loads(std::string) constructor...
 *
 * 
  *@author Emmanuel Promayon
  *
  * $Revision: 51 $
  */
class XMLLoads {
public:

    /** create a list of loads from an LML file.
     * @param fileName the name of the lml file (xml)
     * @param allLoads the pointer to the Loads instance (to store all loads), if null a new one is instanciated
     */
    XMLLoads(std::string fileName, Loads* allLoads);

    /// create an empty list of loads
    XMLLoads();

    /// (obsolete) create a list of loads from an LML file (instanciate a new Loads class object)
    XMLLoads(std::string fileName);

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
