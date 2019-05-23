/***************************************************************************
               CellProperties.h  -  Base of the cell properties
                             -------------------
    begin             : Wed Aug 8 2001
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2004/08/11 14:59:19 $
    Version           : $Revision: 1.5 $
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef BASICCELLPROPERTIES_H
#define BASICCELLPROPERTIES_H

#include "StructureProperties.h"

/** Describes the properties common to all cells.
 *
  * $Revision: 1.5 $
 */
class BasicCellProperties : public StructureProperties {
public:
    /** Default constructor : generate an unique index
     * @param t the type of the cell
     */
    BasicCellProperties(PhysicalModel *, const StructureProperties::GeometricType t);
	/** constructor from xml node: try to read and get the parmaters from xml */
    BasicCellProperties(PhysicalModel *, const StructureProperties::GeometricType, xmlNodePtr);
    /** Use this constructor when you specifically want to set the index
     * @param t the type of the cell
     * @param ind an unique index
     */
    BasicCellProperties(PhysicalModel *, const StructureProperties::GeometricType t, const unsigned int ind);
    /** the destructor...
    	*/
    virtual ~BasicCellProperties() {}
    ;

    /** print to an output stream in "pseaudo" XML format.
       */
    virtual void xmlPrint(std::ostream &) =0;

    /** Reinitialize the unique index to zero (usually that what you want to do when you
     	* start to load a new PhysicalModel
     	*/
    static void resetUniqueIndex();

protected:
    /// write the default xml properties (beginning)
    void beginXML(std::ostream &);
    /// write the default xml properties (end)
    void endXML(std::ostream &);

private:
    /// unique number (used to generate unique index for atoms if not given at the instanciation)
    static unsigned int maxUniqueIndex;

};
#endif //BASICCELLPROPERTIES_H
