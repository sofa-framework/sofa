/***************************************************************************
                                   Cell.h 
                             -------------------
    begin             : Wed Aug 8 2001
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2006/10/17 14:33:21 $
    Version           : $Revision: 1.9 $
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef CELL_H
#define CELL_H

#include "Structure.h"
#include "StructuralComponent.h"

#include <sofa/helper/system/config.h>

class CellProperties;

/** A cell has an unique index in the physical model object, is composed by atoms, and different basic properties.
  * It is the most basic component composing a physical model.
  * $Revision: 1.9 $
  */
class Cell : public Structure , public StructuralComponent {
public:
    /** constructor that generates a unique index
    * @param t the type of the cell
    */
    Cell(PhysicalModel *, const StructureProperties::GeometricType t);
	/** constructor from xml node: try to read and get the parmaters from xml */
	Cell(PhysicalModel *, const StructureProperties::GeometricType t, xmlNodePtr);
    /** When you know the index of the cell, use this constructor.
    * @param t the type of the cell
    * @param ind give the unique index
    */
    Cell(PhysicalModel *, const StructureProperties::GeometricType t, const unsigned int ind);
    /// the destructor, my tailor. BECAREFUL: the atoms should not not be deleted here...
    virtual ~Cell() ;

    /** print to an output stream in "pseudo" XML format.
     *	If the StructuralComponent that calls this method is not the first in the
     *  list of composing SC, then a cellRef tag is printed (otherwise the list
     * 	of atom is printed).
     */
    void xmlPrint(std::ostream &, const StructuralComponent *);

    /** return true only if the parameter is equal to "MultiComponent" */
    virtual bool isInstanceOf(const char *) const;

    /**
     * This method overload the one defined in StructuralComponent.
     * The difference here is that the atoms composing the cell ARE NOT delete, still the
     *  list is cleared.
     * After this methode getNumberOfSubStructures() should return 0
     */
    virtual void deleteAllStructures();

    /// return the property
    CellProperties * getProperties();

    /// overloaded from Structural component, always return StructuralComponent::ATOMS
    StructuralComponent::ComposedBy composedBy();

    /** is this sc the one that will be the one that will make the cell to print out all its data
      * or is this a sc that will just print out the cell ref?
      */
    bool makePrintData(const StructuralComponent *);

    /** set the index.
     *  The index <b>have to be unique</b> otherwise this method
     *  has no effect. 
     *  The sub-classes method will check that this index is not in use.
     *  @return true only if the index of the structure was changed
     */
    virtual bool setIndex(const unsigned int);

	/** compute the normal of the facet
	 *  Warning : Only available for QUAD and TRIANGLE type cells
	 */
	SReal* normal(); 

	/** Return a structural component composed by the facets of the cells.
	 *  the facets are quads or/and triangles 
	 */
	StructuralComponent * getFacets();

	///Compute the surface of the cell
	SReal surface();

	/// Compute the volume of the cell
	SReal volume();


    
private:
    /// unique number (used to generate unique index for atoms if not given at the instanciation)
    static unsigned int nextUniqueIndex;

};

inline bool Cell::isInstanceOf(const char *className) const {
    return (strcmp(className, "Cell")==0);
}

inline StructuralComponent::ComposedBy Cell::composedBy() {
    return StructuralComponent::ATOMS;
}
#endif //CELL_H
