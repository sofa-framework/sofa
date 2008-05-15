/***************************************************************************
                            StructuralComponent.h
                             -------------------
    begin             : Wed Aug 8 2001
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2007/03/26 07:20:54 $
    Version           : $Revision: 1.24 $
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

// Allow to compile the stl types without warnings on MSVC
// this line should be BEFORE the #ifndef
//#ifdef WIN32
//#pragma warning(disable:4786) // disable C4786 warning
//#endif


#ifndef STRUCTURALCOMPONENT_H
#define STRUCTURALCOMPONENT_H

#include "PhysicalModelIO.h"

// imp includes
#include "Component.h"
#include "Structure.h"

#include <sofa/helper/system/config.h>
class Object3D;
#include <StructuralComponentProperties.h>

// other includes
#include <cstring>
#include <vector>
#include <algorithm> // for the remove

/** A structural component is composed either by cell or by atoms.
 * @author Emmanuel Promayon
  * $Revision: 1.24 $
*/
class StructuralComponent : public Component {
public:
    /** Default Constructor.
     */
    StructuralComponent(PhysicalModel *);
	/** constructor from xml node: try to read and get the parmaters from xml */
	StructuralComponent(PhysicalModel *, xmlNodePtr);
    /** constructor that allows to name the structure.
     */
    StructuralComponent(PhysicalModel *, std::string);
    
    /// delete all the structures (call the deleteAllStructures method)
    virtual ~StructuralComponent();

    /// get the number of structures
    unsigned int getNumberOfStructures() const;

    /**
    * Add a Structure in the list (and tells the structure to remove this structural
    * component from its list).
    * @param s the structure to add
    * @param check (default value: true) tell if the method should call isCompatible(Structure *s) before inserting s
    */
    void addStructure(Structure *s, bool check = true);

    /**
    * Add a Structure in the list, only if it is not already in
    * (and tells the structure to remove this structural
    * component from its list).
    * @param s the structure to add
    * @return a boolean telling if s was added or not
    */
    bool addStructureIfNotIn(Structure *s);

    /**
     * Remove a structure from the list (and tells the structure to remove this structural
      * component from its list).
      * Becareful: this method DOES NOT delete the object and/or free the memory.
     * @param s the ptr to the structure to remove
     */
    virtual void removeStructure(Structure *s);

    /**
     * this method free all the sub-components (i.e. delete all the sub component
     *  and clear the list).
     * After this methode getNumberOfSubStructures() should return 0
     */
    virtual void deleteAllStructures();

    /**
    * get a structure by its index (fisrt structure is at index 0)
    */
    Structure * getStructure(const unsigned int) const;

    /**
     *  get a structure by its name
     */
    Structure * getStructureByName(const std::string);

    /**
     *  get a structure by its unique index
     */
    Structure * getStructureByIndex(const unsigned int);
    
    /** print to an output stream in "pseudo" XML format (do nothing if there are no sub structures).
    */
    void xmlPrint(std::ostream &) const;

    /** return true only if the parameter is equal to "StructuralComponent" */
    virtual bool isInstanceOf(const char *) const;

    /// get the total nr of cell of the component
    unsigned int getNumberOfCells() const;

    /// get cell by order number (not cell index)
    Cell * getCell(unsigned int) const;

    /** Return the position of a cell in the StructuralComponent structures list.
    *	return -1 if the cell is not found. */
    //	int getCellPosition(const Cell *cell) const;

    /** Return a StructuralComponent with all the atoms of this structural component.
     *	If this structural component is already a composed of atoms, return this.
     *	If it is composed of cells or mixed atoms and cells, return all the atoms
     *	used. Each atom is present only once in the result. */
    StructuralComponent *getAtoms();

    /// get the structural component properties of this SC
    StructuralComponentProperties * getProperties();

    /// Set the new color (using a StructuralComponentProperties::Color enum)
    void setColor(const StructuralComponentProperties::Color c);
    /// Set the new RGBA color
    void setColor(const SReal r, const SReal b, const SReal g, const SReal a);
    /// Set the new RGB color
    void setColor(const SReal r, const SReal b, const SReal g);
    /** Get the color
    	* @return an array of 4 SReals (red, blue, green and alpha values)
    	*/
    double * getColor() const;
    /** Get the color by its 4 componants r,g,b and a */
    void getColor(double *r, double *g, double *b, double *a) const;

    /// Return the color as a code (see StructuralComponentProperties::Color enum)
    StructuralComponentProperties::Color getStructuralComponentPropertiesColor() const;

    /// set the rendering mode
    void setMode(const RenderingMode::Mode);
    /// get the rendering mode
    RenderingMode::Mode getMode() const;
    /// tell if a specific rendering mode is visible or not
    virtual bool isVisible(const RenderingMode::Mode mode) const;
    /// set the visibility of a specific rendering mode
    virtual void setVisible(const RenderingMode::Mode mode, const bool b);

    /** What this structural component is made of */
    enum ComposedBy {
        NOTHING, /**< there are no structure yet, so everything is possible */
        CELLS, /**< the structural component is made of cells */
        ATOMS /**< the structural component is made of atoms */
    };

    /** return the type of structure composing the structural component:
      * a structural component is either a list of cells or atoms, or of nothing if it is empty
      * (see enum ComposedBy).
      */
    ComposedBy composedBy();

    /** return true if the given structure is compatible with what composes this structural component.
      * E.g. if the structural is made of cell, and the structure is a cell.
      */
    bool isCompatible(Structure *);

    /** optimize the I/O of the std:vector structures.
      * If you know the nr of structures to be in the SC, please give it here,
      * it will greatly speed the building of the structure
      */
    void plannedNumberOfStructures(const unsigned int);

protected:
    /**
    * List of the structure representing this component, all the structure in this list are either all Atom or all Cell (no mix!)
    */
    std::vector <Structure *> structures;

    /** List of all the atoms of this structural component, build the first time.
     *	Return a StructuralComponent is called. */
    StructuralComponent *atomList;

};

// ------- INLINE -----------
inline void StructuralComponent::addStructure(Structure *s, bool check) {
    // add the structure in the list, only if it is compatible
    if (!check || isCompatible(s)) {
        structures.push_back(s);
        // tell the structure that it is a part of this sc
        s->addStructuralComponent(this);
    }
}
inline Structure * StructuralComponent::getStructure(const unsigned int i) const {
    if (i<structures.size())
        return structures[i];
    else
        return NULL;
}
inline Structure * StructuralComponent::getStructureByIndex(const unsigned int i) {
    std::vector <Structure *>::iterator it=structures.begin();
    while (it!=structures.end() && (*it)->getIndex()!=i)
        it++;
    if (it==structures.end())
        return NULL;
    else
        return (*it);
}
inline Structure * StructuralComponent::getStructureByName(const std::string n) {
    std::vector <Structure *>::iterator it=structures.begin();
    while (it!=structures.end() && (*it)->getName()!=n)
        it++;
    if (it==structures.end())
        return NULL;
    else
        return (*it);
}
inline unsigned int StructuralComponent::getNumberOfStructures() const {
    return structures.size();
}
inline	void StructuralComponent::removeStructure(Structure *s) {
    if (s) {
        // remove it from the list
        std::vector <Structure *>::iterator it = std::find(structures.begin(), structures.end(), s);
        if (it != structures.end()) {
            structures.erase(it);
            // tell s that it is no more used by this structural component
            s->removeStructuralComponent(this);
        }
    }
}
inline bool StructuralComponent::isInstanceOf(const char *className) const {
    return (strcmp(className, "StructuralComponent")==0);
}

inline void StructuralComponent::plannedNumberOfStructures(const unsigned int size) {
    structures.reserve(size);
}

inline StructuralComponentProperties * StructuralComponent::getProperties() {
    return ((StructuralComponentProperties *)properties);
}

#endif //STRUCTURALCOMPONENT_H
