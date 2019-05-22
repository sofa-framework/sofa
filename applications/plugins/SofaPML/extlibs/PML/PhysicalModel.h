/***************************************************************************
                              PhysicalModel.h 
                             -------------------
    begin             : Wed Aug 8 2001
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2006/10/17 14:33:22 $
    Version           : $Revision: 1.21 $
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef PHYSICALMODEL_H
#define PHYSICALMODEL_H

#include "PhysicalModelIO.h"
#include <string>
#include <vector>
#include <algorithm>
#include "AbortException.h"
#include "StructuralComponent.h" // so we can put the optimized getAtom method inline
#include "Atom.h" // so we can put the optimized getAtom method inline
#include "Cell.h" // so we can put the optimized getCell method inline

class MultiComponent;
class Component;
class Cell;
class Structure;

/** Definition of a function/method that could be called by the setProgress method.  */
typedef void (*PtrToSetProgressFunction)(const float donePercentage);

namespace std {
/** definition of a couple (=STL pair) (int , Structure *)
 * this associates a global cell/atom index to the ptr to the cell/atom that actually has this index
 */
typedef std::pair<unsigned int, Structure *> GlobalIndexStructurePair;
/** definition of the association set (=map in STL) globalIndexStructureMap.
  *  GlobalIndexStructureMap associate all the global index with their cell/atom ptr.
  *  The key is the global index, so that it is simple to retrieve a cell/atom pointer from the globalIndex
  */
typedef impmap <unsigned int, Structure *> GlobalIndexStructureMap;
/** the iterator corresponding to GlobalIndexStructureMap */
typedef impmap <unsigned int, Structure *> ::iterator GlobalIndexStructureMapIterator;
}




/** @mainpage PML library documentation
 *
 * - To learn more about PML, go to the <a href="http://www-timc.imag.fr/Emmanuel.Promayon/PML">PML/LML website</a>
 * - If you would like to help please check the \ref TODO
 * - To know the exact current PML library version use PhysicalModel::VERSION
 */

/**
  * This is the main class of this project.
  * Following a nice concept, a physical model is able to represent any kind of 3D physical model that appeared, appears or will appear on earth.
  * This include FEM meshes, spring-mass networks, phymulob etc...
  * $Revision: 1.21 $
  */
class PhysicalModel {

public:
    /** @name constructors and destructors.
     */
    //@{    
    /** Default constructor : this one just initialize everything.
      * Structures and atoms are empty.
      */
    PhysicalModel();
    /**
      * Instanciate a physical model object using a XML native format file
      * @param fileName the name of the xml file to use
      * @param pspf is a pointer to the method that will be called by the setProgress(...) method (default = NULL)
    */
    PhysicalModel(const char * fileName, PtrToSetProgressFunction pspf = NULL);
    /// destructor
    virtual ~PhysicalModel();
    /// Current PML library version 
    static const std::string VERSION;
    //@}

    /** @name general manipulation 
     */
    //@{    
    /** Return the name of the physical model */
    const std::string getName() const;
    /// set the name of the physical model
    void setName(const std::string);
    //@}

    /** @name export to files
     */
    //@{    
    /** print the physical model to an output stream in a XML format (see physicalmodel.xsd for
       * detail about the XML format).
       * By default the output is not optimized (optimized = faster loading).
       * In order to optimize, the cell and atom indexes are renumbered to be consecutive, so
       * access to cell<sub>i</sub> or atom<sub>i</sub> is done in linear time.
       * There are many reasons why you would not want to optimize the output, e.g. if you have a specific
       * cell numbering that you are using somewhere else, in a different software or so.
       * @param o the ostream to write to
       * @param opt a boolean indicating if yes or no you want the pm to optimize the output
       */
    void xmlPrint(std::ostream &o, bool opt=false);

    /**
     *  Save the geometry (atoms/cells) of this PhysicalModel
     *  in the Patran format.
     *
     *  %%% This method is usefull only for a  FEM (?) mesh.
     *  Maybe it's better to put it in a femPM class that inherits PhysicalModel???
     */
    void exportPatran(std::string filename);

    /**
     *  Save the mesh (atoms/cells) of this PhysicalModel
     *  in the Ansys format.
     *
     *  //@@@ This method is usefull only for a  FEM (?) mesh. 
     *  Maybe it's better to put it in a femPM class that inherits PhysicalModel???
     */
    void exportAnsysMesh(std::string filename);
    //@}

    /** @name component manipulations
     */
    //@{    
    /// get the total number of exclusive components
    unsigned int getNumberOfExclusiveComponents() const;
    
    /// get the total number of informative components
    unsigned int getNumberOfInformativeComponents() const;
    
    /// get the number of atoms
    unsigned int getNumberOfAtoms() const;
    
    /// get the total nr of cell in the physical model (exclusive as well as informative)
    unsigned int getNumberOfCells() const;

    /// get an exclusive component by its index in the list
    Component * getExclusiveComponent(const unsigned int) const;
            
    /// set the exclusive multi component. Becareful: the physical model takes control of this MultiComponent
    void setExclusiveComponents(MultiComponent *);

    /// get all the exclusive components
    MultiComponent * getExclusiveComponents() const;
    
    /// get all the informative components
    MultiComponent * getInformativeComponents() const;
    
    /// get all the atoms
    StructuralComponent * getAtoms() const;

    /// get an informative component by its index in the list
    Component * getInformativeComponent(const unsigned int) const;
    
    /// set the exclusive multi component. Becareful: the physical model takes control of this MultiComponent
    void setInformativeComponents(MultiComponent *);

    /** set the atom structural component. 
     * Becareful: the physical model takes control of this structural component
     * @param sc the new atom structural component
     * @param deleteOld if true, then the old atoms SC is delete (thus deleting its atoms as well)
     */
    void setAtoms(StructuralComponent *, bool deleteOld=true);

    /** Add a new atom to the atoms' structural component.
     *  It does add the atom only if it has a unique index, otherwise nothing is done.
     *  (if index is correct, then it also call the addGlobalIndexAtomPair method).
     *  @return true only if the atom was added
     */
    bool addAtom(Atom *);

    /** Get the atom that has the global index given in parameters.
     *  @param id the atom index in the physical model
     *  @return the corresponding atom or NULL if non existant (i.e. no atoms have this index)
     */
    Atom * getAtom(const unsigned int id);

    /** add or update a pair to the atom map.
     *  It does nothing if the atom already exist in the map
     *  @return true only if the atom was added
     */ 
    bool addGlobalIndexAtomPair(std::GlobalIndexStructurePair);

    /** add or update a pair to the cell map.
     *  It does nothing if the cell already exist in the map
     *  @return true only if the cell was added
     */ 
    bool addGlobalIndexCellPair(std::GlobalIndexStructurePair);

    /** get the cell that has the global index given in parameters.
      * @param id the cell index in the physical model
      * @return the corresponding cell or NULL if non existant (i.e. no cels have this index)
      */
    Cell * getCell(const unsigned int id);

    /// get a cell using its name
    Structure * getStructureByName(const std::string n);
    
    /** get a structural or multi component by its name.
     *  <b>Becareful:</b> this method never returns a cell (only a structural component or
     *  a multiple component. To get a cell, use getStructureByName(..)
     */
    Component * getComponentByName(const std::string n);
    
    /** this method is called during a long process everytime a little bit of the process is finished.
      * This method should be overloaded by the subclass to give a mean to produce a progress bar or equivalent gui/hmi stuff.
      * @param donePercentage the percentage (between 0 and 100) of the work already done
      */
    virtual void setProgress(const float donePercentage);

    /** Set the new position of an atom.
     *   This method is overloaded in ImpPhysicalModel to update the Objects3D as well.
     */
    virtual void setAtomPosition(Atom *atom, const SReal pos[3]);

    //@}

        
private:
    /** use the XML Parser/Reader to read an XML file conform to physicalmodel.dtd
    * @param n the name of the XML file
    */
    void xmlRead(const char * n);

	/** read the xml tree and call other parse methods to biuld the physicalModel. */
	bool parseTree(xmlNodePtr root);
	/** read the atom list in the xml tree and build them. */
	bool parseAtoms(xmlNodePtr atomsRoot);
	/** read the exclusive components list in the xml tree and build them. */
	bool parseComponents(xmlNodePtr exclusiveRoot, Component * father, bool isExclusive);

    /** Name of the model, read in the header section of the XML/VTK/... file.  */
    std::string name;

    /**
     * Exclusive components are the non-overlaping components : they defined all the components
     * of the physicalModel and the physicalModel could be defined by all this components.
     * exclusiveComponents could only contains StructuralComponents...
     */
    MultiComponent * exclusiveComponents;

    /**
     * Informative components could be overlaping with other components : they are extra
     * components that give information about group of cells.
     * This components are not mandatory.
     */
    MultiComponent * informativeComponents;

    /**
    * List of all the atoms : this is the basic stuff for a physicall model. The smallest entity here
    */
    StructuralComponent * atoms;

    /// Clear everything. That allows to restart an allready instanciated object from scratch
    void clear();

    /// the association couple list, which contains the direct map between the cell's global index and the cell ptr
    std::GlobalIndexStructureMap cellMap;

    /// optimized consecutive cell vector (in here <tt>optimizedCellList[i]->getIndex() == i </tt>)
    std::vector <Cell *> optimizedCellList;

    /// tell if optimizedCellList can be used
    bool cellIndexOptimized;

    /** optimize atom and cell indexes so that each order number is equal to the index
      */
    void optimizeIndexes();

    /// optimize the indexes for a given multi component (new indexing will start using the parameter)
    void optimizeIndexes(MultiComponent*, unsigned int *);

    /// the association couple list, which contains the direct map between the atom's global index and the atom ptr
    std::GlobalIndexStructureMap atomMap;

    // initialization method
    void init();

    // the progress function
    PtrToSetProgressFunction setProgressFunction;

};

// ------------------ simple inline functions ------------------
inline const std::string PhysicalModel::getName() const {
    return this->name;
}
inline void PhysicalModel::setName(const std::string n) {
    this->name = n;
}
inline MultiComponent * PhysicalModel::getExclusiveComponents() const {
    return exclusiveComponents;
}
inline MultiComponent * PhysicalModel::getInformativeComponents() const {
    return informativeComponents;
}
inline StructuralComponent * PhysicalModel::getAtoms() const {
    return atoms;
}

// ------------------ getAtom ------------------
inline Atom * PhysicalModel::getAtom(const unsigned int id) {

    // optimization: first check if the order is the structure is not the same
    // as the atom index (which is the case very often)
    Atom *quickAccessed = (Atom *) atoms->getStructure(id);
    if (quickAccessed && quickAccessed->getIndex()==id) {
        return quickAccessed;
    } else {
        // if not then check if it could be found in the map
        std::GlobalIndexStructureMapIterator mapIt; // a representation map iterator
        mapIt = atomMap.find(id);

        // search in the map, and return the correct result
        return ( (mapIt == atomMap.end()) ? NULL : (Atom *) mapIt->second );
    }
}

// ------------------ getCell ------------------
inline Cell * PhysicalModel::getCell(const unsigned int cellIndex) {
    if (cellIndexOptimized) {
        return optimizedCellList[cellIndex];
    } else {
        std::GlobalIndexStructureMapIterator mapIt; // a representation map iterator

        // check if it was find in the list
        mapIt = cellMap.find(cellIndex);

        // search in the map, and return the correct result
        return ( (mapIt == cellMap.end()) ? NULL : (Cell *) mapIt->second );
    }
}

// ------------------ getStructureByName ------------------
inline Structure * PhysicalModel::getStructureByName(const std::string n) {
    // look for structures into the global maps
    
    // look for a cell with this name
    std::GlobalIndexStructureMapIterator mapIt = cellMap.begin();
    while (mapIt != cellMap.end() && mapIt->second->getName()!=n) {
        mapIt++;
    }

    // if found returns it
    if (mapIt!=cellMap.end())
        return mapIt->second;
    
    // look now in the atoms
    mapIt = atomMap.begin();
    while (mapIt != atomMap.end() && mapIt->second->getName()!=n) {
        mapIt++;
    }

    // if found returns it
    if (mapIt!=atomMap.end())
        return mapIt->second;
        
    return NULL;
}

#endif
