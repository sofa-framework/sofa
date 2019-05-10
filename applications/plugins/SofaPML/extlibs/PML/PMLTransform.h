/***************************************************************************
                                   PMLTransform.h 
                             -------------------
    begin             : Mon Aug 28 2006
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2006/10/17 14:47:58 $
    Version           : $Revision: 1.1 $
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef PMLTRANSFORM_H
#define PMLTRANSFORM_H

#include "PhysicalModel.h"

#include <map>
#include <vector>
//using namespace std;

class CellProperties;




/// class facet to old and compare facet
class Facet {
  public:
    /// create a facet using size nodes and their indexes
    Facet(unsigned int size, unsigned int id[]);
        
    /// destructor
    virtual ~Facet();

    /// if it is the same (equivalent) facet, increment used (return true if equivalence)
    bool testEquivalence(unsigned int size, unsigned int id[]);
        
    /// return the corresponding PML cell
    Cell * getCell(PhysicalModel *) const;
    
    /// print on stdout
    void debug();
    
    /// get the number of time it is being used
    unsigned int getUsed() const;
    
  private:
    /// is this atom index present in this facet (no check on the order)
    bool isIn(unsigned int) const;
    
    /// the facet atom indexes
    unsigned int *id;
    
    /// nr of atoms composing the facet (3 = triangle, 4 = quad)
    unsigned int size;
    
    /// nr of times the facet is used
    unsigned int used;
};




/** PML Transform is composed by static methods 
  * It performs transformations on pml object and do a lot of useful things
  * $Revision: 1.1 $
  */
class PMLTransform {

public :
//-- elem to neighborhhod methods

	/// get the iterator on the correct atom index in the neighMap
	/// if non existant create it
	static std::map<unsigned int, Cell*>::iterator getIterator(unsigned int index);

	/// generate the neighborhoods
	static StructuralComponent * generateNeighborhood(StructuralComponent *sc);

	/// check if equivalent of already existing facet
	static void equivalent(int size, unsigned int id[]);
	
	/// generate the outside surface
	static MultiComponent * generateExternalSurface(StructuralComponent *sc);


private :
	// -------------------- Neigborhood Map ------------------------
	// associative map of all the neighboors for a given index of an atom
	static std::map<unsigned int, Cell*> neighMap;

	// -------------------- All border facets ------------------------
	/// storing all the facets
	static std::vector <Facet *> allFacets;



};

#endif
