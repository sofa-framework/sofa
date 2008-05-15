/***************************************************************************
                                   Atom.h 
                             -------------------
    begin             : Wed Aug 8 2001
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2006/10/17 14:33:21 $
    Version           : $Revision: 1.11 $
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef ATOM_H
#define ATOM_H


#include "Structure.h"
#include "AtomProperties.h"
#include <RenderingMode.h>  
#include <cstring>
#include <sofa/helper/system/config.h>

/** An atom has an unique index in the physical model object, a 3D position, and different basic properties.
  * It is the most basic structure composing a physical model.
  * It is on an atoms that the forces and loads could be applied in order to generate dynamics.
  * $Revision: 1.11 $
  */
class Atom : public Structure {
public:
    /** Default constructor : set the position to the origin, generate a unique index */
    Atom(PhysicalModel *);
	/** constructor from xml node: try to read and get the parmaters from xml */
    Atom(PhysicalModel *, xmlNodePtr);
    /** constructor : generate a unique index
     * @param pos the initial position of the created atom (array of 3 SReal)
     */
    Atom(PhysicalModel *, const SReal pos[3]);
    /** set the position to the origin
    * @param ind give the unique index
    */
    Atom(PhysicalModel *, const unsigned int ind);
    /** constructor : generate a unique index
    * @param ind give the unique index
     * @param pos the initial position of the created atom (array of 3 SReal)
     */
    Atom(PhysicalModel *, const unsigned int ind, const SReal pos[3]);
    /** std destructor
    */
    ~Atom();

    /** print to an output stream in "pseaudo" XML format.
       */
    void xmlPrint(std::ostream &, const StructuralComponent *);

    /// get the position of the atom (array of 3 SReals)
    void getPosition(SReal pos[3]) const;

    /// set the position of the atom
    void setPosition(const SReal [3]);
    /// set the position of the atom
    void setPosition(const SReal ,const SReal ,const SReal );

    /** set the index.
     *  The index <b>have to be unique</b> otherwise this method
     *  has no effect. 
     *  The sub-classes method will check that this index is not in use.
     *  @return true only if the index of the structure was changed
     */
    virtual bool setIndex(const unsigned int);

    /// return true only if the parameter is equal to "Atom"
    virtual bool isInstanceOf(const char *) const;

    /// Get a ptr to the AtomProperties
    AtomProperties * getProperties() const;
};

// -------------------- inline ---------------------
inline void Atom::getPosition(SReal p[3]) const {
    return getProperties()->getPosition(p);
}

inline void Atom::setPosition(const SReal pos[3])  {
    getProperties()->setPosition(pos);    
}
inline void Atom::setPosition(const SReal x, const SReal y,const SReal z)  {
    getProperties()->setPosition(x,y,z);    
}

inline bool Atom::isInstanceOf(const char *className) const {
    return (strcmp(className, "Atom")==0);
}
inline AtomProperties * Atom::getProperties() const {
    return (AtomProperties *) properties;
}

#endif //ATOM_H
