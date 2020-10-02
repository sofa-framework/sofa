/***************************************************************************
                                Structure.h
                             -------------------
    begin             : Wed Aug 8 2001
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2006/10/17 14:33:22 $
    Version           : $Revision: 1.12 $
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

#ifndef STRUCTURE_H
#define STRUCTURE_H

#include "PhysicalModelIO.h"
#include <vector>
#include <algorithm> // for the remove
#include "StructureProperties.h"
class StructuralComponent;

/** Pure virtual class that represent an element of the structure.
 *  This implies that every structure could be represented in 3D and
 *  is a part of a structural component.
  * $Revision: 1.12 $
 */
class Structure {
public:
    /** Base constructor */
    Structure() : properties(NULL) {}
    /** Virtual destructor needed here as this is an abstract class (pure virtual) */
    virtual ~Structure() {}

    /** print to an output stream in "pseaudo" XML format.
     *  this method is called by the structural component that includes this structure.
       */
    virtual void xmlPrint(std::ostream &, const StructuralComponent *) = 0;

    /// pure virtual method, implemented in the child-class
    virtual bool isInstanceOf(const char *) const = 0;

    /// get the strucutre unique index (stored in its property)
    unsigned int getIndex() const;

    /** set the index.
     *  The index <b>have to be unique</b> otherwise this method
     *  has no effect. 
     *  The sub-classes method will check that this index is not in use.
     *  @return true only if the index of the structure was changed
     */
    virtual bool setIndex(const unsigned int);

    /// get the type of index
    StructureProperties::GeometricType getType() const;

    /// get the list of all the StructuralComponent that are using this structure
    std::vector <StructuralComponent *> getAllStructuralComponents();

    /// get the number of StructuralComponent that are using this structure
    unsigned int getNumberOfStructuralComponents() const;

    /// get a particular StructuralComponent that is using this structure
    StructuralComponent * getStructuralComponent(unsigned int i);

    /// add a particular StructuralComponent in the list
    void addStructuralComponent(StructuralComponent *);

    /// remove a particular StructuralComponent from the list
    void removeStructuralComponent(StructuralComponent *);

    /// set the name of the structure
    void setName(std::string);

    /// get the name of the structure
    std::string getName() const;

protected:

    /** Property of the current structure */
    StructureProperties *properties;

private:

    /// list of StructuralComponent that are using this structure
    std::vector <StructuralComponent *> mySCs;

};

// -------------------- inline ---------------------
inline std::vector <StructuralComponent *> Structure::getAllStructuralComponents() {
    return mySCs;
}
inline unsigned int Structure::getNumberOfStructuralComponents() const {
    return mySCs.size();
}
inline StructuralComponent * Structure::getStructuralComponent(unsigned int i) {
    if (i<mySCs.size())
        return mySCs[i];
    else
        return NULL;
}
inline void Structure::addStructuralComponent(StructuralComponent *sc) {
    mySCs.push_back(sc);
}

inline void Structure::removeStructuralComponent(StructuralComponent *sc) {
    std::vector <StructuralComponent *>::iterator it = std::find(mySCs.begin(), mySCs.end(), sc);
    if (it != mySCs.end())
        mySCs.erase(it);
}


#endif     //  STRUCTURE_H
