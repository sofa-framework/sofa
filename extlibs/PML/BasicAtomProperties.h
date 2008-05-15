/***************************************************************************
                            BasicAtomProperties.h
                             -------------------
    begin             : Wed Aug 8 2001
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2004/08/11 14:05:24 $
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

#ifndef BASICATOMPROPERTIES_H
#define BASICATOMPROPERTIES_H



#include "PhysicalModelIO.h"
#include "StructureProperties.h"
#include <sofa/helper/system/config.h>

/**
  * This class is the basic Atom Properties class.
  * You should derive from this class a AtomProperties class and use it to implement your own custom stuff.
  * This is a pure virtual class.
  *
  * @author Emmanuel Promayon
  * $Revision: 1.9 $
  */
class BasicAtomProperties : public StructureProperties {
public:
    /** Default constructor : set the position to the origin, and generate an unique index */
    BasicAtomProperties(PhysicalModel *);
	/** constructor from xml node: try to read and get the parmaters from xml */
    BasicAtomProperties(PhysicalModel *, xmlNodePtr);
    /** set the position to the origin
     * @param ind an unique index
     */
    BasicAtomProperties(PhysicalModel *, const unsigned int ind);
    /** generate an unique index.
     * @param pos the initial position
     */
    BasicAtomProperties(PhysicalModel *, const SReal pos[3]);
    /** everything is given here
     * @param pos the initial position
     * @param ind an unique index
     */
    BasicAtomProperties(PhysicalModel *, const unsigned int ind, const SReal pos[3]);
    /** the destructor...
        */
    virtual ~BasicAtomProperties() {};

    /// print to an output stream in "pseudo" XML format.
    virtual void xmlPrint(std::ostream &) =0;

    /** Reinitialize the unique index to zero (usually that what you want to do when you
        * start to load a new PhysicalModel
        */
    static void resetUniqueIndex();

    /// get the position of the atom (array of 3 SReals)
    void getPosition(SReal pos[3]) const;

    /// set the position of the atom
    void setPosition(const SReal [3]);
    /// set the position of the atom
    void setPosition(const SReal,const SReal,const SReal);
        
    /** Position of the atom */
    SReal X[3];

    /// Set the temporary int property
    void setTempProp(const int);
    /// Set the temporary int property
    int getTempProp() const ;

protected:
    /// write the default xml properties (beginning)
    void beginXML(std::ostream &);
    /// write the default xml properties (end)
    void endXML(std::ostream &);

private:
    /// unique number (used to generate unique index for atoms if not given at the instanciation)
    static unsigned int maxUniqueIndex;
    /// this property is temporary and could be used by various algorithm (like connectivity builds)
    int intTempProp;
};

// inlines
inline void BasicAtomProperties::setTempProp(const int ti) {
    intTempProp = ti;
}
inline int BasicAtomProperties::getTempProp() const {
    return intTempProp;
}

inline void BasicAtomProperties::getPosition(SReal pos[3]) const {
    pos[0]=X[0]; pos[1]=X[1]; pos[2]=X[2];
}

inline void BasicAtomProperties::setPosition(const SReal pos[3]) {
    X[0]=pos[0]; X[1]=pos[1]; X[2]=pos[2];
}

inline void BasicAtomProperties::setPosition(const SReal x,const SReal y,const SReal z) {
    X[0]=x; X[1]=y; X[2]=z;
}

#endif //BasicAtomProperties_H
