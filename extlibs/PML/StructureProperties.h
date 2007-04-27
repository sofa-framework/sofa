/***************************************************************************
                            StructureProperties.h
                             -------------------
    begin             : Wed Aug 8 2001
    copyright         : (C) 2001 TIMC (Emmanuel Promayon, Matthieu Chabanas)
    email             : Emmanuel.Promayon@imag.fr
    Date              : $Date: 2007/03/26 07:20:54 $
    Version           : $Revision: 1.15 $
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef STRUCTUREPROPERTIES_H
#define STRUCTUREPROPERTIES_H

#include "PhysicalModelIO.h"
#include "Properties.h"


/** Describes the properties common to all structures
 *
 * $Revision: 1.15 $
 */
class StructureProperties : public Properties {
public:
    /** Geometric type gives information about which kind of geometric representation
    * is the structure.
    * For 3D geometric shapes, atoms have to be given in a proper order:
    *
    * <pre>
    * TRIANGLE
    *       2           lines:
    *     /  \          0,1
    *    /    \         0,2
    *   0------1        1,2
    *
    * QUAD
    *   3--------2      lines:
    *   |        |      0,1
    *   |        |      1,2
    *   |        |      2,3
    *   0--------1      3,0
    *
    * TETRAHEDRON
    *       3             
    *     /| \          facets (triangles):      lines:
    *    / |  \         0,1,2 (base)             0,1
    *   1..|...\ 2      0,2,3                    0,2
    *   \  |   /        0,3,1                    0,3
    *    \ |  /         2,1,3                    1,2
    *     \|/                                    1,3
    *      0                                     2,3 
    *
    * WEDGE
    *      1-------------4       facets (quad):   facets (triangles):     lines:
    *     /\           . \       2,5,4,1          0,2,1                   0,1      2,5
    *    /  \         /   \      0,1,4,3          3,4,5                   0,2      3,4
    *   0- - \ - - - 3     \     2,0,3,5                                  1,2      4,5
    *     \   \         \   \                                             0,3      5,3
    *       \ 2-----------\--5                                            1,4
    *                    
    * HEXAHEDRON
    *      2-------------6       facets (quad):         lines:
    *     / \           . \      0,3,2,1                0,1     6,7
    *    /   \         /   \     0,4,7,3                1,2     7,4
    *   1- - -\ - - - 5     \    0,1,5,4                2,3     0,4
    *   \     3-------------7    3,7,6,2                3,0     1,5
    *    \   /         \   /     1,2,6,5                4,5     2,6
    *     \ /           . /      4,5,6,7                5,6     3,7
    *      0-------------4
    *
    * </pre>
    */
    enum GeometricType {
        INVALID, /**< invalid geometry type */
        ATOM, /**< the structure is an atom, and hence should be represented by a single point */
        LINE, /**< the structure is a simple line, i.e it must be a cell composed of only 2 atoms */
        TRIANGLE, /**< the structure is a triangle, i.e it must be a cell composed of 3 atoms */
        QUAD, /**< the structure is a quad, i.e it must be a cell composed of 4 atoms */
        TETRAHEDRON, /**< the structure is a tetrahedron, it must be a cell and have sub-structures that are atoms */
        WEDGE, /**< the structure is a wedge (like the Pink Floyd's "Dark Side Of the Moon" prism), it must be a cell and have sub-structures that are atoms */
        HEXAHEDRON, /**< the structure is a hexahedron, it must be a cell and have sub-structures that are atoms */
        POLY_LINE, /**< the structure is a polyline, i.e it must be a cell and the order of the atom in the cell are arranged along a line */
        POLY_VERTEX, /**< the structure is a poly vertex, i.e it must be a cell and it is a point clouds */
    };

    /// return the enum corresponding to this string
    static GeometricType toType(const std::string);

    /// return the string equivalent to this geometric type
    static std::string toString(const GeometricType);

    /** the only default constructor : type must be set */
    StructureProperties(PhysicalModel *, const GeometricType);
    virtual ~StructureProperties() {}

    /// Set the force type
    void setType(const GeometricType t);

    /// Return the type of force
    GeometricType getType() const;

    /** print to an output stream in "pseaudo" XML format.
    */
    void xmlPrint(std::ostream &) const;

    /// return the unique index in the global structure
    unsigned int getIndex() const;

    /// set the index (BECAREFUL: it MUST be unique !!!)
    void setIndex(const unsigned int);

private:
    /** The geometric type,
    	* @see StructureProperties::GeometricType
    	*/
    GeometricType type;

protected:
    /** unique index in the global structure */
    unsigned int index;

};

// ---------------- inlines -------------------
inline StructureProperties::GeometricType StructureProperties::getType() const {
    return type;
}
inline void StructureProperties::setType(const StructureProperties::GeometricType t) {
    type = t;
}
inline unsigned int StructureProperties::getIndex() const {
    return index;
}
inline void StructureProperties::setIndex(const unsigned int newIndex) {
    index = newIndex;
}

#endif //STRUCTUREPROPERTIES_H
