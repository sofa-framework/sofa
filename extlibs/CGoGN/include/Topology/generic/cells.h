/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2012, IGG Team, LSIIT, University of Strasbourg           *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Web site: http://cgogn.unistra.fr/                                           *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

#ifndef CELLS_H_
#define CELLS_H_

#include "Topology/generic/dart.h"

namespace CGoGN
{

/**
 * class for cellular typing
 *
 * warning to automatic conversion
 * cell -> Dart (or const Dart&) ok
 * Dart -> Cell (or const Cell&) ok
 */

class MapMono;
class MapMulti;
template <unsigned int ORBIT>
class Cell
{
    friend class MapMono;
    friend class MapMulti;
public:
    Cell(): dart() {}
    inline Cell(Dart d): dart(d) {}
    inline Cell(const Cell& c): dart(c.dart) {}
    Cell operator=(Cell c) { this->dart = c.dart; return *this; }
    inline ~Cell() {}

    inline unsigned int index() const { return dart.index ;}
    inline operator Dart() const { return dart; }

    inline bool valid() const { return !dart.isNil(); }
    inline bool operator==(Cell c) const { return dart == c.dart; }
    inline bool operator!=(Cell c) const { return dart != c.dart; }
    inline bool operator<(Cell c) const {return this->index() < c.index(); }
    static unsigned int dimension() {return ORBIT;}
    inline bool isNil() const { return dart.isNil(); }
    friend std::ostream& operator<<( std::ostream &out, const Cell<ORBIT>& fa ) { return out << fa.dart; }
    friend inline bool operator==(Cell<ORBIT> c, Dart d)  { return d == c.dart; }

    template<unsigned ORB_FROM>
    inline static Cell convertCell(const Cell<ORB_FROM>& c) {return Cell(Dart::create(c.index()));}
private:
    Dart dart;
    template<unsigned int ORB_FROM>
    Cell(const Cell<ORB_FROM>& ) ;
    template<unsigned int ORB_FROM>
    Cell operator=(const Cell<ORB_FROM> &) ;

};

typedef Cell<VERTEX> Vertex;
typedef Cell<VERTEX> VertexCell;
typedef Cell<EDGE>   Edge;
typedef Cell<EDGE>   EdgeCell;
typedef Cell<FACE>   Face;
typedef Cell<FACE>   FaceCell;
typedef Cell<VOLUME> Vol;  // not Volume because of the namespace Volume
typedef Cell<VOLUME> VolumeCell;

namespace Parallel
{
const unsigned int SIZE_BUFFER_THREAD = 8192;
}

}

#endif /* CELLS_H_ */
