/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_LOADER_GridMeshCreator_H
#define SOFA_COMPONENT_LOADER_GridMeshCreator_H
#include "config.h"

#include <sofa/core/loader/MeshLoader.h>
#include <sofa/helper/SVector.h>
namespace sofa
{

namespace component
{

namespace loader
{


/** Procedurally creates a triangular grid.
  The coordinates range from (0,0,0) to (1,1,0). They can be translated, rotated and scaled using the corresponding attributes of the parent class.

  @author François Faure, 2012
*/
class SOFA_GENERAL_LOADER_API GridMeshCreator : public sofa::core::loader::MeshLoader
{
public:

    SOFA_CLASS(GridMeshCreator,sofa::core::loader::MeshLoader);
    virtual std::string type() { return "This object is procedurally created"; }
    virtual bool canLoad() override { return true; }
    virtual bool load() override; ///< create the grid

    template <class T>
    static bool canCreate ( T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg ) { return BaseLoader::canCreate (obj, context, arg); }

    Data< defaulttype::Vec2i > resolution;  ///< Number of vertices in each direction
    Data< int > trianglePattern;            ///< 0: no triangles, 1: alternate triangles, 2: upward triangles, 3: downward triangles.

protected:
    GridMeshCreator();

    ///< index of a vertex, given its integer coordinates (between 0 and resolution) in the plane.
    unsigned vert( unsigned x, unsigned y) { return x + y*resolution.getValue()[0]; }

    // To avoid edge redundancy, we insert the edges to a set, an then dump the set. Edge (a,b) is considered equal to (b,a), so only one of them is inserted
    std::set<Edge> uniqueEdges;                                ///< edges without redundancy
    void insertUniqueEdge(unsigned a, unsigned b);             ///< insert an edge if it is not redundant
    void insertTriangle(unsigned a, unsigned b, unsigned c);   ///< insert a triangle (no reduncy checking !) and unique edges
    void insertQuad(unsigned a, unsigned b, unsigned c, unsigned d);   ///< insert a quad (no reduncy checking !) and unique edges

};




} // namespace loader

} // namespace component

} // namespace sofa

#endif
