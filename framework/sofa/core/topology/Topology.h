/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_TOPOLOGY_TOPOLOGY_H
#define SOFA_CORE_TOPOLOGY_TOPOLOGY_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/helper/list.h>
#include <sofa/core/DataEngine.h>

#include <sofa/helper/fixed_array.h>
#include <iostream>
#include <sofa/helper/vector.h>
#include <stdlib.h>
#include <string>


namespace sofa
{

namespace core
{

namespace topology
{

using namespace sofa::helper;

/// The enumeration used to give unique identifiers to Topological objects.
enum TopologyObjectType
{
    POINT,
    EDGE,
    TRIANGLE,
    QUAD,
    TETRAHEDRON,
    HEXAHEDRON
};



class Topology : public virtual core::objectmodel::BaseObject
{
public:
    /// Topology global typedefs
    //typedef int index_type;
    typedef unsigned int index_type;
    enum { InvalidID = (unsigned)-1 };
    typedef index_type	        	    PointID;
    typedef index_type          		    EdgeID;
    typedef index_type                          TriangleID;
    typedef index_type                 	    QuadID;
    typedef index_type	                    TetraID;
    typedef index_type	                    TetrahedronID;
    typedef index_type	                    HexaID;
    typedef index_type	                    HexahedronID;


    typedef vector<index_type>                  SetIndex;
    typedef vector<index_type>                  SetIndices;

    typedef PointID                             Point;
    // in the following types, we use wrapper classes to have different types for each element, otherwise Quad and Tetrahedron would be the same
    class Edge : public fixed_array<PointID,2>
    {
    public:
        Edge() {}
        Edge(PointID a, PointID b) : fixed_array<PointID,2>(a,b) {}
    };
    class Triangle : public fixed_array<PointID,3>
    {
    public:
        Triangle() {}
        Triangle(PointID a, PointID b, PointID c) : fixed_array<PointID,3>(a,b,c) {}
    };
    class Quad : public fixed_array<PointID,4>
    {
    public:
        Quad() {}
        Quad(PointID a, PointID b, PointID c, PointID d) : fixed_array<PointID,4>(a,b,c,d) {}
    };
    class Tetrahedron : public fixed_array<PointID,4>
    {
    public:
        Tetrahedron() {}
        Tetrahedron(PointID a, PointID b, PointID c, PointID d) : fixed_array<PointID,4>(a,b,c,d) {}
    };
    typedef Tetrahedron                         Tetra;
    class Hexahedron : public fixed_array<PointID,8>
    {
    public:
        Hexahedron() {}
        Hexahedron(PointID a, PointID b, PointID c, PointID d, PointID e, PointID f, PointID g, PointID h) : fixed_array<PointID,8>(a,b,c,d,e,f,g,h) {}
    };
    typedef Hexahedron                          Hexa;

    SOFA_CLASS(Topology, core::objectmodel::BaseObject);
protected:
    Topology():BaseObject() {}
    virtual ~Topology()
    {}
public:
    // Access to embedded position information (in case the topology is a regular grid for instance)
    // This is not very clean and is quit slow but it should only be used during initialization

    virtual bool hasPos() const { return false; }
    virtual int getNbPoints() const { return 0; }
    virtual void setNbPoints(int /*n*/) {}
    virtual double getPX(int /*i*/) const { return 0.0; }
    virtual double getPY(int /*i*/) const { return 0.0; }
    virtual double getPZ(int /*i*/) const { return 0.0; }
};

} // namespace topology

} // namespace core

} // namespace sofa

#endif
