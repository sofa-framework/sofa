/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETGEOMETRYALGORITHMS_H
#define SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETGEOMETRYALGORITHMS_H

#include <sofa/component/topology/QuadSetGeometryAlgorithms.h>

namespace sofa
{
namespace component
{
namespace topology
{
class HexahedronSetTopologyContainer;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::HexaID HexaID;
typedef BaseMeshTopology::Hexa Hexa;
typedef BaseMeshTopology::SeqHexas SeqHexas;
typedef BaseMeshTopology::VertexHexas VertexHexas;
typedef BaseMeshTopology::EdgeHexas EdgeHexas;
typedef BaseMeshTopology::QuadHexas QuadHexas;
typedef BaseMeshTopology::HexaEdges HexaEdges;
typedef BaseMeshTopology::HexaQuads HexaQuads;

typedef Hexa Hexahedron;
typedef HexaEdges HexahedronEdges;
typedef HexaQuads HexahedronQuads;

/**
* A class that provides geometry information on an HexahedronSet.
*/
template < class DataTypes >
class HexahedronSetGeometryAlgorithms : public QuadSetGeometryAlgorithms<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;

    HexahedronSetGeometryAlgorithms()
        : QuadSetGeometryAlgorithms<DataTypes>()
    {}

    virtual ~HexahedronSetGeometryAlgorithms() {}

    virtual void init();

    /// computes the volume of hexahedron no i and returns it
    Real computeHexahedronVolume(const unsigned int i) const;

    /// computes the hexahedron volume of all hexahedra are store in the array interface
    void computeHexahedronVolume( BasicArrayInterface<Real> &ai) const;

    /// computes the hexahedron volume  of hexahedron no i and returns it
    Real computeRestHexahedronVolume(const unsigned int i) const;

    /** \brief Write the current mesh into a msh file
    */
    void writeMSHfile(const char *filename);

private:
    HexahedronSetTopologyContainer* m_container;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
