/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_MESH2BEZIERTOPOLOGICALMAPPING_H
#define SOFA_COMPONENT_TOPOLOGY_MESH2BEZIERTOPOLOGICALMAPPING_H
#include "config.h"

#include <SofaTopologyMapping/Mesh2PointTopologicalMapping.h>



namespace sofa
{
namespace component
{
namespace topology
{

/**
 * This class, called Mesh2BezierTopologicalMapping, is a specific implementation of the interface TopologicalMapping where :
 *
 * INPUT TOPOLOGY = any Tetrahedral or triangular MeshTopology
 * OUTPUT TOPOLOGY = A BezierTetrahedronSetTopology or BezierTriangleSetTopology as a tesselated version of the input mesh 
 *
 * This Topological mapping is a specific implementation of the Mesh2PointTopologicalMapping with a small overhead
 *
 * Mesh2BezierTopologicalMapping class is templated by the pair (INPUT TOPOLOGY, OUTPUT TOPOLOGY)
 *
*/

    class SOFA_TOPOLOGY_MAPPING_API Mesh2BezierTopologicalMapping : public sofa::component::topology::Mesh2PointTopologicalMapping
{
public:
    SOFA_CLASS(Mesh2BezierTopologicalMapping,sofa::component::topology::Mesh2PointTopologicalMapping);
protected:
    /** \brief Constructor.
     *
     */
    Mesh2BezierTopologicalMapping ();

    /** \brief Destructor.
     *
         * Does nothing.
         */
    virtual ~Mesh2BezierTopologicalMapping() {};
public:
    /** \brief Initializes the target BaseTopology from the source BaseTopology.
     */
    virtual void init();

    /// Fills pointBaryCoords, edgeBaryCoords, triangleBaryCoords and tetraBaryCoords so as to create a Bezier Tetrahedron mesh of a given order
	Data < unsigned int > bezierTetrahedronDegree;
	/// Fills pointBaryCoords, edgeBaryCoords, triangleBaryCoords so as to create a Bezier Triangle mesh of a given order
	Data < unsigned int > bezierTriangleDegree;
};

} // namespace topology
} // namespace component
} // namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_MESH2BEZIERTOPOLOGICALMAPPING_H
