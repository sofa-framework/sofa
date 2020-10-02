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
#ifndef SOFA_COMPONENT_TOPOLOGY_BEZIER2MESHTOPOLOGICALMAPPING_H
#define SOFA_COMPONENT_TOPOLOGY_BEZIER2MESHTOPOLOGICALMAPPING_H
#include "config.h"


#include <sofa/core/topology/TopologicalMapping.h>
#include <sofa/core/topology/Topology.h>

#include <sofa/defaulttype/Vec.h>
#include <map>
#include <set>



namespace sofa { namespace component { namespace mapping { template<typename  D, typename E> class Bezier2MeshMechanicalMapping; } } }


namespace sofa
{
namespace component
{
namespace topology
{
/**
 * This class, called Bezier2MeshTopologicalMapping, is a specific implementation of the interface TopologicalMapping where :
 *
 * INPUT TOPOLOGY = A BezierTetrahedronSetTopology or BezierTriangleSetTopology as a tesselated version of the input mesh 
  * OUTPUT TOPOLOGY = a Tetrahedral or triangular mesh interpolated with a given degree of tesselation from its Bezier mesh
 *
 * This Topological mapping handles the specic input topology of Bezier elements and is made more efficient by using precomputations of maps
 *
 * Bezier2MeshTopologicalMapping class is templated by the pair (INPUT TOPOLOGY, OUTPUT TOPOLOGY)
 *
*/

    class SOFA_TOPOLOGY_MAPPING_API Bezier2MeshTopologicalMapping : public sofa::core::topology::TopologicalMapping
{
public:
    SOFA_CLASS(Bezier2MeshTopologicalMapping,sofa::core::topology::TopologicalMapping);
	template<typename D, typename E>  friend class sofa::component::mapping::Bezier2MeshMechanicalMapping;

protected:
    /** \brief Constructor.
     *
     */
    Bezier2MeshTopologicalMapping ();

    /** \brief Destructor.
     *
         * Does nothing.
         */
    virtual ~Bezier2MeshTopologicalMapping();
public:
    /** \brief Initializes the target BaseTopology from the source BaseTopology.
     */
    virtual void init();

    /// create a number of subtetrahedra depending on the level of tesselation. Degree is the number of times an edge will be split. 
	Data < unsigned int > d_tesselationTetrahedronDegree;
	/// create a number of subtriangles depending on the level of tesselation. Degree is the number of times an edge will be split. 
	Data < unsigned int > d_tesselationTriangleDegree;
protected:
	// local indexing of points inside tessellated triangles
	sofa::helper::vector<sofa::defaulttype::Vec<3,unsigned char > > tesselatedTriangleIndices; 
	sofa::helper::vector<sofa::core::topology::Topology::Edge > edgeTriangleArray; 
	sofa::helper::vector<sofa::helper::vector<size_t> > bezierEdgeArray; 
	/// for each macro triangle set the index of tesselated points inside that triangle (used for nmal computation)
	sofa::helper::vector< sofa::helper::vector<size_t> > globalIndexTesselatedBezierTriangleArray; 
	sofa::helper::vector<size_t> local2GlobalBezierVertexArray;
	sofa::helper::vector<int> global2LocalBezierVertexArray;
	// the number of points in the output triangulation
	size_t nbPoints;
public :
	/// Method called at each topological changes propagation which comes from the INPUT topology to adapt the OUTPUT topology :
	virtual void updateTopologicalMappingTopDown();

};



} // namespace topology
} // namespace component
} // namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_MESH2BEZIERTOPOLOGICALMAPPING_H
