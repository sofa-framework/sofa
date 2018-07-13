/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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

#ifndef MAPTETRAHEDRONSETTOPOLOGYCONTAINER_H
#define MAPTETRAHEDRONSETTOPOLOGYCONTAINER_H

#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>
#include <SofaBaseTopology/VolumeTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace topology
{

class SOFA_BASE_TOPOLOGY_API MapTetrahedronSetTopologyContainer : public TetrahedronSetTopologyContainer
{
	friend class TriangleSetTopologyModifier;
public:
	SOFA_CLASS(MapTetrahedronSetTopologyContainer,TetrahedronSetTopologyContainer);
	template<typename T>
	using Attribute_T = core::topology::MapTopology::Attribute_T<T>;
	using Orbit = VolumeTopologyContainer::Orbit;
	template <typename T, Orbit ORBIT>
	using Attribute = VolumeTopologyContainer::Attribute<T,ORBIT>;
	using Vertex = VolumeTopologyContainer::Vertex;
	using Edge = VolumeTopologyContainer::Edge;
	using Face = VolumeTopologyContainer::Face;
	using Volume = VolumeTopologyContainer::Volume;

	MapTetrahedronSetTopologyContainer();
	virtual ~MapTetrahedronSetTopologyContainer() override;


	// Topology interface
public:
	virtual int getNbPoints() const override;

	// BaseObject interface
public:
	virtual void init() override;
	virtual void bwdInit() override;
	virtual void reinit() override;
	virtual void reset() override;
	virtual void cleanup() override;
	virtual void draw(const core::visual::VisualParams*) override;

	// BaseMeshTopology interface
public:
	virtual bool load(const char* filename) override;
	virtual const SeqEdges&getEdges() override;
	virtual const SeqTriangles&getTriangles() override;
	virtual const SeqTetrahedra&getTetrahedra() override;
	virtual int getNbEdges() override;
	virtual int getNbTriangles() override;
	virtual int getNbTetrahedra() override;
//	virtual const core::topology::Topology::Edge getEdge(EdgeID i) override;
//	virtual const Triangle getTriangle(TriangleID i) override;
//	virtual const Tetra getTetrahedron(TetraID i) override;
	virtual int getNbTetras() override;
//	virtual Tetra getTetra(TetraID i) override;
//	virtual const SeqTetrahedra&getTetras() override;
	virtual const EdgesAroundVertex&getEdgesAroundVertex(PointID i) override;
	virtual const EdgesInTriangle&getEdgesInTriangle(TriangleID i) override;
	virtual const EdgesInTetrahedron&getEdgesInTetrahedron(TetraID i) override;
	virtual const TrianglesAroundVertex&getTrianglesAroundVertex(PointID i) override;
	virtual const TrianglesAroundEdge&getTrianglesAroundEdge(EdgeID i) override;
	virtual const TrianglesInTetrahedron&getTrianglesInTetrahedron(TetraID i) override;
	virtual const TetrahedraAroundVertex&getTetrahedraAroundVertex(PointID i) override;
	virtual const TetrahedraAroundEdge&getTetrahedraAroundEdge(EdgeID i) override;
	virtual const TetrahedraAroundTriangle&getTetrahedraAroundTriangle(TriangleID i) override;
	virtual const VerticesAroundVertex getVerticesAroundVertex(PointID i) override;


	virtual const sofa::helper::vector<index_type> getElementAroundElement(index_type elem) override;
	virtual const sofa::helper::vector<index_type> getElementAroundElements(sofa::helper::vector<index_type> elems) override;

	virtual void clear() override;
	virtual void addPoint(SReal px, SReal py, SReal pz) override;
	virtual void addEdge(int a, int b) override;
	virtual void addTriangle(int a, int b, int c) override;
	virtual void addTetra(int a, int b, int c, int d) override;
	virtual bool checkConnexity() override;
	virtual unsigned int getNumberOfConnectedComponent() override;
	virtual const sofa::helper::vector<index_type> getConnectedElement(index_type elem) override;
	virtual void reOrientateTriangle(TriangleID id) override;
	virtual const sofa::helper::vector<TriangleID>&getTrianglesOnBorder() override;
	virtual const sofa::helper::vector<EdgeID>&getEdgesOnBorder() override;
	virtual const sofa::helper::vector<PointID>&getPointsOnBorder() override;

	// PointSetTopologyContainer interface
public:
	virtual unsigned int getNumberOfElements() const override;
	virtual bool checkTopology() const override;

private:
	VolumeTopologyContainer* map_;
};

} // namespace topology
} // namespace component
} // namespace sofa

#endif // MAPTETRAHEDRONSETTOPOLOGYCONTAINER_H
