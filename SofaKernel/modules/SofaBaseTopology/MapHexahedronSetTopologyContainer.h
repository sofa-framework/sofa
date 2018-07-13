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

#ifndef MAPHEXAHEDRONSETTOPOLOGYCONTAINER_H
#define MAPHEXAHEDRONSETTOPOLOGYCONTAINER_H

#include <SofaBaseTopology/HexahedronSetTopologyContainer.h>
#include <SofaBaseTopology/VolumeTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace topology
{

class SOFA_BASE_TOPOLOGY_API MaHexahedronSetTopologyContainer : public HexahedronSetTopologyContainer
{
	friend class QuadSetTopologyModifier;
public:
	SOFA_CLASS(MaHexahedronSetTopologyContainer,HexahedronSetTopologyContainer);
	template<typename T>
	using Attribute_T = core::topology::MapTopology::Attribute_T<T>;
	using Orbit = VolumeTopologyContainer::Orbit;
	template <typename T, Orbit ORBIT>
	using Attribute = VolumeTopologyContainer::Attribute<T,ORBIT>;
	using Vertex = VolumeTopologyContainer::Vertex;
	using Edge = VolumeTopologyContainer::Edge;
	using Face = VolumeTopologyContainer::Face;
	using Volume = VolumeTopologyContainer::Volume;

	MaHexahedronSetTopologyContainer();
	virtual ~MaHexahedronSetTopologyContainer() override;


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
	virtual const SeqQuads&getQuads() override;
	virtual const SeqHexahedra&getHexahedra() override;
	virtual int getNbEdges() override;
	virtual int getNbQuads() override;
	virtual int getNbHexahedra() override;
	virtual int getNbHexas() override;
	virtual const EdgesAroundVertex&getEdgesAroundVertex(PointID i) override;
	virtual const EdgesInQuad&getEdgesInQuad(QuadID i) override;
	virtual const EdgesInHexahedron&getEdgesInHexahedron(HexaID i) override;
	virtual const QuadsAroundVertex&getQuadsAroundVertex(PointID i) override;
	virtual const QuadsAroundEdge&getQuadsAroundEdge(EdgeID i) override;
	virtual const QuadsInHexahedron&getQuadsInHexahedron(HexaID i) override;
	virtual const HexahedraAroundVertex&getHexahedraAroundVertex(PointID i) override;
	virtual const HexahedraAroundEdge&getHexahedraAroundEdge(EdgeID i) override;
	virtual const HexahedraAroundQuad&getHexahedraAroundQuad(QuadID i) override;
	virtual const VerticesAroundVertex getVerticesAroundVertex(PointID i) override;


	virtual const sofa::helper::vector<index_type> getElementAroundElement(index_type elem) override;
	virtual const sofa::helper::vector<index_type> getElementAroundElements(sofa::helper::vector<index_type> elems) override;

	virtual void clear() override;
	virtual void addPoint(SReal px, SReal py, SReal pz) override;
	virtual void addEdge(int a, int b) override;
	virtual void addQuad(int a, int b, int c, int d) override;
	virtual void addHexa(int a, int b, int c, int d, int e, int f, int g, int h) override;
	virtual bool checkConnexity() override;
	virtual unsigned int getNumberOfConnectedComponent() override;
	virtual const sofa::helper::vector<index_type> getConnectedElement(index_type elem) override;
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

#endif // MAPHEXAHEDRONSETTOPOLOGYCONTAINER_H
