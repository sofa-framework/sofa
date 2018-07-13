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

#ifndef MAPTRIANGLESETTOPOLOGYCONTAINER_H
#define MAPTRIANGLESETTOPOLOGYCONTAINER_H

#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
#include <SofaBaseTopology/SurfaceTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace topology
{

class SOFA_BASE_TOPOLOGY_API MapTriangleSetTopologyContainer : public TriangleSetTopologyContainer
{
    friend class TriangleSetTopologyModifier;
public:
	SOFA_CLASS(MapTriangleSetTopologyContainer,TriangleSetTopologyContainer);
	template<typename T>
	using Attribute_T = core::topology::MapTopology::Attribute_T<T>;
	using Orbit = SurfaceTopologyContainer::Orbit;
	template <typename T, Orbit ORBIT>
	using Attribute = SurfaceTopologyContainer::Attribute<T,ORBIT>;
	using Vertex = SurfaceTopologyContainer::Vertex;
	using Edge = SurfaceTopologyContainer::Edge;
	using Face = SurfaceTopologyContainer::Face;

	MapTriangleSetTopologyContainer();
	virtual ~MapTriangleSetTopologyContainer() override;

    // BaseObject interface
public:
    virtual void init() override;
    virtual void bwdInit() override;
    virtual void reinit() override;
    virtual void reset() override;
    virtual void cleanup() override;

    // BaseMeshTopology interface
public:
    virtual const SeqEdges&getEdges() override;
    virtual const SeqTriangles&getTriangles() override;

    virtual int getNbEdges() override;
    virtual int getNbTriangles() override;
    virtual const core::topology::Topology::Edge getEdge(EdgeID i) override;
    virtual const Triangle getTriangle(TriangleID i) override;
    virtual const EdgesAroundVertex& getEdgesAroundVertex(PointID i) override;
    virtual const EdgesInTriangle& getEdgesInTriangle(TriangleID i) override;
    virtual const TrianglesAroundVertex& getTrianglesAroundVertex(PointID i) override;
    virtual const TrianglesAroundEdge& getTrianglesAroundEdge(EdgeID i) override;
//    virtual const VerticesAroundVertex getVerticesAroundVertex(PointID i) override;
//    virtual int getEdgeIndex(PointID v1, PointID v2) override;
//    virtual int getTriangleIndex(PointID v1, PointID v2, PointID v3) override;

//    virtual int getVertexIndexInTriangle(const Triangle& t, PointID vertexIndex) const override;
//    virtual int getEdgeIndexInTriangle(const EdgesInTriangle& t, EdgeID edgeIndex) const override;

    virtual void clear() override;
    virtual void addEdge(int a, int b) override;
    virtual void addTriangle(int a, int b, int c) override;

    virtual bool checkConnexity() override;
    virtual unsigned int getNumberOfConnectedComponent() override;
    virtual int getRevision() const override;
    virtual void reOrientateTriangle(TriangleID id) override;
    virtual const sofa::helper::vector<TriangleID>&getTrianglesOnBorder() override;
    virtual const sofa::helper::vector<EdgeID>&getEdgesOnBorder() override;
    virtual const sofa::helper::vector<PointID>&getPointsOnBorder() override;

    // TopologyContainer interface
protected:
    virtual void updateTopologyEngineGraph() override;

    // PointSetTopologyContainer interface
public:
    virtual unsigned int getNumberOfElements() const override;
    virtual bool checkTopology() const override;

    // EdgeSetTopologyContainer interface
protected:
    virtual void createEdgeSetArray() override;

    // TriangleSetTopologyContainer interface
public:
    virtual const VecTriangleID getConnectedElement(TriangleID elem) override;
    virtual const VecTriangleID getElementAroundElement(TriangleID elem) override;
    virtual const VecTriangleID getElementAroundElements(VecTriangleID elems) override;

protected:
    virtual void createTriangleSetArray() override;
    virtual void createEdgesInTriangleArray() override;
    virtual void createTrianglesAroundVertexArray() override;
    virtual void createTrianglesAroundEdgeArray() override;
    virtual TrianglesAroundVertex&getTrianglesAroundVertexForModification(const PointID vertexIndex) override;
    virtual TrianglesAroundEdge&getTrianglesAroundEdgeForModification(const EdgeID edgeIndex) override;

private:
	SurfaceTopologyContainer* map_;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif // MAPTRIANGLESETTOPOLOGYCONTAINER_H
