/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/component/topology/container/grid/GridTopology.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/type/Vec.h>

namespace sofa::component::topology::container::grid
{

int GridTopologyClass = core::RegisterObject("Base class fo a regular grid in 3D")
        .addAlias("Grid")
        .add< GridTopology >()
        ;


GridTopology::GridUpdate::GridUpdate(GridTopology *t):
    m_topology(t)
{
    addInput(&t->d_n);
    addOutput(&t->seqEdges);
    addOutput(&t->seqQuads);
    addOutput(&t->seqHexahedra);
    setDirtyValue();
}

void GridTopology::GridUpdate::doUpdate()
{
    if (m_topology->d_computeHexaList.getValue())
        updateHexas();

    if (m_topology->d_computeQuadList.getValue())
        updateQuads();

    if (m_topology->d_computeTriangleList.getValue())
        updateTriangles();

    if (m_topology->d_computeEdgeList.getValue())
        updateEdges();
}

void GridTopology::parse(core::objectmodel::BaseObjectDescription* arg)
{
    this->MeshTopology::parse(arg);

    if (arg->getAttribute("nx")!=nullptr && arg->getAttribute("ny")!=nullptr && arg->getAttribute("nz")!=nullptr )
    {
        int nx = arg->getAttributeAsInt("nx", d_n.getValue().x());
        int ny = arg->getAttributeAsInt("ny", d_n.getValue().y());
        int nz = arg->getAttributeAsInt("nz", d_n.getValue().z());
        d_n.setValue(type::Vec3i(nx,ny,nz));
    }

    this->setNbGridPoints();
}

Size GridTopology::getNbHexahedra()
{
    const auto n = d_n.getValue();
    return (n[0] - 1) * (n[1] - 1) * (n[2] - 1);
}


void GridTopology::GridUpdate::updateEdges()
{
    SeqEdges& edges = *m_topology->seqEdges.beginWriteOnly();
    edges.clear();
    const SeqTriangles& triangles = m_topology->seqTriangles.getValue();
    if (triangles.empty()) // if has triangles will create edges using triangles, otherwise will use the quads from the grid
    {
        const type::Vec3i& n = m_topology->d_n.getValue();
        edges.reserve((n[0] - 1)*n[1] * n[2] +
            n[0] * (n[1] - 1)*n[2] +
            n[0] * n[1] * (n[2] - 1));
        // lines along X
        for (int z = 0; z<n[2]; z++)
            for (int y = 0; y<n[1]; y++)
                for (int x = 0; x<n[0] - 1; x++)
                    edges.push_back(Edge(m_topology->point(x, y, z), m_topology->point(x + 1, y, z)));
        // lines along Y
        for (int z = 0; z<n[2]; z++)
            for (int y = 0; y<n[1] - 1; y++)
                for (int x = 0; x<n[0]; x++)
                    edges.push_back(Edge(m_topology->point(x, y, z), m_topology->point(x, y + 1, z)));
        // lines along Z
        for (int z = 0; z<n[2] - 1; z++)
            for (int y = 0; y<n[1]; y++)
                for (int x = 0; x<n[0]; x++)
                    edges.push_back(Edge(m_topology->point(x, y, z), m_topology->point(x, y, z + 1)));
    }
    else
    {
        // Similar algo as createEdgeSetArray in TriangleSetTopologyContainer
        // create a temporary map to find redundant edges
        std::map<Edge, EdgeID> edgeMap;
        for (size_t i = 0; i<triangles.size(); ++i)
        {
            const Triangle &t = triangles[i];
            for (unsigned int j = 0; j<3; ++j)
            {
                const PointID v1 = t[(j + 1) % 3];
                const PointID v2 = t[(j + 2) % 3];

                // sort vertices in lexicographic order
                const Edge e = ((v1<v2) ? Edge(v1, v2) : Edge(v2, v1));

                if (edgeMap.find(e) == edgeMap.end())
                {
                    // edge not in edgeMap so create a new one
                    const size_t edgeIndex = edgeMap.size();
                    edgeMap[e] = (EdgeID)edgeIndex;
                    //m_edge.push_back(e); Changed to have oriented edges on the border of the triangulation
                    edges.push_back(Edge(v1, v2));
                }
            }
        }
    }

    m_topology->seqEdges.endEdit();
}

void GridTopology::GridUpdate::updateTriangles()
{
    // need quads to create the triangulation
    if (m_topology->seqQuads.getValue().empty())
        updateQuads();

    // base on quads
    const SeqQuads& quads = m_topology->seqQuads.getValue();
    SeqTriangles& triangles = *m_topology->seqTriangles.beginWriteOnly();
    triangles.clear();
    triangles.reserve(quads.size()*2);

    for (unsigned int i=0; i<quads.size(); ++i)
    {
        triangles.push_back(Triangle(quads[i][0], quads[i][1], quads[i][2]));
        triangles.push_back(Triangle(quads[i][0], quads[i][2], quads[i][3]));
    }

    m_topology->seqTriangles.endEdit();
}

void GridTopology::GridUpdate::updateQuads()
{
    SeqQuads& quads = *m_topology->seqQuads.beginWriteOnly();
    const type::Vec3i& n = m_topology->d_n.getValue();
    quads.clear();
    quads.reserve((n[0]-1)*(n[1]-1)*n[2]+(n[0]-1)*n[1]*(n[2]-1)+n[0]*(n[1]-1)*(n[2]-1));
    // quads along XY plane
    for (int z=0; z<n[2]; z++)
        for (int y=0; y<n[1]-1; y++)
            for (int x=0; x<n[0]-1; x++)
                quads.push_back(Quad(m_topology->point(x,y,z),
                        m_topology->point(x+1,y,z),
                        m_topology->point(x+1,y+1,z),
                        m_topology->point(x,y+1,z)));
    // quads along XZ plane
    for (int z=0; z<n[2]-1; z++)
        for (int y=0; y<n[1]; y++)
            for (int x=0; x<n[0]-1; x++)
                quads.push_back(Quad(m_topology->point(x,y,z),
                        m_topology->point(x+1,y,z),
                        m_topology->point(x+1,y,z+1),
                        m_topology->point(x,y,z+1)));
    // quads along YZ plane
    for (int z=0; z<n[2]-1; z++)
        for (int y=0; y<n[1]-1; y++)
            for (int x=0; x<n[0]; x++)
                quads.push_back(Quad(m_topology->point(x,y,z),
                        m_topology->point(x,y+1,z),
                        m_topology->point(x,y+1,z+1),
                        m_topology->point(x,y,z+1)));

    m_topology->seqQuads.endEdit();
}

void GridTopology::GridUpdate::updateHexas()
{
    SeqHexahedra& hexahedra = *m_topology->seqHexahedra.beginWriteOnly();
    const type::Vec3i& n = m_topology->d_n.getValue();
    hexahedra.clear();
    hexahedra.reserve((n[0]-1)*(n[1]-1)*(n[2]-1));
    for (int z=0; z<n[2]-1; z++)
        for (int y=0; y<n[1]-1; y++)
            for (int x=0; x<n[0]-1; x++)
                hexahedra.push_back(Hexa(m_topology->point(x  ,y  ,z  ),m_topology->point(x+1,y  ,z  ),
                        m_topology->point(x+1,y+1,z  ),m_topology->point(x  ,y+1,z  ),
                        m_topology->point(x  ,y  ,z+1),m_topology->point(x+1,y  ,z+1),
                        m_topology->point(x+1,y+1,z+1),m_topology->point(x  ,y+1,z+1)));

    m_topology->seqHexahedra.endEdit();
}

/// To avoid duplicating the code in the different variants of the constructor
/// this object is using the delegating constructor feature of c++ x11.
/// The following constructor is "chained" by the other constructors to
/// defined only one the member initialization.
GridTopology::GridTopology()
    : d_n(initData(&d_n,type::Vec3i(2,2,2),"n","grid resolution. (default = 2 2 2)"))
    , d_computeHexaList(initData(&d_computeHexaList, true, "computeHexaList", "put true if the list of Hexahedra is needed during init (default=true)"))
    , d_computeQuadList(initData(&d_computeQuadList, true, "computeQuadList", "put true if the list of Quad is needed during init (default=true)"))
    , d_computeTriangleList(initData(&d_computeTriangleList, true, "computeTriangleList", "put true if the list of triangle is needed during init (default=true)"))
    , d_computeEdgeList(initData(&d_computeEdgeList, true, "computeEdgeList", "put true if the list of Lines is needed during init (default=true)"))
    , d_computePointList(initData(&d_computePointList, true, "computePointList", "put true if the list of Points is needed during init (default=true)"))
    , d_createTexCoords(initData(&d_createTexCoords, (bool)false, "createTexCoords", "If set to true, virtual texture coordinates will be generated using 3D interpolation (default=false)."))
{
    setNbGridPoints();
    const GridUpdate::SPtr gridUpdate = sofa::core::objectmodel::New<GridUpdate>(this);
    this->addSlave(gridUpdate);
}

/// This constructor is chained with the one without parameter
GridTopology::GridTopology(const type::Vec3i& dimXYZ ) :
    GridTopology()
{
    d_n.setValue(dimXYZ);
    checkGridResolution();
}

/// This constructor is chained with the one with a type::Vec3i parameter
GridTopology::GridTopology(int nx, int ny, int nz) :
    GridTopology(type::Vec3i(nx,ny,nz))
{
}

void GridTopology::init()
{
    // first check resolution
    checkGridResolution();

    if (d_computePointList.getValue())
        this->computePointList();

    if (d_createTexCoords.getValue())
        this->createTexCoords();

    if (d_computeHexaList.getValue())
        this->computeHexaList();

    if (d_computeQuadList.getValue())
        this->computeQuadList();

    if (d_computeEdgeList.getValue())
        this->computeEdgeList();

    Inherit1::init();
}

void GridTopology::reinit()
{
    checkGridResolution();
}

void GridTopology::setSize(int nx, int ny, int nz)
{
    const auto n = this->d_n.getValue();
    if (nx == n[0] && ny == n[1] && nz == n[2])
        return;
    this->d_n.setValue(type::Vec3i(nx,ny,nz));
    setNbGridPoints();

    checkGridResolution();
}

void GridTopology::checkGridResolution()
{
    const type::Vec3i& _n = d_n.getValue();

    if (_n[0] < 1 || _n[1] < 1 || _n[2] < 1)
    {
        msg_warning() << "The grid resolution: ["<< _n[0] << " ; " << _n[1] << " ; " << _n[2] <<
                         "] is outside the validity range. At least a resolution of 1 is needed in each 3D direction."
                         " Continuing with default value=[2; 2; 2]."
                         " Set a valid grid resolution to remove this warning message.";

        this->d_n.setValue(type::Vec3i(2,2,2));
        changeGridResolutionPostProcess();
    }

    setNbGridPoints();
}

Grid_dimension GridTopology::getDimensions() const
{
	const type::Vec3i& _n = d_n.getValue();
	int dim = 0;
	for (int i = 0; i<3; i++)
		if (_n[i] > 1)
			dim++;

	return (Grid_dimension)dim;
}

void GridTopology::setSize(type::Vec3i n)
{
    setSize(n[0],n[1],n[2]);
}

void GridTopology::setNbGridPoints()
{
    this->setNbPoints(d_n.getValue()[0]*d_n.getValue()[1]*d_n.getValue()[2]);
}


void GridTopology::computeHexaList()
{
    updateHexahedra();
}

void GridTopology::computeQuadList()
{
}

void GridTopology::computeEdgeList()
{
}

void GridTopology::computePointList()
{
    const auto nbPoints= this->getNbPoints();
    // put the result in seqPoints
    SeqPoints& seq_P= *(seqPoints.beginWriteOnly());
    seq_P.resize(nbPoints);

    for (Size i=0; i<nbPoints; i++)
    {
        seq_P[i] = this->getPoint(i);
    }

    seqPoints.endEdit();
}

GridTopology::Index GridTopology::getIndex( int i, int j, int k ) const
{
    const auto& n = d_n.getValue();
    return Index(n[0]* ( n[1]*k + j ) + i);
}


sofa::type::Vec3 GridTopology::getPoint(Index i) const
{
    const auto& n = d_n.getValue();
    const int x = i%n[0]; i/=n[0];
    const int y = i%n[1]; i/=n[1];
    const int z = int(i);

    return getPointInGrid(x,y,z);
}

sofa::type::Vec3 GridTopology::getPointInGrid(int i, int j, int k) const
{
    const auto& spoints = seqPoints.getValue();

    const Index id = this->getIndex(i, j, k);
    if (id < spoints.size())
        return spoints[id];
    else
        return sofa::type::Vec3();
}


GridTopology::Hexa GridTopology::getHexaCopy(Index i)
{
    const auto& n = d_n.getValue();

    const int x = i%(n[0]-1); i/=(n[0]-1);
    const int y = i%(n[1]-1); i/=(n[1]-1);
    const int z = int(i);
    return getHexahedron(x,y,z);
}

GridTopology::Hexa GridTopology::getHexahedron(int x, int y, int z)
{

    return Hexa(point(x  ,y  ,z  ),point(x+1,y  ,z  ),
            point(x+1,y+1,z  ),point(x  ,y+1,z  ),
            point(x  ,y  ,z+1),point(x+1,y  ,z+1),
            point(x+1,y+1,z+1),point(x  ,y+1,z+1));
}

GridTopology::Quad GridTopology::getQuadCopy(Index i)
{
    const auto& n = d_n.getValue();

    if (n[0] == 1)
    {
        const int y = i%(n[1]-1);
        i/=(n[1]-1);
        const int z = i%(n[2]-1);

        return getQuad(1,y,z);
    }
    else if (n[1] == 1)
    {
        const int x = i%(n[0]-1);
        i/=(n[0]-1);
        const int z = i%(n[2]-1);

        return getQuad(x,1,z);
    }
    else
    {
        const int x = i%(n[0]-1);
        i/=(n[0]-1);
        const int y = i%(n[1]-1);

        return getQuad(x,y,1);
    }
}

GridTopology::Quad GridTopology::getQuad(int x, int y, int z)
{
    const auto& n = d_n.getValue();

    if (n[2] == 1)
        return Quad(point(x, y, 1), point(x+1, y, 1),
                point(x+1, y+1, 1), point(x, y+1, 1));
    else if (n[1] == 1)
        return Quad(point(x, 1, z), point(x+1, 1, z),
                point(x+1, 1, z+1), point(x, 1, z+1));
    else
        return Quad(point(1, y, z),point(1, y+1, z),
                point(1, y+1, z+1),point(1, y, z+1));
}

} //namespace sofa::component::topology::container::grid
