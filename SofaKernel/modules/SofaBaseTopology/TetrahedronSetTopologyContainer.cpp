/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace topology
{

using namespace std;
using namespace sofa::defaulttype;

int TetrahedronSetTopologyContainerClass = core::RegisterObject("Tetrahedron set topology container")
        .add< TetrahedronSetTopologyContainer >()
        ;

const unsigned int edgesInTetrahedronArray[6][2] = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};
///convention triangles in tetra (orientation interior)

TetrahedronSetTopologyContainer::TetrahedronSetTopologyContainer()
    : TriangleSetTopologyContainer()
	, d_createTriangleArray(initData(&d_createTriangleArray, bool(false),"createTriangleArray", "Force the creation of a set of triangles associated with each tetrahedron"))
    , d_tetrahedron(initData(&d_tetrahedron, "tetrahedra", "List of tetrahedron indices"))
{
    addAlias(&d_tetrahedron, "tetras");
}



void TetrahedronSetTopologyContainer::addTetra( int a, int b, int c, int d )
{
    helper::WriteAccessor< Data< sofa::helper::vector<Tetrahedron> > > m_tetrahedron = d_tetrahedron;
    m_tetrahedron.push_back(Tetra(a,b,c,d));
    if (a >= getNbPoints()) setNbPoints(a+1);
    if (b >= getNbPoints()) setNbPoints(b+1);
    if (c >= getNbPoints()) setNbPoints(c+1);
    if (d >= getNbPoints()) setNbPoints(d+1);
}

void TetrahedronSetTopologyContainer::init()
{
    d_tetrahedron.updateIfDirty(); // make sure m_tetrahedron is up to date
    helper::ReadAccessor< Data< sofa::helper::vector<Tetrahedron> > > m_tetrahedron = d_tetrahedron;

    // Todo (epernod 2019-03-12): optimise by removing this loop or at least create tetrahedronAV at the same time.
    if (!m_tetrahedron.empty())
    {
        for (size_t i=0; i<m_tetrahedron.size(); ++i)
        {
            for(PointID j=0; j<4; ++j)
            {
                int a = m_tetrahedron[i][j];
                if (a >= getNbPoints()) setNbPoints(a+1);
            }
        }
    }

    if (!m_tetrahedron.empty())
        initTopology();
}

void TetrahedronSetTopologyContainer::initTopology()
{
    TriangleSetTopologyContainer::initTopology();

    // Create tetrahedron cross element buffers.
    createTrianglesInTetrahedronArray();
    createEdgesInTetrahedronArray();

    createTetrahedraAroundTriangleArray();
    createTetrahedraAroundEdgeArray();
    createTetrahedraAroundVertexArray();
}

void TetrahedronSetTopologyContainer::createTetrahedronSetArray()
{
	if (CHECK_TOPOLOGY)
      msg_error() << "createTetrahedronSetArray method must be implemented by a child topology.";
}

void TetrahedronSetTopologyContainer::createEdgeSetArray()
{
    if(!hasTetrahedra()) // this method should only be called when tetrahedra exist
        createTetrahedronSetArray();

    if(hasEdges())
    {
        EdgeSetTopologyContainer::clear();

        clearEdgesInTriangle();
        clearTrianglesAroundEdge();

        clearEdgesInTetrahedron();
        clearTetrahedraAroundEdge();
    }

    // create a temporary map to find redundant edges
    std::map<Edge,EdgeID> edgeMap;
    helper::WriteAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;
    helper::ReadAccessor< Data< sofa::helper::vector<Tetrahedron> > > m_tetrahedron = d_tetrahedron;

    /// create the m_edge array at the same time than it fills the m_edgesInTetrahedron array
    for (size_t i = 0; i < m_tetrahedron.size(); ++i)
    {
        const Tetrahedron &t = m_tetrahedron[i];
        for (EdgeID j=0; j<6; ++j)
        {
            const PointID v1 = t[edgesInTetrahedronArray[j][0]];
            const PointID v2 = t[edgesInTetrahedronArray[j][1]];

            // sort vertices in lexicographic order
            const Edge e((v1<v2) ? Edge(v1,v2) : Edge(v2,v1));

            if (edgeMap.find(e)==edgeMap.end())
            {
                // edge not in edgeMap so create a new one
                const size_t edgeIndex = edgeMap.size();
                edgeMap[e] = (EdgeID)edgeIndex;
                m_edge.push_back(e);
            }
        }
    }
}

void TetrahedronSetTopologyContainer::createEdgesInTetrahedronArray()
{
    // first clear potential previous buffer
    clearEdgesInTetrahedron();

    if(!hasTetrahedra()) // this method should only be called when triangles exist
        createTetrahedronSetArray();

    if (hasEdgesInTetrahedron()) // created by upper topology
        return;


    helper::ReadAccessor< Data< sofa::helper::vector<Tetrahedron> > > m_tetrahedron = d_tetrahedron;

    if(!hasEdges()) // To optimize, this method should be called without creating edgesArray before.
    {
        /// create edge array and triangle edge array at the same time
        const size_t numTetra = getNumberOfTetrahedra();
        m_edgesInTetrahedron.resize (numTetra);

        // create a temporary map to find redundant edges
        std::map<Edge,EdgeID> edgeMap;
        helper::WriteAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;

        /// create the m_edge array at the same time than it fills the m_edgesInTetrahedron array
        for (size_t i = 0; i < m_tetrahedron.size(); ++i)
        {
            const Tetrahedron &t = m_tetrahedron[i];
            for (EdgeID j=0; j<6; ++j)
            {
                const PointID v1 = t[edgesInTetrahedronArray[j][0]];
                const PointID v2 = t[edgesInTetrahedronArray[j][1]];

                // sort vertices in lexicographic order
                const Edge e((v1<v2) ? Edge(v1,v2) : Edge(v2,v1));

                if (edgeMap.find(e)==edgeMap.end())
                {
                    // edge not in edgeMap so create a new one
                    const size_t edgeIndex = edgeMap.size();
                    edgeMap[e] = (EdgeID)edgeIndex;
                    m_edge.push_back(e);
                }
                m_edgesInTetrahedron[i][j] = edgeMap[e];
            }
        }
    }
    else
    {
        /// there are already existing edges : must use an inefficient method. Parse all triangles and find the edge that match each triangle edge
        const size_t numTetra = getNumberOfTetrahedra();
        const EdgeID numEdges = (EdgeID)getNumberOfEdges();
        helper::ReadAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;

        m_edgesInTetrahedron.resize(numTetra);
        /// create a multi map where the key is a vertex index and the content is the indices of edges adjacent to that vertex.
        std::multimap<PointID, EdgeID> edgesAroundVertexMap;
        std::multimap<PointID, EdgeID>::iterator it;
        bool foundEdge;

        for (EdgeID edge=0; edge<numEdges; ++edge)  //Todo: check if not better using multimap <PointID ,TriangleID> and for each edge, push each triangle present in both shell
        {
            edgesAroundVertexMap.insert(std::pair<PointID, EdgeID> (m_edge[edge][0],edge));
            edgesAroundVertexMap.insert(std::pair<PointID, EdgeID> (m_edge[edge][1],edge));
        }

        for(size_t i=0; i<numTetra; ++i)
        {
            const Tetrahedron &t = m_tetrahedron[i];
            // adding edge i in the edge shell of both points
            for(EdgeID j=0; j<6; ++j)
            {
                //finding edge i in edge array
                std::pair<std::multimap<PointID, EdgeID>::iterator, std::multimap<PointID, EdgeID>::iterator > itPair=edgesAroundVertexMap.equal_range(t[edgesInTetrahedronArray[j][0]]);

                foundEdge=false;
                for(it=itPair.first; (it!=itPair.second) && (foundEdge==false); ++it)
                {
                    EdgeID edge = (*it).second;
                    if ( (m_edge[edge][0] == t[edgesInTetrahedronArray[j][0]] && m_edge[edge][1] == t[edgesInTetrahedronArray[j][1]]) || (m_edge[edge][0] == t[edgesInTetrahedronArray[j][1]] && m_edge[edge][1] == t[edgesInTetrahedronArray[j][0]]))
                    {
                        m_edgesInTetrahedron[i][j] = edge;
                        foundEdge=true;
                    }
                }

                if (CHECK_TOPOLOGY)
                    msg_warning_when(!foundEdge) << " In getTetrahedronArray, cannot find edge for tetrahedron " << i << "and edge "<< j;
            }
        }
    }
}

void TetrahedronSetTopologyContainer::createTriangleSetArray()
{
    if(!hasTetrahedra()) // this method should only be called when tetrahedra exist
        createTetrahedronSetArray();

    if(hasTriangles())
    {
        TriangleSetTopologyContainer::clear();

        clearTrianglesInTetrahedron();
        clearTetrahedraAroundTriangle();
    }

    // create a temporary map to find redundant triangles
    std::map<Triangle,TriangleID> triangleMap;
    helper::WriteAccessor< Data< sofa::helper::vector<Triangle> > > m_triangle = d_triangle;
    helper::ReadAccessor< Data< sofa::helper::vector<Tetrahedron> > > m_tetrahedron = d_tetrahedron;

    /// create the m_edge array at the same time than it fills the m_edgesInTetrahedron array
    for (size_t i=0; i<m_tetrahedron.size(); ++i)
    {
        const Tetrahedron &t = m_tetrahedron[i];

        for (TriangleID j=0; j<4; ++j)
        {
            PointID v[3];
            for (PointID k=0; k<3; ++k)
                v[k] = t[sofa::core::topology::trianglesOrientationInTetrahedronArray[j][k]];

            // sort v such that v[0] is the smallest one
            while ((v[0]>v[1]) || (v[0]>v[2]))
            {
                PointID val=v[0];
                v[0]=v[1];
                v[1]=v[2];
                v[2]=val;
            }

            // check if a triangle with an opposite orientation already exists
            Triangle tr = Triangle(v[0], v[2], v[1]);

            if (triangleMap.find(tr) == triangleMap.end())
            {
                // triangle not in triangleMap so create a new one
                tr = Triangle(v[0], v[1], v[2]);
                if (triangleMap.find(tr) == triangleMap.end())
                {
                    triangleMap[tr] = (TriangleID)m_triangle.size();
                    m_triangle.push_back(tr);
                }
                else
                {
					msg_error() << "Duplicate triangle " << tr << " in tetra " << i <<" : " << t;
                }
            }
        }
    }
}

void TetrahedronSetTopologyContainer::createTrianglesInTetrahedronArray()
{
    // first clear potential previous buffer
    clearTrianglesInTetrahedron();

    if(!hasTriangles())
        createTriangleSetArray();

    if(hasTrianglesInTetrahedron()) // created by upper topology
        return;

    m_trianglesInTetrahedron.resize( getNumberOfTetrahedra());
    helper::ReadAccessor< Data< sofa::helper::vector<Tetrahedron> > > m_tetrahedron = d_tetrahedron;

    for(size_t i = 0; i < m_tetrahedron.size(); ++i)
    {
        const Tetrahedron &t=m_tetrahedron[i];

        // adding triangles in the triangle list of the ith tetrahedron  i
        for (TriangleID j=0; j<4; ++j)
        {
            TriangleID triangleIndex = getTriangleIndex(t[(j+1)%4], t[(j+2)%4], t[(j+3)%4]);
            m_trianglesInTetrahedron[i][j] = triangleIndex;
        }
    }
}

void TetrahedronSetTopologyContainer::createTetrahedraAroundVertexArray()
{
    // first clear potential previous buffer
    clearTetrahedraAroundVertex();

    if (getNbPoints() == 0) // in case only Data have been copied and not going thourgh AddTriangle methods.
        this->setNbPoints(d_initPoints.getValue().size());

    m_tetrahedraAroundVertex.resize( getNbPoints() );
    helper::ReadAccessor< Data< sofa::helper::vector<Tetrahedron> > > m_tetrahedron = d_tetrahedron;

    for (size_t i = 0; i < getNumberOfTetrahedra(); ++i)
    {
        // adding edge i in the edge shell of both points
        for (PointID j=0; j<4; ++j)
        {
            m_tetrahedraAroundVertex[ m_tetrahedron[i][j]  ].push_back( (TetrahedronID)i );
        }
    }
}

void TetrahedronSetTopologyContainer::createTetrahedraAroundEdgeArray ()
{
    // first clear potential previous buffer
    clearTetrahedraAroundEdge();

    if(!hasEdgesInTetrahedron())
        createEdgesInTetrahedronArray();

    m_tetrahedraAroundEdge.resize(getNumberOfEdges());
    for (size_t i=0; i< getNumberOfTetrahedra(); ++i)
    {
        // adding edge i in the edge shell of both points
        for (EdgeID j=0; j<6; ++j)
        {
            m_tetrahedraAroundEdge[ m_edgesInTetrahedron[i][j] ].push_back( (TetrahedronID)i );
        }
    }
}

void TetrahedronSetTopologyContainer::createTetrahedraAroundTriangleArray ()
{
    // first clear potential previous buffer
    clearTetrahedraAroundTriangle();

    if(!hasTrianglesInTetrahedron())
        createTrianglesInTetrahedronArray();

    m_tetrahedraAroundTriangle.resize( getNumberOfTriangles());

    for (size_t i=0; i<getNumberOfTetrahedra(); ++i)
    {
        // adding tetrahedron i in the shell of all neighbors triangles
        for (TriangleID j=0; j<4; ++j)
        {
            m_tetrahedraAroundTriangle[ m_trianglesInTetrahedron[i][j] ].push_back( (TetrahedronID)i );
        }
    }
}

const sofa::helper::vector<TetrahedronSetTopologyContainer::Tetrahedron> &TetrahedronSetTopologyContainer::getTetrahedronArray()
{
    if (CHECK_TOPOLOGY)
        msg_warning_when(!hasTetrahedra() && getNbPoints()>0) << "Tetrahedron array is empty with " << getNbPoints() << " vertices.";

    return d_tetrahedron.getValue();
}


const TetrahedronSetTopologyContainer::Tetrahedron TetrahedronSetTopologyContainer::getTetrahedron (TetraID i)
{
    if ((size_t)i >= getNbTetrahedra())
        return Tetrahedron(InvalidID, InvalidID, InvalidID, InvalidID);
    else
        return (d_tetrahedron.getValue())[i];
}



TetrahedronSetTopologyContainer::TetrahedronID TetrahedronSetTopologyContainer::getTetrahedronIndex(PointID v1, PointID v2, PointID v3, PointID v4)
{
    if(!hasTetrahedraAroundVertex())
    {
        if(CHECK_TOPOLOGY)
            msg_warning() << "TetrahedraAroundVertex array is empty with " << getNbPoints() << " vertices.";

        return InvalidID;
    }

    sofa::helper::vector<TetrahedronID> set1 = getTetrahedraAroundVertex(v1);
    sofa::helper::vector<TetrahedronID> set2 = getTetrahedraAroundVertex(v2);
    sofa::helper::vector<TetrahedronID> set3 = getTetrahedraAroundVertex(v3);
    sofa::helper::vector<TetrahedronID> set4 = getTetrahedraAroundVertex(v4);

    sort(set1.begin(), set1.end());
    sort(set2.begin(), set2.end());
    sort(set3.begin(), set3.end());
    sort(set4.begin(), set4.end());

    // The destination vector must be large enough to contain the result.
    sofa::helper::vector<TetrahedronID> out1(set1.size()+set2.size());
    sofa::helper::vector<TetrahedronID>::iterator result1;
    result1 = std::set_intersection(set1.begin(),set1.end(),set2.begin(),set2.end(),out1.begin());
    out1.erase(result1,out1.end());

    sofa::helper::vector<TetrahedronID> out2(set3.size()+out1.size());
    sofa::helper::vector<TetrahedronID>::iterator result2;
    result2 = std::set_intersection(set3.begin(),set3.end(),out1.begin(),out1.end(),out2.begin());
    out2.erase(result2,out2.end());

    sofa::helper::vector<TetrahedronID> out3(set4.size()+out2.size());
    sofa::helper::vector<TetrahedronID>::iterator result3;
    result3 = std::set_intersection(set4.begin(),set4.end(),out2.begin(),out2.end(),out3.begin());
    out3.erase(result3,out3.end());

    assert(out3.size()==0 || out3.size()==1);

    if(CHECK_TOPOLOGY)
        msg_warning_when(out3.size() > 1) << "More than one Tetrahedron found for indices: [" << v1 << "; " << v2 << "; " << v3 << "; " << v4 << "]";

    if (out3.size()==1)
        return (int) (out3[0]);

    return InvalidID;
}

size_t TetrahedronSetTopologyContainer::getNumberOfTetrahedra() const
{
    helper::ReadAccessor< Data< sofa::helper::vector<Tetrahedron> > > m_tetrahedron = d_tetrahedron;
    return m_tetrahedron.size();
}

size_t TetrahedronSetTopologyContainer::getNumberOfElements() const
{
    return this->getNumberOfTetrahedra();
}

const sofa::helper::vector< TetrahedronSetTopologyContainer::TetrahedraAroundVertex > &TetrahedronSetTopologyContainer::getTetrahedraAroundVertexArray()
{
    if (CHECK_TOPOLOGY)
        msg_warning_when(!hasTetrahedraAroundVertex()) << "TetrahedraAroundVertex shell array is empty.";

    return m_tetrahedraAroundVertex;
}

const sofa::helper::vector< TetrahedronSetTopologyContainer::TetrahedraAroundEdge > &TetrahedronSetTopologyContainer::getTetrahedraAroundEdgeArray()
{
    if (CHECK_TOPOLOGY)
        msg_warning_when(!hasTetrahedraAroundEdge()) << "TetrahedraAroundEdge shell array is empty.";

    return m_tetrahedraAroundEdge;
}

const sofa::helper::vector< TetrahedronSetTopologyContainer::TetrahedraAroundTriangle > &TetrahedronSetTopologyContainer::getTetrahedraAroundTriangleArray()
{
    if (CHECK_TOPOLOGY)
        msg_warning_when(!hasTetrahedraAroundTriangle()) << "TetrahedraAroundTriangle shell array is empty.";

    return m_tetrahedraAroundTriangle;
}

const sofa::helper::vector< TetrahedronSetTopologyContainer::EdgesInTetrahedron> &TetrahedronSetTopologyContainer::getEdgesInTetrahedronArray()
{
    if (CHECK_TOPOLOGY)
        msg_warning_when(!hasEdgesInTetrahedron()) << "EdgesInTetrahedron shell array is empty.";

    return m_edgesInTetrahedron;
}

TetrahedronSetTopologyContainer::Edge TetrahedronSetTopologyContainer::getLocalEdgesInTetrahedron (const EdgeID i) const
{
    assert(i<6);
    return Edge (edgesInTetrahedronArray[i][0], edgesInTetrahedronArray[i][1]);
}

TetrahedronSetTopologyContainer::Triangle TetrahedronSetTopologyContainer::getLocalTrianglesInTetrahedron (const TriangleID i) const
{
    assert(i<4);
    return Triangle (sofa::core::topology::trianglesOrientationInTetrahedronArray[i][0],
            sofa::core::topology::trianglesOrientationInTetrahedronArray[i][1],
            sofa::core::topology::trianglesOrientationInTetrahedronArray[i][2]);
}

const sofa::helper::vector< TetrahedronSetTopologyContainer::TrianglesInTetrahedron> &TetrahedronSetTopologyContainer::getTrianglesInTetrahedronArray()
{
    if (CHECK_TOPOLOGY)
        msg_warning_when(!hasTrianglesInTetrahedron()) << "TrianglesInTetrahedron shell array is empty.";

    return m_trianglesInTetrahedron;
}

const TetrahedronSetTopologyContainer::TetrahedraAroundVertex &TetrahedronSetTopologyContainer::getTetrahedraAroundVertex(const PointID id)
{
    if (id < m_tetrahedraAroundVertex.size())
        return m_tetrahedraAroundVertex[id];
    else if (CHECK_TOPOLOGY)
        msg_error() << "TetrahedraAroundVertexArray array access out of bounds: " << id << " >= " << m_tetrahedraAroundVertex.size();

    return InvalidSet;
}

const TetrahedronSetTopologyContainer::TetrahedraAroundEdge &TetrahedronSetTopologyContainer::getTetrahedraAroundEdge(const EdgeID id)
{
    if (id < m_tetrahedraAroundEdge.size())
        return m_tetrahedraAroundEdge[id];
    else if (CHECK_TOPOLOGY)
        msg_error() << "TetrahedraAroundEdge array access out of bounds: " << id << " >= " << m_tetrahedraAroundEdge.size();

    return InvalidSet;
}

const TetrahedronSetTopologyContainer::TetrahedraAroundTriangle &TetrahedronSetTopologyContainer::getTetrahedraAroundTriangle(const TriangleID id)
{
    if (id < m_tetrahedraAroundTriangle.size())
        return m_tetrahedraAroundTriangle[id];
    else if (CHECK_TOPOLOGY)
        msg_error() << "TetrahedraAroundTriangle array access out of bounds: " << id << " >= " << m_tetrahedraAroundTriangle.size();

    return InvalidSet;
}

const TetrahedronSetTopologyContainer::EdgesInTetrahedron &TetrahedronSetTopologyContainer::getEdgesInTetrahedron(const EdgeID id)
{
    if (id < m_edgesInTetrahedron.size())
        return m_edgesInTetrahedron[id];
    else if (CHECK_TOPOLOGY)
        msg_error() << "EdgesInTetrahedron array access out of bounds: " << id << " >= " << m_edgesInTetrahedron.size();

    return InvalidEdgesInTetrahedron;
}

const TetrahedronSetTopologyContainer::TrianglesInTetrahedron &TetrahedronSetTopologyContainer::getTrianglesInTetrahedron(const TriangleID id)
{
    if (id < m_trianglesInTetrahedron.size())
        return m_trianglesInTetrahedron[id];
    else if (CHECK_TOPOLOGY)
        msg_error() << "TrianglesInTetrahedron array access out of bounds: " << id << " >= " << m_trianglesInTetrahedron.size();

    return InvalidTetrahedron;
}

int TetrahedronSetTopologyContainer::getVertexIndexInTetrahedron(const Tetrahedron &t,
        PointID vertexIndex) const
{
    if (t[0]==vertexIndex)
        return 0;
    else if (t[1]==vertexIndex)
        return 1;
    else if (t[2]==vertexIndex)
        return 2;
    else if (t[3]==vertexIndex)
        return 3;
    else
        return -1;
}

int TetrahedronSetTopologyContainer::getEdgeIndexInTetrahedron(const EdgesInTetrahedron &t,
        const EdgeID edgeIndex) const
{
    if (t[0]==edgeIndex)
        return 0;
    else if (t[1]==edgeIndex)
        return 1;
    else if (t[2]==edgeIndex)
        return 2;
    else if (t[3]==edgeIndex)
        return 3;
    else if (t[4]==edgeIndex)
        return 4;
    else if (t[5]==edgeIndex)
        return 5;
    else
        return -1;
}

int TetrahedronSetTopologyContainer::getTriangleIndexInTetrahedron(const TrianglesInTetrahedron &t,
        const TriangleID triangleIndex) const
{
    if (t[0]==triangleIndex)
        return 0;
    else if (t[1]==triangleIndex)
        return 1;
    else if (t[2]==triangleIndex)
        return 2;
    else if (t[3]==triangleIndex)
        return 3;
    else
        return -1;
}

TetrahedronSetTopologyContainer::TetrahedraAroundEdge &TetrahedronSetTopologyContainer::getTetrahedraAroundEdgeForModification(const EdgeID i)
{
    if (!hasTetrahedraAroundEdge())
        createTetrahedraAroundEdgeArray();

    assert(i < m_tetrahedraAroundEdge.size());

    return m_tetrahedraAroundEdge[i];
}

TetrahedronSetTopologyContainer::TetrahedraAroundVertex &TetrahedronSetTopologyContainer::getTetrahedraAroundVertexForModification(const PointID i)
{
    if (!hasTetrahedraAroundVertex())
        createTetrahedraAroundVertexArray();

    assert(i < m_tetrahedraAroundVertex.size());

    return m_tetrahedraAroundVertex[i];
}

TetrahedronSetTopologyContainer::TetrahedraAroundTriangle &TetrahedronSetTopologyContainer::getTetrahedraAroundTriangleForModification(const TriangleID i)
{
    if (!hasTetrahedraAroundTriangle())
        createTetrahedraAroundTriangleArray();

    assert(i < m_tetrahedraAroundTriangle.size());

    return m_tetrahedraAroundTriangle[i];
}


bool TetrahedronSetTopologyContainer::checkTopology() const
{
    if (CHECK_TOPOLOGY)
    {
        bool ret = true;
        helper::ReadAccessor< Data< sofa::helper::vector<Tetrahedron> > > m_tetrahedron = d_tetrahedron;

        if (hasTetrahedraAroundVertex())
        {
            for (size_t i = 0; i < m_tetrahedraAroundVertex.size(); ++i)
            {
                const sofa::helper::vector<TetrahedronID> &tvs = m_tetrahedraAroundVertex[i];
                for (size_t j = 0; j < tvs.size(); ++j)
                {
                    bool check_tetra_vertex_shell = (m_tetrahedron[tvs[j]][0] == i)
                        || (m_tetrahedron[tvs[j]][1] == i)
                        || (m_tetrahedron[tvs[j]][2] == i)
                        || (m_tetrahedron[tvs[j]][3] == i);
                    if (!check_tetra_vertex_shell)
                    {
                        msg_error() << "*** CHECK FAILED : check_tetra_vertex_shell, i = " << i << " , j = " << j ;
                        ret = false;
                    }
                }
            }
        }

        if (hasTetrahedraAroundEdge())
        {
            for (size_t i = 0; i < m_tetrahedraAroundEdge.size(); ++i)
            {
                const sofa::helper::vector<TetrahedronID> &tes = m_tetrahedraAroundEdge[i];
                for (size_t j = 0; j < tes.size(); ++j)
                {
                    bool check_tetra_edge_shell = (m_edgesInTetrahedron[tes[j]][0] == i)
                        || (m_edgesInTetrahedron[tes[j]][1] == i)
                        || (m_edgesInTetrahedron[tes[j]][2] == i)
                        || (m_edgesInTetrahedron[tes[j]][3] == i)
                        || (m_edgesInTetrahedron[tes[j]][4] == i)
                        || (m_edgesInTetrahedron[tes[j]][5] == i);
                    if (!check_tetra_edge_shell)
                    {
                        msg_error() << "*** CHECK FAILED : check_tetra_edge_shell, i = " << i << " , j = " << j ;
                        ret = false;
                    }
                }
            }
        }

        if (hasTetrahedraAroundTriangle())
        {
            for (size_t i = 0; i < m_tetrahedraAroundTriangle.size(); ++i)
            {
                const sofa::helper::vector<TetrahedronID> &tes = m_tetrahedraAroundTriangle[i];
                for (size_t j = 0; j < tes.size(); ++j)
                {
                    bool check_tetra_triangle_shell = (m_trianglesInTetrahedron[tes[j]][0] == i)
                        || (m_trianglesInTetrahedron[tes[j]][1] == i)
                        || (m_trianglesInTetrahedron[tes[j]][2] == i)
                        || (m_trianglesInTetrahedron[tes[j]][3] == i);
                    if (!check_tetra_triangle_shell)
                    {
                        msg_error() << "*** CHECK FAILED : check_tetra_triangle_shell, i = " << i << " , j = " << j ;
                        ret = false;
                    }
                }
            }
        }

        return ret && TriangleSetTopologyContainer::checkTopology();
    }

    return true;
}





/// Get information about connexity of the mesh
/// @{
bool TetrahedronSetTopologyContainer::checkConnexity()
{

    size_t nbr = this->getNbTetrahedra();

    if (nbr == 0)
    {
        if(CHECK_TOPOLOGY)
            msg_warning() << "Can't compute connexity as there are no tetrahedra";

        return false;
    }

    VecTetraID elemAll = this->getConnectedElement(0);

    if (elemAll.size() != nbr)
    {
        msg_warning() << "In computing connexity, tetrahedra are missings. There is more than one connexe component.";
        return false;
    }

    return true;
}


size_t TetrahedronSetTopologyContainer::getNumberOfConnectedComponent()
{
    size_t nbr = this->getNbTetrahedra();

    if (nbr == 0)
    {
        if(CHECK_TOPOLOGY)
            msg_warning() << "Can't compute connexity as there are no tetrahedra";

        return 0;
    }

    VecTetraID elemAll = this->getConnectedElement(0);
    size_t cpt = 1;

    while (elemAll.size() < nbr)
    {
        std::sort(elemAll.begin(), elemAll.end());
        TetraID other_tetraID = (TetraID)elemAll.size();

        for (TetraID i = 0; i<elemAll.size(); ++i)
            if (elemAll[i] != i)
            {
                other_tetraID = i;
                break;
            }

        VecTetraID elemTmp = this->getConnectedElement(other_tetraID);
        cpt++;

        elemAll.insert(elemAll.begin(), elemTmp.begin(), elemTmp.end());
    }

    return cpt;
}


const TetrahedronSetTopologyContainer::VecTetraID TetrahedronSetTopologyContainer::getConnectedElement(TetraID elem)
{
    if(!hasTetrahedraAroundVertex())	// this method should only be called when the shell array exists
    {
        if(CHECK_TOPOLOGY)
            msg_warning() << "Tetrahedra vertex shell array is empty.";

        createTetrahedraAroundVertexArray();
    }

    VecTetraID elemAll;
    VecTetraID elemOnFront, elemPreviousFront, elemNextFront;
    bool end = false;
    size_t cpt = 0;
    size_t nbr = this->getNbTetrahedra();

    // init algo
    elemAll.push_back(elem);
    elemOnFront.push_back(elem);
    elemPreviousFront.clear();
    cpt++;

    while (!end && cpt < nbr)
    {
        // First Step - Create new region
        elemNextFront = this->getElementAroundElements(elemOnFront); // for each TetraID on the propagation front

        // Second Step - Avoid backward direction
        for (size_t i = 0; i<elemNextFront.size(); ++i)
        {
            bool find = false;
            TetraID id = elemNextFront[i];

            for (size_t j = 0; j<elemAll.size(); ++j)
                if (id == elemAll[j])
                {
                    find = true;
                    break;
                }

            if (!find)
            {
                elemAll.push_back(id);
                elemPreviousFront.push_back(id);
            }
        }

        // cpt for connexity
        cpt +=elemPreviousFront.size();

        if (elemPreviousFront.empty())
        {
            end = true;
            if(CHECK_TOPOLOGY)
                msg_warning() << "Loop for computing connexity has reach end.";

        }

        // iterate
        elemOnFront = elemPreviousFront;
        elemPreviousFront.clear();
    }

    return elemAll;
}


const TetrahedronSetTopologyContainer::VecTetraID TetrahedronSetTopologyContainer::getElementAroundElement(TetraID elem)
{
    VecTetraID elems;

    if (!hasTetrahedraAroundVertex())
    {
        if(CHECK_TOPOLOGY)
            msg_warning() << "Tetrahedra vertex shell array is empty.";

        createTetrahedraAroundVertexArray();
    }

    Tetra the_tetra = this->getTetra(elem);

    for(PointID i = 0; i<4; ++i) // for each node of the tetra
    {
        TetrahedraAroundVertex tetraAV = this->getTetrahedraAroundVertex(the_tetra[i]);

        for (size_t j = 0; j<tetraAV.size(); ++j) // for each tetra around the node
        {
            bool find = false;
            TetraID id = tetraAV[j];

            if (id == elem)
                continue;

            for (size_t k = 0; k<elems.size(); ++k) // check no redundancy
                if (id == elems[k])
                {
                    find = true;
                    break;
                }

            if (!find)
                elems.push_back(id);
        }
    }

    return elems;
}


const TetrahedronSetTopologyContainer::VecTetraID TetrahedronSetTopologyContainer::getElementAroundElements(VecTetraID elems)
{
    VecTetraID elemAll;
    VecTetraID elemTmp;

    if (!hasTetrahedraAroundVertex())
    {
        if(CHECK_TOPOLOGY)
            msg_warning() << "Tetrahedra vertex shell array is empty.";

        createTetrahedraAroundVertexArray();
    }

    for (size_t i = 0; i <elems.size(); ++i) // for each TetraID of input vector
    {
        VecTetraID elemTmp2 = this->getElementAroundElement(elems[i]);

        elemTmp.insert(elemTmp.end(), elemTmp2.begin(), elemTmp2.end());
    }

    for (size_t i = 0; i<elemTmp.size(); ++i) // for each tetra Id found
    {
        bool find = false;
        TetraID id = elemTmp[i];

        for (size_t j = 0; j<elems.size(); ++j) // check no redundancy with input vector
            if (id == elems[j])
            {
                find = true;
                break;
            }

        if (!find)
        {
            for (size_t j = 0; j<elemAll.size(); ++j) // check no redundancy in output vector
                if (id == elemAll[j])
                {
                    find = true;
                    break;
                }
        }

        if (!find)
            elemAll.push_back(id);
    }


    return elemAll;
}

/// @}


bool TetrahedronSetTopologyContainer::hasTetrahedra() const
{
    d_tetrahedron.updateIfDirty(); // make sure m_tetrahedron is valid
    return !(d_tetrahedron.getValue()).empty();
}

bool TetrahedronSetTopologyContainer::hasEdgesInTetrahedron() const
{
    return !m_edgesInTetrahedron.empty();
}

bool TetrahedronSetTopologyContainer::hasTrianglesInTetrahedron() const
{
    return !m_trianglesInTetrahedron.empty();
}

bool TetrahedronSetTopologyContainer::hasTetrahedraAroundVertex() const
{
    return !m_tetrahedraAroundVertex.empty();
}

bool TetrahedronSetTopologyContainer::hasTetrahedraAroundEdge() const
{
    return !m_tetrahedraAroundEdge.empty();
}

bool TetrahedronSetTopologyContainer::hasTetrahedraAroundTriangle() const
{
    return !m_tetrahedraAroundTriangle.empty();
}

void TetrahedronSetTopologyContainer::clearTetrahedra()
{
    helper::WriteAccessor< Data< sofa::helper::vector<Tetrahedron> > > m_tetrahedron = d_tetrahedron;
    m_tetrahedron.clear();
}

void TetrahedronSetTopologyContainer::clearEdgesInTetrahedron()
{
    m_edgesInTetrahedron.clear();
}

void TetrahedronSetTopologyContainer::clearTrianglesInTetrahedron()
{
    m_trianglesInTetrahedron.clear();
}

void TetrahedronSetTopologyContainer::clearTetrahedraAroundVertex()
{
    m_tetrahedraAroundVertex.clear();
}

void TetrahedronSetTopologyContainer::clearTetrahedraAroundEdge()
{
    m_tetrahedraAroundEdge.clear();
}

void TetrahedronSetTopologyContainer::clearTetrahedraAroundTriangle()
{
    m_tetrahedraAroundTriangle.clear();
}

void TetrahedronSetTopologyContainer::clear()
{
    clearTetrahedraAroundVertex();
    clearTetrahedraAroundEdge();
    clearTetrahedraAroundTriangle();
    clearEdgesInTetrahedron();
    clearTrianglesInTetrahedron();
    clearTetrahedra();

    TriangleSetTopologyContainer::clear();
}

//add removed tetrahedron index
void TetrahedronSetTopologyContainer::addRemovedTetraIndex(sofa::helper::vector< TetrahedronID >& tetrahedra)
{
    for(size_t i=0; i<tetrahedra.size(); i++)
        m_removedTetraIndex.push_back(tetrahedra[i]);
}

//get removed tetrahedron index
sofa::helper::vector< TetrahedronSetTopologyContainer::TetrahedronID >& TetrahedronSetTopologyContainer::getRemovedTetraIndex()
{
    return m_removedTetraIndex;
}

void TetrahedronSetTopologyContainer::setTetrahedronTopologyToDirty()
{
    // set this container to dirty
    m_tetrahedronTopologyDirty = true;

    // set all engines link to this container to dirty
    std::list<sofa::core::topology::TopologyEngine *>::iterator it;
    for (it = m_enginesList.begin(); it!=m_enginesList.end(); ++it)
    {
        sofa::core::topology::TopologyEngine* topoEngine = (*it);
        topoEngine->setDirtyValue();
        if (CHECK_TOPOLOGY)
            msg_info() << "Tetrahedron Topology Set dirty engine: " << topoEngine->name;
    }
}

void TetrahedronSetTopologyContainer::cleanTetrahedronTopologyFromDirty()
{
    m_tetrahedronTopologyDirty = false;

    // security, clean all engines to avoid loops
    std::list<sofa::core::topology::TopologyEngine *>::iterator it;
    for ( it = m_enginesList.begin(); it!=m_enginesList.end(); ++it)
    {
        if ((*it)->isDirty())
        {
            if (CHECK_TOPOLOGY)
                msg_warning() << "Tetrahedron Topology update did not clean engine: " << (*it)->name;
            (*it)->cleanDirty();
        }
    }
}

void TetrahedronSetTopologyContainer::updateTopologyEngineGraph()
{
    // calling real update Data graph function implemented once in PointSetTopologyModifier
    this->updateDataEngineGraph(this->d_tetrahedron, this->m_enginesList);

    // will concatenate with edges one:
    TriangleSetTopologyContainer::updateTopologyEngineGraph();
}

} // namespace topology

} // namespace component

} // namespace sofa

