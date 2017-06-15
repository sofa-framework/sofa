/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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

#include <SofaBaseTopology/QuadSetTopologyContainer.h>
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

SOFA_DECL_CLASS(QuadSetTopologyContainer)
int QuadSetTopologyContainerClass = core::RegisterObject("Quad set topology container")
        .add< QuadSetTopologyContainer >()
        ;

QuadSetTopologyContainer::QuadSetTopologyContainer()
    : EdgeSetTopologyContainer()
    , d_quad(initData(&d_quad, "quads", "List of quad indices"))
{
}


void QuadSetTopologyContainer::addQuad( int a, int b, int c, int d )
{
    helper::WriteAccessor< Data< sofa::helper::vector<Quad> > > m_quad = d_quad;
    m_quad.push_back(Quad(a,b,c,d));
    if (a >= getNbPoints()) nbPoints.setValue(a+1);
    if (b >= getNbPoints()) nbPoints.setValue(b+1);
    if (c >= getNbPoints()) nbPoints.setValue(c+1);
    if (d >= getNbPoints()) nbPoints.setValue(d+1);
}

void QuadSetTopologyContainer::init()
{
    EdgeSetTopologyContainer::init();
    d_quad.updateIfDirty(); // make sure m_quad is up to date
}


void QuadSetTopologyContainer::createQuadSetArray()
{
#ifndef NDEBUG
    sout << "Error. [QuadSetTopologyContainer::createQuadSetArray] This method must be implemented by a child topology." << sendl;
#endif
}

void QuadSetTopologyContainer::createQuadsAroundVertexArray()
{
    if(!hasQuads()) // this method should only be called when quads exist
    {
#ifndef NDEBUG
        sout << "Warning. [QuadSetTopologyContainer::createQuadsAroundVertexArray] quad array is empty." << sendl;
#endif
        createQuadSetArray();
    }

    if(hasQuadsAroundVertex())
    {
        clearQuadsAroundVertex();
    }

    m_quadsAroundVertex.resize( getNbPoints() );
    helper::ReadAccessor< Data< sofa::helper::vector<Quad> > > m_quad = d_quad;

    for (size_t i=0; i<m_quad.size(); ++i)
    {
        // adding quad i in the quad shell of all points
        for (size_t j=0; j<4; ++j)
        {
            m_quadsAroundVertex[ m_quad[i][j] ].push_back((unsigned int)i);
        }
    }
}

void QuadSetTopologyContainer::createQuadsAroundEdgeArray()
{
    if(!hasQuads()) // this method should only be called when quads exist
    {
#ifndef NDEBUG
        sout << "Warning. [QuadSetTopologyContainer::createQuadsAroundEdgeArray] quad array is empty." << sendl;
#endif
        createQuadSetArray();
    }

    if(!hasEdges()) // this method should only be called when edges exist
    {
#ifndef NDEBUG
        sout << "Warning. [QuadSetTopologyContainer::createQuadsAroundEdgeArray] edge array is empty." << sendl;
#endif
        createEdgeSetArray();
    }

    if(!hasEdgesInQuad())
        createEdgesInQuadArray();

    const size_t numQuads = getNumberOfQuads();
    const size_t numEdges = getNumberOfEdges();

    if(hasQuadsAroundEdge())
    {
        clearQuadsAroundEdge();
    }

    m_quadsAroundEdge.resize(numEdges);

    for (size_t i=0; i<numQuads; ++i)
    {
        // adding quad i in the quad shell of all edges
        for (size_t j=0; j<4; ++j)
        {
            m_quadsAroundEdge[ m_edgesInQuad[i][j] ].push_back((unsigned int)i);
        }
    }
}

void QuadSetTopologyContainer::createEdgeSetArray()
{
    if(!hasQuads()) // this method should only be called when quads exist
    {
#ifndef NDEBUG
        sout << "Warning. [QuadSetTopologyContainer::createEdgeSetArray] quad array is empty." << sendl;
#endif
        createQuadSetArray();
    }

    if(hasEdges())
    {
#ifndef NDEBUG
        sout << "Warning. [QuadSetTopologyContainer::createEdgeSetArray] edge array is not empty." << sendl;
#endif

        // clear edges and all shells that depend on edges
        EdgeSetTopologyContainer::clear();

        if(hasEdgesInQuad())
            clearEdgesInQuad();

        if(hasQuadsAroundEdge())
            clearQuadsAroundEdge();
    }

    // create a temporary map to find redundant edges
    std::map<Edge, unsigned int> edgeMap;
    helper::WriteAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;
    helper::ReadAccessor< Data< sofa::helper::vector<Quad> > > m_quad = d_quad;

    for (size_t i=0; i<m_quad.size(); ++i)
    {
        const Quad &t = m_quad[i];
        for(size_t j=0; j<4; ++j)
        {
            const unsigned int v1 = t[(j+1)%4];
            const unsigned int v2 = t[(j+2)%4];

            // sort vertices in lexicographic order
            const Edge e = ((v1<v2) ? Edge(v1,v2) : Edge(v2,v1));

            if(edgeMap.find(e) == edgeMap.end())
            {
                // edge not in edgeMap so create a new one
                const unsigned int edgeIndex = (unsigned int)edgeMap.size();
                edgeMap[e] = edgeIndex;
                m_edge.push_back(e);
            }
        }
    }
}

void QuadSetTopologyContainer::createEdgesInQuadArray()
{
    if(!hasQuads()) // this method should only be called when quads exist
    {
#ifndef NDEBUG
        sout << "Warning. [QuadSetTopologyContainer::createEdgesInQuadArray] quad array is empty." << sendl;
#endif
        createQuadSetArray();
    }

    if(!hasEdges()) // this method should only be called when edges exist
    {
#ifndef NDEBUG
        sout << "Warning. [QuadSetTopologyContainer::createEdgesInQuadArray] edge array is empty." << sendl;
#endif
        createEdgeSetArray();
    }

    if(hasEdgesInQuad())
        clearEdgesInQuad();

    const size_t numQuads = getNumberOfQuads();

    m_edgesInQuad.resize( numQuads );
    helper::ReadAccessor< Data< sofa::helper::vector<Quad> > > m_quad = d_quad;

    for(size_t i=0; i<numQuads; ++i)
    {
        const Quad &t = m_quad[i];
        // adding edge i in the edge shell of both points
        for (size_t j=0; j<4; ++j)
        {
            const int edgeIndex = getEdgeIndex(t[(j+1)%4],t[(j+2)%4]);
            m_edgesInQuad[i][j]=edgeIndex;
        }
    }
}

const sofa::helper::vector<QuadSetTopologyContainer::Quad> &QuadSetTopologyContainer::getQuadArray()
{
    if(!hasQuads() && getNbPoints()>0)
    {
#ifndef NDEBUG
        sout << "Warning. [QuadSetTopologyContainer::getQuadArray] creating quad array." << sendl;
#endif
        createQuadSetArray();
    }

    return d_quad.getValue();
}

const QuadSetTopologyContainer::Quad QuadSetTopologyContainer::getQuad (QuadID i)
{
    if(!hasQuads())
        createQuadSetArray();

    return (d_quad.getValue())[i];
}


int QuadSetTopologyContainer::getQuadIndex(PointID v1, PointID v2, PointID v3, PointID v4)
{
    if(!hasQuadsAroundVertex())
        createQuadsAroundVertexArray();

    sofa::helper::vector<unsigned int> set1 = getQuadsAroundVertex(v1);
    sofa::helper::vector<unsigned int> set2 = getQuadsAroundVertex(v2);
    sofa::helper::vector<unsigned int> set3 = getQuadsAroundVertex(v3);
    sofa::helper::vector<unsigned int> set4 = getQuadsAroundVertex(v4);

    sort(set1.begin(), set1.end());
    sort(set2.begin(), set2.end());
    sort(set3.begin(), set3.end());
    sort(set4.begin(), set4.end());

    // The destination vector must be large enough to contain the result.
    sofa::helper::vector<unsigned int> out1(set1.size()+set2.size());
    sofa::helper::vector<unsigned int>::iterator result1;
    result1 = std::set_intersection(set1.begin(),set1.end(),set2.begin(),set2.end(),out1.begin());
    out1.erase(result1,out1.end());

    sofa::helper::vector<unsigned int> out2(set3.size()+out1.size());
    sofa::helper::vector<unsigned int>::iterator result2;
    result2 = std::set_intersection(set3.begin(),set3.end(),out1.begin(),out1.end(),out2.begin());
    out2.erase(result2,out2.end());

    sofa::helper::vector<unsigned int> out3(set4.size()+out2.size());
    sofa::helper::vector<unsigned int>::iterator result3;
    result3 = std::set_intersection(set4.begin(),set4.end(),out2.begin(),out2.end(),out3.begin());
    out3.erase(result3,out3.end());

#ifndef NDEBUG
    if(out3.size() > 1)
        sout << "Warning. [QuadSetTopologyContainer::getQuadIndex] more than one quad found" << sendl;
#endif

    if(out3.size()==1)
        return (int) (out3[0]);
    else
        return -1;
}

unsigned int QuadSetTopologyContainer::getNumberOfQuads() const
{
    helper::ReadAccessor< Data< sofa::helper::vector<Quad> > > m_quad = d_quad;
    return (unsigned int)m_quad.size();
}


unsigned int QuadSetTopologyContainer::getNumberOfElements() const
{
    return this->getNumberOfQuads();
}


const sofa::helper::vector< sofa::helper::vector<unsigned int> > &QuadSetTopologyContainer::getQuadsAroundVertexArray()
{
    if(!hasQuadsAroundVertex())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        sout << "Warning. [QuadSetTopologyContainer::getQuadsAroundVertexArray] quad vertex shell array is empty." << sendl;
#endif
        createQuadsAroundVertexArray();
    }

    return m_quadsAroundVertex;
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &QuadSetTopologyContainer::getQuadsAroundEdgeArray()
{
    if(!hasQuadsAroundEdge())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        sout << "Warning. [QuadSetTopologyContainer::getQuadsAroundEdgeArray] quad edge shell array is empty." << sendl;
#endif
        createQuadsAroundEdgeArray();
    }

    return m_quadsAroundEdge;
}

const sofa::helper::vector< QuadSetTopologyContainer::EdgesInQuad> &QuadSetTopologyContainer::getEdgesInQuadArray()
{
    if(m_edgesInQuad.empty())
        createEdgesInQuadArray();

    return m_edgesInQuad;
}

const QuadSetTopologyContainer::QuadsAroundVertex& QuadSetTopologyContainer::getQuadsAroundVertex(PointID i)
{
    if(!hasQuadsAroundVertex())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        sout << "Warning. [QuadSetTopologyContainer::getQuadsAroundVertex] quad vertex shell array is empty." << sendl;
#endif
        createQuadsAroundVertexArray();
    }
    else if( i >= m_quadsAroundVertex.size())
    {
#ifndef NDEBUG
        sout << "Error. [QuadSetTopologyContainer::getQuadsAroundVertex] index out of bounds." << sendl;
#endif
        createQuadsAroundVertexArray();
    }

    return m_quadsAroundVertex[i];
}

const QuadSetTopologyContainer::QuadsAroundEdge& QuadSetTopologyContainer::getQuadsAroundEdge(EdgeID i)
{
    if(!hasQuadsAroundEdge())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        sout << "Warning. [QuadSetTopologyContainer::getQuadsAroundEdge] quad edge shell array is empty." << sendl;
#endif
        createQuadsAroundEdgeArray();
    }
    else if( i >= m_quadsAroundEdge.size())
    {
#ifndef NDEBUG
        sout << "Error. [QuadSetTopologyContainer::getQuadsAroundEdge] index out of bounds." << sendl;
#endif
        createQuadsAroundEdgeArray();
    }

    return m_quadsAroundEdge[i];
}

const QuadSetTopologyContainer::EdgesInQuad &QuadSetTopologyContainer::getEdgesInQuad(const unsigned int i)
{
    if(m_edgesInQuad.empty())
        createEdgesInQuadArray();

    if( i >= m_edgesInQuad.size())
    {
#ifndef NDEBUG
        sout << "Error. [QuadSetTopologyContainer::getEdgesInQuad] index out of bounds." << sendl;
#endif
        createEdgesInQuadArray();
    }

    return m_edgesInQuad[i];
}

int QuadSetTopologyContainer::getVertexIndexInQuad(const Quad &t, unsigned int vertexIndex) const
{
    if(t[0]==vertexIndex)
        return 0;
    else if(t[1]==vertexIndex)
        return 1;
    else if(t[2]==vertexIndex)
        return 2;
    else if(t[3]==vertexIndex)
        return 3;
    else
        return -1;
}

int QuadSetTopologyContainer::getEdgeIndexInQuad(const EdgesInQuad &t, unsigned int edgeIndex) const
{
    if(t[0]==edgeIndex)
        return 0;
    else if(t[1]==edgeIndex)
        return 1;
    else if(t[2]==edgeIndex)
        return 2;
    else if(t[3]==edgeIndex)
        return 3;
    else
        return -1;
}

sofa::helper::vector< unsigned int > &QuadSetTopologyContainer::getQuadsAroundEdgeForModification(const unsigned int i)
{
    if(!hasQuadsAroundEdge())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        sout << "Warning. [QuadSetTopologyContainer::getQuadsAroundEdgeForModification] quad edge shell array is empty." << sendl;
#endif
        createQuadsAroundEdgeArray();
    }

    if( i >= m_quadsAroundEdge.size())
    {
#ifndef NDEBUG
        sout << "Error. [QuadSetTopologyContainer::getQuadsAroundEdgeForModification] index out of bounds." << sendl;
#endif
        createQuadsAroundEdgeArray();
    }

    return m_quadsAroundEdge[i];
}

sofa::helper::vector< unsigned int > &QuadSetTopologyContainer::getQuadsAroundVertexForModification(const unsigned int i)
{
    if(!hasQuadsAroundVertex())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        sout << "Warning. [QuadSetTopologyContainer::getQuadsAroundVertexForModification] quad vertex shell array is empty." << sendl;
#endif
        createQuadsAroundVertexArray();
    }

    if( i >= m_quadsAroundVertex.size())
    {
#ifndef NDEBUG
        sout << "Error. [QuadSetTopologyContainer::getQuadsAroundVertexForModification] index out of bounds." << sendl;
#endif
        createQuadsAroundVertexArray();
    }

    return m_quadsAroundVertex[i];
}


bool QuadSetTopologyContainer::checkTopology() const
{
#ifndef NDEBUG
    bool ret = true;
    helper::ReadAccessor< Data< sofa::helper::vector<Quad> > > m_quad = d_quad;

    if(hasQuadsAroundVertex())
    {
        for (size_t i=0; i<m_quadsAroundVertex.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &tvs = m_quadsAroundVertex[i];
            for (size_t j=0; j<tvs.size(); ++j)
            {
                if((m_quad[tvs[j]][0]!=i)
                   && (m_quad[tvs[j]][1]!=i)
                   && (m_quad[tvs[j]][2]!=i)
                   && (m_quad[tvs[j]][3]!=i))
                {
                    ret = false;
                    std::cout << "*** CHECK FAILED : check_quad_vertex_shell, i = " << i << " , j = " << j << std::endl;
                }
            }
        }
    }

    if(hasQuadsAroundEdge())
    {
        for (size_t i=0; i<m_quadsAroundEdge.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &tes = m_quadsAroundEdge[i];
            for (size_t j=0; j<tes.size(); ++j)
            {
                if((m_edgesInQuad[tes[j]][0]!=i)
                   && (m_edgesInQuad[tes[j]][1]!=i)
                   && (m_edgesInQuad[tes[j]][2]!=i)
                   && (m_edgesInQuad[tes[j]][3]!=i))
                {
                    ret = false;
                    std::cout << "*** CHECK FAILED : check_quad_edge_shell, i = " << i << " , j = " << j << std::endl;
                }
            }
        }
    }

    return ret && EdgeSetTopologyContainer::checkTopology();
#else
    return true;
#endif
}



/// Get information about connexity of the mesh
/// @{

bool QuadSetTopologyContainer::checkConnexity()
{

    size_t nbr = this->getNbQuads();

    if (nbr == 0)
    {
#ifndef NDEBUG
        serr << "Warning. [QuadSetTopologyContainer::checkConnexity] Can't compute connexity as there are no Quads" << sendl;
#endif
        return false;
    }

    VecQuadID elemAll = this->getConnectedElement(0);

    if (elemAll.size() != nbr)
    {
        serr << "Warning: in computing connexity, Quads are missings. There is more than one connexe component." << sendl;
        return false;
    }

    return true;
}


unsigned int QuadSetTopologyContainer::getNumberOfConnectedComponent()
{
    size_t nbr = this->getNbQuads();

    if (nbr == 0)
    {
#ifndef NDEBUG
        serr << "Warning. [QuadSetTopologyContainer::getNumberOfConnectedComponent] Can't compute connexity as there are no Quads" << sendl;
#endif
        return 0;
    }

    VecQuadID elemAll = this->getConnectedElement(0);
    unsigned int cpt = 1;

    while (elemAll.size() < nbr)
    {
        std::sort(elemAll.begin(), elemAll.end());
        QuadID other_QuadID = (QuadID)elemAll.size();

        for (QuadID i = 0; i<elemAll.size(); ++i)
            if (elemAll[i] != i)
            {
                other_QuadID = i;
                break;
            }

        VecQuadID elemTmp = this->getConnectedElement(other_QuadID);
        cpt++;

        elemAll.insert(elemAll.begin(), elemTmp.begin(), elemTmp.end());
    }

    return cpt;
}


const QuadSetTopologyContainer::VecQuadID QuadSetTopologyContainer::getConnectedElement(QuadID elem)
{
    if(!hasQuadsAroundVertex())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        serr << "Warning. [QuadSetTopologyContainer::getConnectedElement] Quad vertex shell array is empty." << sendl;
#endif
        createQuadsAroundVertexArray();
    }

    VecQuadID elemAll;
    VecQuadID elemOnFront, elemPreviousFront, elemNextFront;
    bool end = false;
    size_t cpt = 0;
    size_t nbr = this->getNbQuads();

    // init algo
    elemAll.push_back(elem);
    elemOnFront.push_back(elem);
    elemPreviousFront.clear();
    cpt++;

    while (!end && cpt < nbr)
    {
        // First Step - Create new region
        elemNextFront = this->getElementAroundElements(elemOnFront); // for each QuadID on the propagation front

        // Second Step - Avoid backward direction
        for (size_t i = 0; i<elemNextFront.size(); ++i)
        {
            bool find = false;
            QuadID id = elemNextFront[i];

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
#ifndef NDEBUG
            serr << "Loop for computing connexity has reach end." << sendl;
#endif
        }

        // iterate
        elemOnFront = elemPreviousFront;
        elemPreviousFront.clear();
    }

    return elemAll;
}


const QuadSetTopologyContainer::VecQuadID QuadSetTopologyContainer::getElementAroundElement(QuadID elem)
{
    VecQuadID elems;

    if (!hasQuadsAroundVertex())
    {
#ifndef NDEBUG
        serr << "Warning. [QuadSetTopologyContainer::getElementAroundElement] Quad vertex shell array is empty." << sendl;
#endif
        createQuadsAroundVertexArray();
    }

    Quad the_quad = this->getQuad(elem);

    for(size_t i = 0; i<4; ++i) // for each node of the Quad
    {
        QuadsAroundVertex quadAV = this->getQuadsAroundVertex(the_quad[i]);

        for (size_t j = 0; j<quadAV.size(); ++j) // for each Quad around the node
        {
            bool find = false;
            QuadID id = quadAV[j];

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


const QuadSetTopologyContainer::VecQuadID QuadSetTopologyContainer::getElementAroundElements(VecQuadID elems)
{
    VecQuadID elemAll;
    VecQuadID elemTmp;

    if (!hasQuadsAroundVertex())
    {
#ifndef NDEBUG
        serr << "Warning. [QuadSetTopologyContainer::getElementAroundElements] Quad vertex shell array is empty." << sendl;
#endif
        createQuadsAroundVertexArray();
    }

    for (size_t i = 0; i <elems.size(); ++i) // for each QuadId of input vector
    {
        VecQuadID elemTmp2 = this->getElementAroundElement(elems[i]);

        elemTmp.insert(elemTmp.end(), elemTmp2.begin(), elemTmp2.end());
    }

    for (size_t i = 0; i<elemTmp.size(); ++i) // for each Quad Id found
    {
        bool find = false;
        QuadID id = elemTmp[i];

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




bool QuadSetTopologyContainer::hasQuads() const
{
    d_quad.updateIfDirty();
    return !(d_quad.getValue()).empty();
}

bool QuadSetTopologyContainer::hasEdgesInQuad() const
{
    return !m_edgesInQuad.empty();
}

bool QuadSetTopologyContainer::hasQuadsAroundVertex() const
{
    return !m_quadsAroundVertex.empty();
}

bool QuadSetTopologyContainer::hasQuadsAroundEdge() const
{
    return !m_quadsAroundEdge.empty();
}

void QuadSetTopologyContainer::clearQuadsAroundVertex()
{
    for(size_t i=0; i<m_quadsAroundVertex.size(); ++i)
        m_quadsAroundVertex[i].clear();

    m_quadsAroundVertex.clear();
}

void QuadSetTopologyContainer::clearQuadsAroundEdge()
{
    for(size_t i=0; i<m_quadsAroundEdge.size(); ++i)
        m_quadsAroundEdge[i].clear();

    m_quadsAroundEdge.clear();
}

void QuadSetTopologyContainer::clearEdgesInQuad()
{
    m_edgesInQuad.clear();
}

void QuadSetTopologyContainer::clearQuads()
{
    helper::WriteAccessor< Data< sofa::helper::vector<Quad> > > m_quad = d_quad;
    m_quad.clear();
}

void QuadSetTopologyContainer::clear()
{
    clearQuadsAroundVertex();
    clearQuadsAroundEdge();
    clearEdgesInQuad();
    clearQuads();

    EdgeSetTopologyContainer::clear();
}

void QuadSetTopologyContainer::updateTopologyEngineGraph()
{
    // calling real update Data graph function implemented once in PointSetTopologyModifier
    this->updateDataEngineGraph(this->d_quad, this->m_enginesList);

    // will concatenate with edges one:
    EdgeSetTopologyContainer::updateTopologyEngineGraph();
}

} // namespace topology

} // namespace component

} // namespace sofa

