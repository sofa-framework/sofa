/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_MANIFOLDEDGESETTOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_TOPOLOGY_MANIFOLDEDGESETTOPOLOGYCONTAINER_H
#include <ManifoldTopologies/config.h>

#include <ManifoldTopologies/config.h>

#include <SofaBaseTopology/EdgeSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace topology
{
class ManifoldEdgeSetTopologyModifier;

using core::topology::BaseMeshTopology;

typedef BaseMeshTopology::PointID			PointID;
typedef BaseMeshTopology::EdgeID			EdgeID;
typedef BaseMeshTopology::Edge				Edge;
typedef BaseMeshTopology::SeqEdges			SeqEdges;
typedef BaseMeshTopology::EdgesAroundVertex		EdgesAroundVertex;

/** a class that stores a set of edges and provides access to the adjacency between points and edges.
  this topology is constraint by the manifold property : each vertex is adjacent either to one vertex or to two vertices. */
class SOFA_MANIFOLD_TOPOLOGIES_API ManifoldEdgeSetTopologyContainer : public EdgeSetTopologyContainer
{
    friend class ManifoldEdgeSetTopologyModifier;

public:
    SOFA_CLASS(ManifoldEdgeSetTopologyContainer,EdgeSetTopologyContainer);

    ManifoldEdgeSetTopologyContainer();

    virtual ~ManifoldEdgeSetTopologyContainer() {}

    virtual void init();

    virtual void clear();

    /** \brief Checks if the topology is coherent
    *
    * Check if the shell arrays are coherent
    */
    virtual bool checkTopology() const;

    /** \brief Returns the number of connected components from the graph containing all edges and give, for each vertex, which component it belongs to  (use BOOST GRAPH LIBRAIRY)
    @param components the array containing the optimal vertex permutation according to the Reverse CuthillMckee algorithm
    */
    virtual int getNumberConnectedComponents(sofa::helper::vector<unsigned int>& components);

protected:
    /** \brief Creates the EdgeSetIndex.
    *
    * This function is only called if the EdgeShell member is required.
    * EdgeShell[i] contains the indices of all edges having the ith DOF as
    * one of their ends.
    */
    virtual void createEdgesAroundVertexArray();

private:
    // Describe each connected component, which can be seen as an oriented line
    class ConnectedComponent
    {
    public:

        ConnectedComponent(unsigned int FirstVertexIndex=-1, unsigned int LastVertexIndex=-1, unsigned int size=0,unsigned int ccIndex=0)
            : FirstVertexIndex(FirstVertexIndex), LastVertexIndex(LastVertexIndex), size(size), ccIndex(ccIndex)
        {}

        virtual ~ConnectedComponent() {}

    public:
        unsigned int FirstVertexIndex; // index of the first vertex on the line
        unsigned int LastVertexIndex; // index of the last vertex on the line

        unsigned int size; // number of the vertices on the line

        unsigned int ccIndex; // index of the connected component stored in the m_ConnectedComponentArray
    };

    /** \brief Resets the array of connected components and the ComponentVertex array (which are not valide anymore).
    *
    */
    void resetConnectedComponent()
    {
        m_ComponentVertexArray.clear();
        m_ConnectedComponentArray.clear();
    }

    /** \brief Returns true iff the array of connected components and the ComponentVertex array are valide (ie : not void)
    *
    */
    bool isvoid_ConnectedComponent()
    {
        return m_ConnectedComponentArray.size()==0;
    }

    /** \brief Computes the array of connected components and the ComponentVertex array (which makes them valide).
    *
    */
    void computeConnectedComponent();

    /** \brief Returns the number of connected components.
    *
    */
    virtual int getNumberOfConnectedComponents()
    {
        computeConnectedComponent();
        return m_ConnectedComponentArray.size();
    }

    /** \brief Returns the FirstVertexIndex of the ith connected component.
    *
    */
    virtual int getFirstVertexIndex(unsigned int i)
    {
        computeConnectedComponent();
        assert(i<m_ConnectedComponentArray.size());
        return m_ConnectedComponentArray[i].FirstVertexIndex;
    }

    /** \brief Returns the LastVertexIndex of the ith connected component.
    *
    */
    virtual int getLastVertexIndex(unsigned int i)
    {
        computeConnectedComponent();
        assert(i<m_ConnectedComponentArray.size());
        return m_ConnectedComponentArray[i].LastVertexIndex;
    }

    /** \brief Returns the size of the ith connected component.
    *
    */
    virtual int getComponentSize(unsigned int i)
    {
        computeConnectedComponent();
        assert(i<m_ConnectedComponentArray.size());
        return m_ConnectedComponentArray[i].size;
    }

    /** \brief Returns the index of the ith connected component.
    *
    */
    virtual int getComponentIndex(unsigned int i)
    {
        computeConnectedComponent();
        assert(i<m_ConnectedComponentArray.size());
        return m_ConnectedComponentArray[i].ccIndex;
    }

    /** \brief Returns true iff the ith connected component is closed (ie : iff FirstVertexIndex == LastVertexIndex).
    *
    */
    virtual bool isComponentClosed(unsigned int i)
    {
        computeConnectedComponent();
        assert(i<m_ConnectedComponentArray.size());
        return (m_ConnectedComponentArray[i].FirstVertexIndex == m_ConnectedComponentArray[i].LastVertexIndex);
    }

    /** \brief Returns the indice of the vertex which is next to the vertex indexed by i.
    */
    int getNextVertex(const unsigned int i)
    {
        assert(getEdgesAroundVertex(i).size()>0);
        if((getEdgesAroundVertex(i)).size()==1 && (getEdge((getEdgesAroundVertex(i))[0]))[1]==i)
        {
            return -1;
        }
        else
        {
            return (getEdge((getEdgesAroundVertex(i))[0]))[1];
        }
    }

    /** \brief Returns the indice of the vertex which is previous to the vertex indexed by i.
    */
    int getPreviousVertex(const unsigned int i)
    {
        assert(getEdgesAroundVertex(i).size()>0);
        if((getEdgesAroundVertex(i)).size()==1 && (getEdge((getEdgesAroundVertex(i))[0]))[0]==i)
        {
            return -1;
        }
        else
        {
            return (getEdge((getEdgesAroundVertex(i))[0]))[0];
        }
    }

    /** \brief Returns the indice of the edge which is next to the edge indexed by i.
    */
    int getNextEdge(const unsigned int i)
    {
        if((getEdgesAroundVertex(getEdge(i)[1])).size()==1)
        {
            return -1;
        }
        else
        {
            return (getEdgesAroundVertex(getEdge(i)[1]))[1];
        }
    }

    /** \brief Returns the indice of the edge which is previous to the edge indexed by i.
    */
    int getPreviousEdge(const unsigned int i)
    {
        if((getEdgesAroundVertex(getEdge(i)[0])).size()==1)
        {
            return -1;
        }
        else
        {
            return (getEdgesAroundVertex(getEdge(i)[0]))[0];
        }
    }

    /** \brief Returns the ComponentVertex array.
    *
    */
    const sofa::helper::vector< unsigned int > &getComponentVertexArray() const
    {
        return m_ComponentVertexArray;
    }

    /** \brief Returns the array of connected components.
    *
    */
    const sofa::helper::vector< ConnectedComponent > &getConnectedComponentArray() const
    {
        return m_ConnectedComponentArray;
    }

private:
    /** The array that stores for each vertex index, the connected component the vertex belongs to */
    sofa::helper::vector< unsigned int > m_ComponentVertexArray;

    /** The array that stores the connected components */
    sofa::helper::vector< ConnectedComponent > m_ConnectedComponentArray;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
