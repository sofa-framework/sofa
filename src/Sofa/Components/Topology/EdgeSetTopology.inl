#ifndef SOFA_COMPONENTS_EDGESETTOPOLOGY_INL
#define SOFA_COMPONENTS_EDGESETTOPOLOGY_INL

#include "EdgeSetTopology.h"
#include "TopologyChangedEvent.h"
#include <Sofa/Components/Graph/PropagateEventAction.h>
#include <Sofa/Components/Graph/GNode.h>
#include <algorithm>
#include <functional>

namespace Sofa
{
namespace Components
{

using namespace Common;
using namespace Sofa::Core;



/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////EdgeSetTopologyModifier//////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
template<class DataTypes>
void EdgeSetTopologyModifier<DataTypes>::addEdgesProcess(const std::vector< Edge > &edges)
{
    EdgeSetTopology<DataTypes> *topology = dynamic_cast<EdgeSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);
    EdgeSetTopologyContainer * container = dynamic_cast<EdgeSetTopologyContainer *>( topology->getTopologyContainer() );
    assert (container != 0);
    for (unsigned int i = 0; i < edges.size(); ++i)
    {
        Edge &e = edges[i];
        container->m_edge.push_back(e);
        container->getEdgeShell( e.first ).push_back( container->m_edge.size() - 1 );
        container->getEdgeShell( e.second ).push_back( container->m_edge.size() - 1 );
    }
}



template<class DataTypes>
void EdgeSetTopologyModifier<DataTypes>::addEdgesWarning(const unsigned int nEdges, const std::vector< std::vector< Edge > > &ancestors)
{
    // Warning that edges just got created
    EdgesAdded e(nEdges, ancestors);
    addTopologyChange(e);
}




template<class DataTypes>
void EdgeSetTopologyModifier<DataTypes>::removeEdgesWarning( const std::vector<unsigned int> &edges )
{
    // Warning that these edges will be deleted
    EdgesRemoved e(edges);
    addTopologyChange(e);
}



template<class DataTypes>
void EdgeSetTopologyModifier<DataTypes>::removeEdgesProcess(const unsigned int nEdges, std::vector<unsigned int> &indices)
{
    std::sort( indices.begin(), indices.end(), greater<unsigned int>() );
    EdgeSetTopology<DataTypes> *topology = dynamic_cast<EdgeSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);
    EdgeSetTopologyContainer * container = dynamic_cast<EdgeSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    for (unsigned int i = 0; i < indices.size(); ++i)
    {
        Edge *e = &container->m_edge[ indices[i] ];
        unsigned int point1 = e->first, point2 = e->second;

        std::vector< unsigned int > shell = container->m_edgeShell[ point1 ];

        // removes the first occurence (should be the only one) of the edge in the edge shell of the point
        shell.erase( std::find( shell.begin(), shell.end(), indices[i] ) );

        shell = container->m_edgeShell[ point2 ];

        // removes the first occurence (should be the only one) of the edge in the edge shell of the other point
        shell.erase( std::find( shell.begin(), shell.end(), indices[i] ) );

        // removes the edge from the edgelist
        container->m_edge[ indices[i] ] = container->m_edge[ container->m_edge.size() - 1 ]; // overwriting with last valid value.
        container->m_edge.resize( container->m_edge.size() - 1 ); // resizing to erase multiple occurence of the edge.

    }
}



template<class DataTypes >
void EdgeSetTopologyModifier< DataTypes >::addPointsProcess(const unsigned int nPoints,
        const std::vector< std::vector< unsigned int > >& ancestors,
        const std::vector< std::vector< double > >& baryCoefs)
{
    // start by calling the standard method.
    PointSetTopologyModifier< DataTypes >::addPointsProcess( nPoints, ancestors, baryCoefs );

    // now update the local container structures.
    EdgeSetTopology<DataTypes> *topology = dynamic_cast<EdgeSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);
    EdgeSetTopologyContainer * container = dynamic_cast<EdgeSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    container->m_edgeShell.resize( container->m_edgeShell.size() + nPoints );
}



template< class DataTypes >
void EdgeSetTopologyModifier< DataTypes >::removePointsProcess(const unsigned int nPoints, std::vector<unsigned int> &indices)
{
    // start by calling the standard method.
    PointSetTopologyModifier< DataTypes >::removePointsProcess( nPoints, indices );

    // now update the local container structures
    EdgeSetTopology<DataTypes> *topology = dynamic_cast<EdgeSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);
    EdgeSetTopologyContainer * container = dynamic_cast<EdgeSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);

    unsigned int lastPoint = container->m_edgeShell.size() - 1;

    for (unsigned int i = 0; i < indices.size(); ++i)
    {
        // updating the edges connected to the point replacing the removed one:
        // for all edges connected to the last point
        for (unsigned int j = 0; j < container->m_edgeShell[lastPoint].size(); ++j)
        {
            // change the old index for the new one
            if ( container->m_edge[ container->m_edgeShell[lastPoint][j] ].first == lastPoint )
                container->m_edge[ container->m_edgeShell[lastPoint][j] ].first = indices[i];
            else
                container->m_edge[ container->m_edgeShell[lastPoint][j] ].second = indices[i];
        }

        // updating the edge shell itself (change the old index for the new one)
        container->m_edgeShell[ indices[i] ] = container->m_edgeShell[ lastPoint ];

        --lastPoint;
    }

    container->m_edgeShell.resize( container->m_edgeShell.size() - indices.size() );
}



template< class DataTypes >
void EdgeSetTopologyModifier< DataTypes >::renumberPointsProcess( const std::vector<unsigned int> &index)
{
    // start by calling the standard method
    PointSetTopologyModifier< DataTypes >::renumberPointsProcess( index );

    // now update the local container structures.
    EdgeSetTopology<DataTypes> *topology = dynamic_cast<EdgeSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);
    EdgeSetTopologyContainer * container = dynamic_cast<EdgeSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);

    std::vector< std::vector< unsigned int > > edgeShell_cp = container->m_edgeShell;
    for (unsigned int i = 0; i < index.size(); ++i)
    {
        container->m_edgeShell[i] = edgeShell_cp[ index[i] ];
    }

    for (unsigned int i = 0; i < container->m_edge.size(); ++i)
    {
        container->m_edge[i].first  = index[ container->m_edge[i].first  ];
        container->m_edge[i].second = index[ container->m_edge[i].second ];
    }


}



template<class DataTypes>
void EdgeSetTopologyAlgorithms< DataTypes >::fuseEdgesProcess(const std::vector< Edge > >& edgesPair)
{
    EdgeSetTopology< DataTypes > *topology = dynamic_cast< EdgeSetTopology< DataTypes >* >(m_basicTopology);
    assert(topology != 0);

    EdgeSetTopologyContainer * container = dynamic_cast< EdgeSetTopologyContainer* >(topology->getTopologyContainer());
    EdgeSetTopologyModifier< DataTypes >* modifier  = dynamic_cast< EdgeSetTopologyModifier< DataTypes >* >(topology->getTopologyModifier());
    assert(container != 0 && modifier != 0);


    // first create the edges
    std::vector< Edge > v;

    for (unsigned int i = 0; i < edgesPair.size(); ++i)
    {
        unsigned int i1 = edgesPair[i].first,
                     i2 = edgesPair[i].second;

        unsigned int p11 = container->getEdge(i1).first,
                     p12 = container->getEdge(i1).second,
                     p21 = container->getEdge(i2).first,
                     p22 = container->getEdge(i2).second;

        Edge e1 (p11, p21),
             e2 (p12, p22);
        v.push_back(e1);
        v.push_back(e2);
    }

    modifier->addEdgesProcess( v );

    // now warn about the creation
    EdgesAdded ea( 2 * edgesPair.size(), v );
    addTopologyChange( ea );

    // now warn about the destruction of the old edges
    std::vector< unsigned int > indices;
    for (unsigned int i = 0; i < edgesPair.size(); ++i)
    {
        indices.push_back( edgesPair[i].first  );
        indices.push_back( edgesPair[i].second );
    }

    EdgesRemoved er( indices );
    addTopologyChange( er );

    // propagate the warnings
    m_basicTopology->propagateTopologicalChanges();


    // now destroy the old edges.
    removeEdgesProcess( indices.size(), indices );

    // TODO : final warning?

}



template<class DataTypes>
void EdgeSetTopologyAlgorithms< DataTypes >::splitEdgesProcess(const std::vector<unsigned int> indices)
{
    EdgeSetTopology< DataTypes > *topology = dynamic_cast<EdgeSetTopology< DataTypes >* >(m_basicTopology);
    assert (topology != 0);
    EdgeSetTopologyContainer * container = dynamic_cast< EdgeSetTopologyContainer* >(topology->getTopologyContainer());
    EdgeSetTopologyModifier< DataTypes >* modifier  = dynamic_cast< EdgeSetTopologyModifier< DataTypes >* >(topology->getTopologyModifier());
    assert(container != 0 && modifier != 0);

    std::vector< std::vector< unsigned int > > v;
    std::vector< std::vector< Edge > > edges;
    for (unsigned int i = 0; i < indices.size(); ++i)
    {
        unsigned int p1 = container->getEdge( indices[i] ).first,
                     p2 = container->getEdge( indices[i] ).second;

        // Adding the new point
        v[i].push_back( p1 );
        v[i].push_back( p2 );

        // Adding the new Edges
        Edge e1( p1, container->getDOFIndexArray().size() + i ),
             e2( p2, container->getDOFIndexArray().size() + i );
        edges[i].push_back( e1 );
        edges[i].push_back( e2 );
    }

    modifier->addPointsProcess( indices.size(), v );
    modifier->addEdgesProcess( edges );


    // warn about addedd points and edges
    PointsAdded pa( indices.size() );
    addTopologyChange( pa );

    EdgesAdded ea( indices.size() * 2 );
    addTopologyChange( ea );

    // warn about old edges about to be removed
    EdgesRemoved er( indices );

    // Removing the old edges
    modifier->removeEdgesProcess( indices.size(), indices );
}



/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////EdgeSetTopology//////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
template<class DataTypes>
void EdgeSetTopology<DataTypes>::init()
{
}


} // namespace Components

} // namespace Sofa

#endif // SOFA_COMPONENTS_EDGESETTOPOLOGY_INL
