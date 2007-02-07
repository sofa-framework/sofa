#ifndef SOFA_COMPONENT_TOPOLOGY_EDGESETTOPOLOGY_INL
#define SOFA_COMPONENT_TOPOLOGY_EDGESETTOPOLOGY_INL

#include <sofa/component/topology/EdgeSetTopology.h>
#include <sofa/component/topology/PointSetTopology.inl>
#include <sofa/component/topology/TopologyChangedEvent.h>
#include <sofa/simulation/tree/PropagateEventAction.h>
#include <sofa/simulation/tree/GNode.h>
#include <algorithm>
#include <functional>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::behavior;



/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////EdgeSetTopologyModifier//////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////



template<class DataTypes>
class EdgeSetTopologyLoader : public PointSetTopologyLoader<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;

    VecCoord pointArray;
    EdgeSetTopologyModifier<DataTypes> *estm;

    EdgeSetTopologyLoader(EdgeSetTopologyModifier<DataTypes> *tm) :PointSetTopologyLoader<DataTypes>(), estm(tm)
    {
    }

    virtual void addLine(int p1, int p2)
    {
        estm->addEdge(Edge((unsigned int)p1,(unsigned int)p2));
    }
};
template<class DataTypes>
void EdgeSetTopologyModifier<DataTypes>::addEdge(Edge e)
{

    EdgeSetTopology<DataTypes> *topology = dynamic_cast<EdgeSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    EdgeSetTopologyContainer * container = static_cast<EdgeSetTopologyContainer *>( topology->getTopologyContainer() );
    assert (container != 0);
    container->m_edge.push_back(e);

}
template<class DataTypes>
bool EdgeSetTopologyModifier<DataTypes>::load(const char *filename)
{

    EdgeSetTopologyLoader<DataTypes> loader(this);
    if (!loader.load(filename))
        return false;
    else
    {
        loadPointSet(&loader);
        return true;
    }
}

template<class DataTypes>
void EdgeSetTopologyModifier<DataTypes>::addEdgesProcess(const std::vector< Edge > &edges)
{
    EdgeSetTopology<DataTypes> *topology = dynamic_cast<EdgeSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    EdgeSetTopologyContainer * container = static_cast<EdgeSetTopologyContainer *>( topology->getTopologyContainer() );
    assert (container != 0);
    if (container->m_edge.size()>0)
    {
        const std::vector< std::vector<unsigned int> > &sa=container->getEdgeShellsArray();
        for (unsigned int i = 0; i < edges.size(); ++i)
        {
            const Edge &e = edges[i];
            // check if the 2 vertices are different
            assert(e.first!=e.second);
            // check if there already exists an edge
            assert(container->getEdgeIndex(e.first,e.second)== -1);
            container->m_edge.push_back(e);
            if (sa.size()>0)
            {
                container->getEdgeShellForModification( e.first ).push_back( container->m_edge.size() - 1 );
                container->getEdgeShellForModification( e.second ).push_back( container->m_edge.size() - 1 );
            }
        }
    }
}



template<class DataTypes>
void EdgeSetTopologyModifier<DataTypes>::addEdgesWarning(const unsigned int nEdges, const std::vector< Edge >& edgesList,
        const std::vector< unsigned int >& edgesIndexList,
        const std::vector< std::vector< unsigned int > > & ancestors,
        const std::vector< std::vector< double > >& baryCoefs)
{
    // Warning that edges just got created
    EdgesAdded *e=new EdgesAdded(nEdges, edgesList,edgesIndexList,ancestors,baryCoefs);
    this->addTopologyChange(e);
}




template<class DataTypes>
void EdgeSetTopologyModifier<DataTypes>::removeEdgesWarning( std::vector<unsigned int> &edges )
{
    /// sort vertices to remove in a descendent order
    std::sort( edges.begin(), edges.end(), std::greater<unsigned int>() );

    // Warning that these edges will be deleted
    EdgesRemoved *e=new EdgesRemoved(edges);
    this->addTopologyChange(e);
}



template<class DataTypes>
void EdgeSetTopologyModifier<DataTypes>::removeEdgesProcess(const unsigned int , const std::vector<unsigned int> &indices)
{
    EdgeSetTopology<DataTypes> *topology = dynamic_cast<EdgeSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    EdgeSetTopologyContainer * container = static_cast<EdgeSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);

    for (unsigned int i = 0; i < indices.size(); ++i)
    {
        Edge *e = &container->m_edge[ indices[i] ];
        unsigned int point1 = e->first, point2 = e->second;
        // first check that the edge shell array has been initialized
        if (container->m_edgeShell.size()>0)
        {

            std::vector< unsigned int > &shell = container->m_edgeShell[ point1 ];

            // removes the first occurence (should be the only one) of the edge in the edge shell of the point
            assert(std::find( shell.begin(), shell.end(), indices[i] ) !=shell.end());
            shell.erase( std::find( shell.begin(), shell.end(), indices[i] ) );

            std::vector< unsigned int > &shell2 = container->m_edgeShell[ point2 ];

            // removes the first occurence (should be the only one) of the edge in the edge shell of the other point
            assert(std::find( shell2.begin(), shell2.end(), indices[i] ) !=shell2.end());
            shell2.erase( std::find( shell2.begin(), shell2.end(), indices[i] ) );
        }

        // removes the edge from the edgelist
        container->m_edge[ indices[i] ] = container->m_edge[ container->m_edge.size() - 1 ]; // overwriting with last valid value.
        container->m_edge.resize( container->m_edge.size() - 1 ); // resizing to erase multiple occurence of the edge.

        // now updates the shell information of the edge formely at the end of the array
        // first check that the edge shell array has been initialized
        if ( indices[i] < container->m_edge.size() )
        {

            unsigned int oldEdgeIndex=container->m_edge.size();
            e = &container->m_edge[ indices[i] ];
            point1 = e->first; point2 = e->second;

            //replaces the edge index oldEdgeIndex with indices[i] for the first vertex
            std::vector< unsigned int > &shell3 = container->m_edgeShell[ point1 ];
            assert(std::find( shell3.begin(), shell3.end(), oldEdgeIndex ) !=shell3.end());
            std::vector< unsigned int >::iterator it=std::find( shell3.begin(), shell3.end(), oldEdgeIndex );
            (*it)=indices[i];

            //replaces the edge index oldEdgeIndex with indices[i] for the second vertex
            std::vector< unsigned int > &shell4 = container->m_edgeShell[ point2 ];
            assert(std::find( shell4.begin(), shell4.end(), oldEdgeIndex ) !=shell4.end());
            it=std::find( shell4.begin(), shell4.end(), oldEdgeIndex );
            (*it)=indices[i];
        }
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
    EdgeSetTopology<DataTypes> *topology = dynamic_cast<EdgeSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    EdgeSetTopologyContainer * container = static_cast<EdgeSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    container->m_edgeShell.resize( container->m_edgeShell.size() + nPoints );
}



template< class DataTypes >
void EdgeSetTopologyModifier< DataTypes >::removePointsProcess(const unsigned int nPoints, std::vector<unsigned int> &indices)
{
    // start by calling the standard method.
    PointSetTopologyModifier< DataTypes >::removePointsProcess( nPoints, indices );

    // now update the local container structures
    EdgeSetTopology<DataTypes> *topology = dynamic_cast<EdgeSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    EdgeSetTopologyContainer * container = static_cast<EdgeSetTopologyContainer *>(topology->getTopologyContainer());
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
    EdgeSetTopology<DataTypes> *topology = dynamic_cast<EdgeSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    EdgeSetTopologyContainer * container = static_cast<EdgeSetTopologyContainer *>(topology->getTopologyContainer());
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
void EdgeSetTopologyModifier< DataTypes >::fuseEdgesProcess(const std::vector< std::pair< unsigned int, unsigned int > >& edgesPair)
{
    EdgeSetTopology< DataTypes > *topology = dynamic_cast< EdgeSetTopology< DataTypes >* >(this->m_basicTopology);
    assert(topology != 0);

    EdgeSetTopologyContainer * container = static_cast< EdgeSetTopologyContainer* >(topology->getTopologyContainer());
    EdgeSetTopologyModifier< DataTypes >* modifier  =this;
    assert(container != 0 );


    // first create the edges
    std::vector< Edge > v;
    std::vector< unsigned int > edgeIndexList;
    std::vector<std::vector<unsigned int> > ancestorsArray;
    unsigned int nbEdges=container->getNumberOfEdges();

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
        edgeIndexList.push_back(nbEdges+i);
        std::vector<unsigned int> ancestors;
        ancestors[0]=i1;
        ancestors[1]=i2;
        ancestorsArray.push_back(ancestors);

    }

    modifier->addEdgesProcess( v );


    // now warn about the creation
    modifier->addEdgesWarning( v.size(), v,edgeIndexList,ancestorsArray);

    //   EdgesAdded ea( 2 * edgesPair.size(), v );
    // addTopologyChange( ea );

    // now warn about the destruction of the old edges
    std::vector< unsigned int > indices;
    for (unsigned int i = 0; i < edgesPair.size(); ++i)
    {
        indices.push_back( edgesPair[i].first  );
        indices.push_back( edgesPair[i].second );
    }
    modifier->removeEdgesWarning(indices );

//            EdgesRemoved er( indices );
//            addTopologyChange( er );

    // propagate the warnings
    this->m_basicTopology->propagateTopologicalChanges();


    // now destroy the old edges.
    modifier->removeEdgesProcess( indices.size(), indices );

    // TODO : final warning?

}



template<class DataTypes>
void EdgeSetTopologyModifier< DataTypes >::splitEdgesProcess( std::vector<unsigned int> &indices,
        const std::vector< std::vector< double > >& baryCoefs)
{
    EdgeSetTopology< DataTypes > *topology = dynamic_cast<EdgeSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    EdgeSetTopologyContainer * container = static_cast< EdgeSetTopologyContainer* >(topology->getTopologyContainer());
    EdgeSetTopologyModifier< DataTypes >* modifier  = this;
    assert(container != 0);

    std::vector< std::vector< double > > defaultBaryCoefs;

    std::vector< std::vector< unsigned int > > v;
    v.resize(indices.size());
    std::vector< Edge >  edges;
    std::vector< unsigned int >  edgesIndex;
    for (unsigned int i = 0; i < indices.size(); ++i)
    {
        unsigned int p1 = container->getEdge( indices[i] ).first,
                     p2 = container->getEdge( indices[i] ).second;

        // Adding the new point
        v[i].push_back( p1 );
        v[i].push_back( p2 );

        // Adding the new Edges
        Edge e1( p1, topology->getDOFNumber() + i ),
             e2( p2, topology->getDOFNumber() + i );
        edges.push_back( e1 );
        edges.push_back( e2 );
        edgesIndex.push_back(container->getNumberOfEdges()+i);
    }
    if (baryCoefs.size()==0)
    {
        defaultBaryCoefs.resize(indices.size());
        for (unsigned int i = 0; i < indices.size(); ++i)
        {
            defaultBaryCoefs[i].push_back(0.5);
            defaultBaryCoefs[i].push_back(0.5);
        }
        modifier->addPointsProcess( indices.size(), v ,defaultBaryCoefs);

    }
    else
        modifier->addPointsProcess( indices.size(), v ,baryCoefs);
    modifier->addEdgesProcess( edges );


    // warn about addedd points and edges
    if (baryCoefs.size()==0)
        modifier->addPointsWarning( indices.size(), v ,defaultBaryCoefs);
    else
        modifier->addPointsWarning( indices.size(), v , baryCoefs);
    modifier->addEdgesWarning( edges.size(),edges,edgesIndex);

    // warn about old edges about to be removed
    modifier->removeEdgesWarning( indices);

    this->m_basicTopology->propagateTopologicalChanges();

    // Removing the old edges
    modifier->removeEdgesProcess( indices.size(), indices );
}

template<class DataTypes>
void EdgeSetTopologyAlgorithms< DataTypes >::removeEdges(std::vector< unsigned int >& edges)
{
    EdgeSetTopology< DataTypes > *topology = dynamic_cast<EdgeSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    EdgeSetTopologyModifier< DataTypes >* modifier  = static_cast< EdgeSetTopologyModifier< DataTypes >* >(topology->getTopologyModifier());
    assert(modifier != 0);
    /// add the topological changes in the queue
    modifier->removeEdgesWarning(edges);
    // inform other objects that the edges are going to be removed
    topology->propagateTopologicalChanges();
    // now destroy the old edges.
    modifier->removeEdgesProcess( edges.size(), edges );
}
template<class DataTypes>
void EdgeSetTopologyAlgorithms< DataTypes >::addEdges(const std::vector< Edge >& edges,
        const std::vector< std::vector< unsigned int > > & ancestors ,
        const std::vector< std::vector< double > >& baryCoefs)
{
    EdgeSetTopology< DataTypes > *topology = dynamic_cast<EdgeSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    EdgeSetTopologyModifier< DataTypes >* modifier  = static_cast< EdgeSetTopologyModifier< DataTypes >* >(topology->getTopologyModifier());
    assert(modifier != 0);
    unsigned int nEdges=topology->getEdgeSetTopologyContainer()->getNumberOfEdges();

    /// actually add edges in the topology container
    modifier->addEdgesProcess(edges);

    std::vector<unsigned int> edgesIndex;
    unsigned int i;
    for (i=0; i<edges.size(); ++i)
    {
        edgesIndex[i]=nEdges+i;
    }
    // add topology event in the stack of topological events
    modifier->addEdgesWarning( edges.size(), edges,edgesIndex,ancestors,baryCoefs);
    // inform other objects that the edges are already added
    topology->propagateTopologicalChanges();

}
template<class DataTypes>
void EdgeSetTopologyAlgorithms< DataTypes >::fuseEdges(const std::vector< std::pair< unsigned int, unsigned int > >& edgesPair)
{
    EdgeSetTopology< DataTypes > *topology = dynamic_cast<EdgeSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    EdgeSetTopologyModifier< DataTypes >* modifier  = static_cast< EdgeSetTopologyModifier< DataTypes >* >(topology->getTopologyModifier());
    assert(modifier != 0);
    modifier->fuseEdgesProcess(edgesPair);
}

template<class DataTypes>
void EdgeSetTopologyAlgorithms< DataTypes >::splitEdges( std::vector<unsigned int> &indices,
        const std::vector< std::vector< double > >& baryCoefs)
{
    EdgeSetTopology< DataTypes > *topology = dynamic_cast<EdgeSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    EdgeSetTopologyModifier< DataTypes >* modifier  = static_cast< EdgeSetTopologyModifier< DataTypes >* >(topology->getTopologyModifier());
    assert(modifier != 0);
    modifier->splitEdgesProcess(indices,baryCoefs);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////EdgeSetGeometryAlgorithms//////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
template< class DataTypes>
typename DataTypes::Real EdgeSetGeometryAlgorithms< DataTypes >::computeEdgeLength( const unsigned int i) const
{
    EdgeSetTopology< DataTypes > *topology = dynamic_cast<EdgeSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    EdgeSetTopologyContainer * container = static_cast< EdgeSetTopologyContainer* >(topology->getTopologyContainer());
    const Edge &e=container->getEdge(i);
    const VecCoord& p = *topology->getDOF()->getX();
    Real length=(p[e.first]-p[e.second]).norm();
    return length;
}
template< class DataTypes>
typename DataTypes::Real EdgeSetGeometryAlgorithms< DataTypes >::computeRestEdgeLength( const unsigned int i) const
{
    EdgeSetTopology< DataTypes > *topology = dynamic_cast<EdgeSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    EdgeSetTopologyContainer * container = static_cast< EdgeSetTopologyContainer* >(topology->getTopologyContainer());
    const Edge &e=container->getEdge(i);
    const VecCoord& p = *topology->getDOF()->getX0();
    Real length=(p[e.first]-p[e.second]).norm();
    return length;
}
/// computes the edge length of all edges are store in the array interface
template<class DataTypes>
void EdgeSetGeometryAlgorithms<DataTypes>::computeEdgeLength( BasicArrayInterface<Real> &ai) const
{
    EdgeSetTopology< DataTypes > *topology = dynamic_cast<EdgeSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    EdgeSetTopologyContainer * container = static_cast< EdgeSetTopologyContainer* >(topology->getTopologyContainer());
    const std::vector<Edge> &ea=container->getEdgeArray();
    const typename DataTypes::VecCoord& p = *topology->getDOF()->getX();
    unsigned int i;
    for (i=0; i<ea.size(); ++i)
    {
        const Edge &e=ea[i];
        ai[i]=(p[e.first]-p[e.second]).norm();
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////EdgeSetTopology//////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
template<class DataTypes>
void EdgeSetTopology<DataTypes>::init()
{
}
template<class DataTypes>
EdgeSetTopology<DataTypes>::EdgeSetTopology(MechanicalObject<DataTypes> *obj) : PointSetTopology<DataTypes>( obj,(PointSetTopology<DataTypes> *)0)
{
    this->m_topologyContainer= new EdgeSetTopologyContainer(this);
    this->m_topologyModifier= new EdgeSetTopologyModifier<DataTypes>(this);
    this->m_topologyAlgorithms= new EdgeSetTopologyAlgorithms<DataTypes>(this);
    this->m_geometryAlgorithms= new EdgeSetGeometryAlgorithms<DataTypes>(this);
}


} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_EDGESETTOPOLOGY_INL
