#ifndef SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGY_INL
#define SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETTOPOLOGY_INL

#include <sofa/component/topology/TetrahedronSetTopology.h>
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
////////////////////////////////////TetrahedronSetTopologyModifier//////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////


const unsigned int tetrahedronEdgeArray[6][2]= {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};

template<class DataTypes>
class TetrahedronSetTopologyLoader : public PointSetTopologyLoader<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;

    VecCoord pointArray;
    TetrahedronSetTopologyModifier<DataTypes> *tstm;

    TetrahedronSetTopologyLoader(TetrahedronSetTopologyModifier<DataTypes> *tm) :PointSetTopologyLoader<DataTypes>(), tstm(tm)
    {
    }

    virtual void addTetrahedron(int p1, int p2, int p3,int p4)
    {
        tstm->addTetrahedron(Tetrahedron(make_array<unsigned int>((unsigned int)p1,(unsigned int)p2,(unsigned int) p3,(unsigned int) p4)));
    }
};
template<class DataTypes>
void TetrahedronSetTopologyModifier<DataTypes>::addTetrahedron(Tetrahedron t)
{

    TetrahedronSetTopology<DataTypes> *topology = dynamic_cast<TetrahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast<TetrahedronSetTopologyContainer *>( topology->getTopologyContainer() );
    assert (container != 0);
    container->m_tetrahedron.push_back(t);

}
template<class DataTypes>
bool TetrahedronSetTopologyModifier<DataTypes>::load(const char *filename)
{

    TetrahedronSetTopologyLoader<DataTypes> loader(this);
    if (!loader.load(filename))
        return false;
    else
    {
        loadPointSet(&loader);
        return true;
    }
}

template<class DataTypes>
void TetrahedronSetTopologyModifier<DataTypes>::addTetrahedraProcess(const std::vector< Tetrahedron > &tetrahedra)
{
    TetrahedronSetTopology<DataTypes> *topology = dynamic_cast<TetrahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast<TetrahedronSetTopologyContainer *>( topology->getTopologyContainer() );
    assert (container != 0);
    if (container->m_tetrahedron.size()>0)
    {
        unsigned int tetrahedronIndex;
        const std::vector< std::vector<unsigned int> > &tvsa=container->getTetrahedronVertexShellArray();
        const std::vector< std::vector<unsigned int> > &tesa=container->getTetrahedronEdgeShellArray();
        const std::vector< std::vector<unsigned int> > &ttsa=container->getTetrahedronTriangleShellArray();

        unsigned int j;


        for (unsigned int i = 0; i < tetrahedra.size(); ++i)
        {
            const Tetrahedron &t = tetrahedra[i];
            // check if the 3 vertices are different
            assert(t[0]!=t[1]);
            assert(t[0]!=t[2]);
            assert(t[0]!=t[3]);
            assert(t[1]!=t[2]);
            assert(t[1]!=t[3]);
            assert(t[2]!=t[3]);
            // check if there already exists a tetrahedron with the same indices
            assert(container->getTetrahedronIndex(t[0],t[1],t[2],t[3])== -1);
            container->m_tetrahedron.push_back(t);
            tetrahedronIndex=container->m_tetrahedron.size() - 1 ;
            if (tvsa.size()>0)
            {
                container->getTetrahedronVertexShellForModification( t[0] ).push_back( tetrahedronIndex );
                container->getTetrahedronVertexShellForModification( t[1] ).push_back( tetrahedronIndex );
                container->getTetrahedronVertexShellForModification( t[2] ).push_back( tetrahedronIndex );
                container->getTetrahedronVertexShellForModification( t[3] ).push_back( tetrahedronIndex );

            }
            if (container->m_tetrahedronEdge.size()>0)
            {
                int edgeIndex;
                for (j=0; j<6; ++j)
                {
                    edgeIndex=container->getEdgeIndex(tetrahedronEdgeArray[j][0],
                            tetrahedronEdgeArray[j][1]);
                    assert(edgeIndex!= -1);
                    container->m_tetrahedronEdge[tetrahedronIndex][j]= edgeIndex;
                }
            }
            if (container->m_tetrahedronTriangle.size()>0)
            {
                int triangleIndex;
                for (j=0; j<4; ++j)
                {
                    triangleIndex=container->getTriangleIndex(t[(j+1)%4],t[(j+2)%4],
                            t[(j+3)%4]);
                    assert(triangleIndex!= -1);
                    container->m_tetrahedronTriangle[tetrahedronIndex][j]= triangleIndex;
                }
            }
            if (tesa.size()>0)
            {
                for (j=0; j<3; ++j)
                {
                    container->m_tetrahedronEdgeShell[container->m_tetrahedronEdge[tetrahedronIndex][j]].push_back( tetrahedronIndex );
                }
            }
            if (ttsa.size()>0)
            {
                for (j=0; j<3; ++j)
                {
                    container->m_tetrahedronTriangleShell[container->m_tetrahedronTriangle[tetrahedronIndex][j]].push_back( tetrahedronIndex );
                }
            }


        }
    }
}



template<class DataTypes>
void TetrahedronSetTopologyModifier<DataTypes>::addTetrahedraWarning(const unsigned int nTetrahedra, const std::vector< Tetrahedron >& tetrahedraList,
        const std::vector< unsigned int >& tetrahedraIndexList,
        const std::vector< std::vector< unsigned int > > & ancestors,
        const std::vector< std::vector< double > >& baryCoefs)
{
    // Warning that tetrahedra just got created
    TetrahedraAdded *e=new TetrahedraAdded(nTetrahedra, tetrahedraList,tetrahedraIndexList,ancestors,baryCoefs);
    this->addTopologyChange(e);
}




template<class DataTypes>
void TetrahedronSetTopologyModifier<DataTypes>::removeTetrahedraWarning( std::vector<unsigned int> &tetrahedra )
{
    /// sort vertices to remove in a descendent order
    std::sort( tetrahedra.begin(), tetrahedra.end(), std::greater<unsigned int>() );

    // Warning that these edges will be deleted
    TetrahedraRemoved *e=new TetrahedraRemoved(tetrahedra);
    this->addTopologyChange(e);
}



template<class DataTypes>
void TetrahedronSetTopologyModifier<DataTypes>::removeTetrahedraProcess( const std::vector<unsigned int> &indices,const bool removeIsolatedItems)
{
    TetrahedronSetTopology<DataTypes> *topology = dynamic_cast<TetrahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast<TetrahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);

    if (container->m_tetrahedron.size()>0)
    {

        for (unsigned int i = 0; i < indices.size(); ++i)
        {
            Tetrahedron &t = container->m_tetrahedron[ indices[i] ];
            // first check that the tetrahedron vertex shell array has been initialized
            if (container->m_tetrahedronVertexShell.size()>0)
            {

                std::vector< unsigned int > &shell0 = container->m_tetrahedronVertexShell[ t[0] ];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell0.begin(), shell0.end(), indices[i] ) !=shell0.end());
                shell0.erase( std::find( shell0.begin(), shell0.end(), indices[i] ) );

                std::vector< unsigned int > &shell1 = container->m_tetrahedronVertexShell[ t[1] ];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell1.begin(), shell1.end(), indices[i] ) !=shell1.end());
                shell1.erase( std::find( shell1.begin(), shell1.end(), indices[i] ) );

                std::vector< unsigned int > &shell2 = container->m_tetrahedronVertexShell[ t[2] ];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell2.begin(), shell2.end(), indices[i] ) !=shell2.end());
                shell2.erase( std::find( shell2.begin(), shell2.end(), indices[i] ) );

                std::vector< unsigned int > &shell3 = container->m_tetrahedronVertexShell[ t[3] ];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell3.begin(), shell3.end(), indices[i] ) !=shell3.end());
                shell3.erase( std::find( shell3.begin(), shell3.end(), indices[i] ) );

            }

            /** first check that the tetrahedron edge shell array has been initialized */
            if (container->m_tetrahedronEdgeShell.size()>0)
            {
                std::vector< unsigned int > &shell0 = container->m_tetrahedronEdgeShell[ container->m_tetrahedronEdge[indices[i]][0]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell0.begin(), shell0.end(), indices[i] ) !=shell0.end());
                shell0.erase( std::find( shell0.begin(), shell0.end(), indices[i] ) );

                std::vector< unsigned int > &shell1 = container->m_tetrahedronEdgeShell[ container->m_tetrahedronEdge[indices[i]][1]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell1.begin(), shell1.end(), indices[i] ) !=shell1.end());
                shell1.erase( std::find( shell1.begin(), shell1.end(), indices[i] ) );

                std::vector< unsigned int > &shell2 = container->m_tetrahedronEdgeShell[ container->m_tetrahedronEdge[indices[i]][2]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell2.begin(), shell2.end(), indices[i] ) !=shell2.end());
                shell2.erase( std::find( shell2.begin(), shell2.end(), indices[i] ) );

                std::vector< unsigned int > &shell3 = container->m_tetrahedronEdgeShell[ container->m_tetrahedronEdge[indices[i]][3]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell3.begin(), shell3.end(), indices[i] ) !=shell3.end());
                shell3.erase( std::find( shell3.begin(), shell3.end(), indices[i] ) );

            }
            /** first check that the tetrahedron triangle shell array has been initialized */
            if (container->m_tetrahedronTriangleShell.size()>0)
            {
                std::vector< unsigned int > &shell0 = container->m_tetrahedronTriangleShell[ container->m_tetrahedronTriangle[indices[i]][0]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell0.begin(), shell0.end(), indices[i] ) !=shell0.end());
                shell0.erase( std::find( shell0.begin(), shell0.end(), indices[i] ) );

                std::vector< unsigned int > &shell1 = container->m_tetrahedronTriangleShell[ container->m_tetrahedronTriangle[indices[i]][1]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell1.begin(), shell1.end(), indices[i] ) !=shell1.end());
                shell1.erase( std::find( shell1.begin(), shell1.end(), indices[i] ) );

                std::vector< unsigned int > &shell2 = container->m_tetrahedronTriangleShell[ container->m_tetrahedronTriangle[indices[i]][2]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell2.begin(), shell2.end(), indices[i] ) !=shell2.end());
                shell2.erase( std::find( shell2.begin(), shell2.end(), indices[i] ) );

                std::vector< unsigned int > &shell3 = container->m_tetrahedronTriangleShell[ container->m_tetrahedronTriangle[indices[i]][3]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell3.begin(), shell3.end(), indices[i] ) !=shell3.end());
                shell3.erase( std::find( shell3.begin(), shell3.end(), indices[i] ) );

            }

            // removes the tetrahedron from the tetrahedronArray
            container->m_tetrahedron[ indices[i] ] = container->m_tetrahedron[ container->m_tetrahedron.size() - 1 ]; // overwriting with last valid value.

            if (container->m_tetrahedronEdge.size()>0)
            {
                // removes the tetrahedronEdges from the tetrahedronEdgeArray
                container->m_tetrahedronEdge[ indices[i] ] = container->m_tetrahedronEdge[ container->m_tetrahedron.size() - 1 ]; // overwriting with last valid value.
                container->m_tetrahedronEdge.resize( container->m_tetrahedronEdge.size() - 1 ); // resizing to erase multiple occurence of the edge.
            }

            if (container->m_tetrahedronTriangle.size()>0)
            {
                // removes the tetrahedronTriangles from the tetrahedronTriangleArray
                container->m_tetrahedronTriangle[ indices[i] ] = container->m_tetrahedronTriangle[ container->m_tetrahedron.size() - 1 ]; // overwriting with last valid value.
                container->m_tetrahedronTriangle.resize( container->m_tetrahedronTriangle.size() - 1 ); // resizing to erase multiple occurence of the edge.
            }
            container->m_tetrahedron.resize( container->m_tetrahedron.size() - 1 ); // resizing to erase multiple occurence of the edge.


            // now updates the shell information of the edge formely at the end of the array
            // first check that the edge shell array has been initialized
            if ( indices[i] < container->m_tetrahedron.size() )
            {
                unsigned int oldTetrahedronIndex=container->m_tetrahedron.size();
                t = container->m_tetrahedron[ indices[i] ];
                if (container->m_tetrahedronVertexShell.size()>0)
                {

                    std::vector< unsigned int > &shell0 = container->m_tetrahedronVertexShell[ t[0] ];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell0.begin(), shell0.end(), oldTetrahedronIndex ) !=shell0.end());
                    std::vector< unsigned int >::iterator it=std::find( shell0.begin(), shell0.end(), oldTetrahedronIndex );
                    (*it)=indices[i];

                    std::vector< unsigned int > &shell1 = container->m_tetrahedronVertexShell[ t[1] ];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell1.begin(), shell1.end(), oldTetrahedronIndex ) !=shell1.end());
                    it=std::find( shell1.begin(), shell1.end(), oldTetrahedronIndex );
                    (*it)=indices[i];

                    std::vector< unsigned int > &shell2 = container->m_tetrahedronVertexShell[ t[2] ];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell2.begin(), shell2.end(), oldTetrahedronIndex ) !=shell2.end());
                    it=std::find( shell2.begin(), shell2.end(), oldTetrahedronIndex );
                    (*it)=indices[i];

                    std::vector< unsigned int > &shell3 = container->m_tetrahedronVertexShell[ t[3] ];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell3.begin(), shell3.end(), oldTetrahedronIndex ) !=shell3.end());
                    it=std::find( shell3.begin(), shell3.end(), oldTetrahedronIndex );
                    (*it)=indices[i];


                }
                if (container->m_tetrahedronEdgeShell.size()>0)
                {

                    std::vector< unsigned int > &shell0 =  container->m_tetrahedronEdgeShell[ container->m_tetrahedronEdge[indices[i]][0]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell0.begin(), shell0.end(), oldTetrahedronIndex) !=shell0.end());
                    std::vector< unsigned int >::iterator it=std::find( shell0.begin(), shell0.end(), oldTetrahedronIndex );
                    (*it)=indices[i];

                    std::vector< unsigned int > &shell1 =  container->m_tetrahedronEdgeShell[ container->m_tetrahedronEdge[indices[i]][1]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell1.begin(), shell1.end(), oldTetrahedronIndex ) !=shell1.end());
                    it=std::find( shell1.begin(), shell1.end(), oldTetrahedronIndex );
                    (*it)=indices[i];

                    std::vector< unsigned int > &shell2 =  container->m_tetrahedronEdgeShell[ container->m_tetrahedronEdge[indices[i]][2]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell2.begin(), shell2.end(), oldTetrahedronIndex ) !=shell2.end());
                    it=std::find( shell2.begin(), shell2.end(), oldTetrahedronIndex );
                    (*it)=indices[i];

                    std::vector< unsigned int > &shell3 =  container->m_tetrahedronEdgeShell[ container->m_tetrahedronEdge[indices[i]][3]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell3.begin(), shell3.end(), oldTetrahedronIndex ) !=shell3.end());
                    it=std::find( shell3.begin(), shell3.end(), oldTetrahedronIndex );
                    (*it)=indices[i];

                }
                if (container->m_tetrahedronTriangleShell.size()>0)
                {

                    std::vector< unsigned int > &shell0 =  container->m_tetrahedronTriangleShell[ container->m_tetrahedronTriangle[indices[i]][0]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell0.begin(), shell0.end(), oldTetrahedronIndex ) !=shell0.end());
                    std::vector< unsigned int >::iterator it=std::find( shell0.begin(), shell0.end(), oldTetrahedronIndex );
                    (*it)=indices[i];

                    std::vector< unsigned int > &shell1 =  container->m_tetrahedronTriangleShell[ container->m_tetrahedronTriangle[indices[i]][1]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell1.begin(), shell1.end(), oldTetrahedronIndex ) !=shell1.end());
                    it=std::find( shell1.begin(), shell1.end(), oldTetrahedronIndex );
                    (*it)=indices[i];

                    std::vector< unsigned int > &shell2 =  container->m_tetrahedronTriangleShell[ container->m_tetrahedronTriangle[indices[i]][2]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell2.begin(), shell2.end(), oldTetrahedronIndex ) !=shell2.end());
                    it=std::find( shell2.begin(), shell2.end(), oldTetrahedronIndex );
                    (*it)=indices[i];

                    std::vector< unsigned int > &shell3 =  container->m_tetrahedronTriangleShell[ container->m_tetrahedronTriangle[indices[i]][3]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell3.begin(), shell3.end(), oldTetrahedronIndex ) !=shell3.end());
                    it=std::find( shell3.begin(), shell3.end(), oldTetrahedronIndex );
                    (*it)=indices[i];

                }
            }
        }
    }
}



template<class DataTypes >
void TetrahedronSetTopologyModifier< DataTypes >::addPointsProcess(const unsigned int nPoints,
        const std::vector< std::vector< unsigned int > >& ancestors,
        const std::vector< std::vector< double > >& baryCoefs)
{
    // start by calling the standard method.
    EdgeSetTopologyModifier< DataTypes >::addPointsProcess( nPoints, ancestors, baryCoefs );

    // now update the local container structures.
    TetrahedronSetTopology<DataTypes> *topology = dynamic_cast<TetrahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast<TetrahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    container->m_tetrahedronVertexShell.resize( container->m_tetrahedronVertexShell.size() + nPoints );
}


template<class DataTypes >
void TetrahedronSetTopologyModifier< DataTypes >::addEdgesProcess(const std::vector< Edge > &edges)
{
    // start by calling the standard method.
    EdgeSetTopologyModifier< DataTypes >::addEdgesProcess( edges );

    // now update the local container structures.
    TetrahedronSetTopology<DataTypes> *topology = dynamic_cast<TetrahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast<TetrahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    container->m_tetrahedronEdgeShell.resize( container->m_tetrahedronEdgeShell.size() + edges.size() );
}

template<class DataTypes >
void TetrahedronSetTopologyModifier< DataTypes >::addTrianglesProcess(const std::vector< Triangle > &triangles)
{
    // start by calling the standard method.
    TriangleSetTopologyModifier< DataTypes >::addTrianglesProcess( triangles );

    // now update the local container structures.
    TetrahedronSetTopology<DataTypes> *topology = dynamic_cast<TetrahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast<TetrahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    container->m_tetrahedronTriangleShell.resize( container->m_tetrahedronTriangleShell.size() + triangles.size() );
}



template< class DataTypes >
void TetrahedronSetTopologyModifier< DataTypes >::removePointsProcess( std::vector<unsigned int> &indices)
{
    // start by calling the standard method.
    TriangleSetTopologyModifier< DataTypes >::removePointsProcess(  indices );

    // now update the local container structures
    TetrahedronSetTopology<DataTypes> *topology = dynamic_cast<TetrahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast<TetrahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    int vertexIndex;

    assert(container->m_tetrahedronVertexShell.size()>0);

    unsigned int lastPoint = container->m_tetrahedronVertexShell.size() - 1;

    for (unsigned int i = 0; i < indices.size(); ++i)
    {
        // updating the edges connected to the point replacing the removed one:
        // for all edges connected to the last point
        std::vector<unsigned int>::iterator itt=container->m_tetrahedronVertexShell[lastPoint].begin();
        for (; itt!=container->m_tetrahedronVertexShell[lastPoint].end(); ++itt)
        {

            vertexIndex=container->getVertexIndexInTetrahedron(container->m_tetrahedron[(*itt)],lastPoint);
            assert(vertexIndex!= -1);
            container->m_tetrahedron[(*itt)][(unsigned int)vertexIndex]=indices[i];
        }

        // updating the edge shell itself (change the old index for the new one)
        container->m_tetrahedronVertexShell[ indices[i] ] = container->m_tetrahedronVertexShell[ lastPoint ];

        --lastPoint;
    }

    container->m_tetrahedronVertexShell.resize( container->m_tetrahedronVertexShell.size() - indices.size() );
}

template< class DataTypes >
void TetrahedronSetTopologyModifier< DataTypes >::removeEdgesProcess( const std::vector<unsigned int> &indices,const bool removeIsolatedItems)
{
    // start by calling the standard method.
    TriangleSetTopologyModifier< DataTypes >::removeEdgesProcess(  indices );

    // now update the local container structures
    TetrahedronSetTopology<DataTypes> *topology = dynamic_cast<TetrahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast<TetrahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);


    if (container->m_tetrahedronEdgeShell.size()>0)
    {
        unsigned int lastEdge = container->m_tetrahedronEdgeShell.size() - 1;

        for (unsigned int i = 0; i < indices.size(); ++i)
        {
            // updating the edge shell itself (change the old index for the new one)
            container->m_tetrahedronEdgeShell[ indices[i] ] = container->m_tetrahedronEdgeShell[ lastEdge ];

            --lastEdge;
        }

        container->m_tetrahedronEdgeShell.resize( container->m_tetrahedronEdgeShell.size() - indices.size() );
    }
}

template< class DataTypes >
void TetrahedronSetTopologyModifier< DataTypes >::removeTrianglesProcess(  const std::vector<unsigned int> &indices,const bool removeIsolatedItems)
{
    // start by calling the standard method.
    TriangleSetTopologyModifier< DataTypes >::removeTrianglesProcess( indices );

    // now update the local container structures
    TetrahedronSetTopology<DataTypes> *topology = dynamic_cast<TetrahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast<TetrahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);


    if (container->m_tetrahedronTriangleShell.size()>0)
    {
        unsigned int lastTriangle = container->m_tetrahedronTriangleShell.size() - 1;

        for (unsigned int i = 0; i < indices.size(); ++i)
        {
            // updating the triangle shell itself (change the old index for the new one)
            container->m_tetrahedronTriangleShell[ indices[i] ] = container->m_tetrahedronTriangleShell[ lastTriangle ];

            --lastTriangle;
        }

        container->m_tetrahedronTriangleShell.resize( container->m_tetrahedronTriangleShell.size() - indices.size() );
    }
}


template< class DataTypes >
void TetrahedronSetTopologyModifier< DataTypes >::renumberPointsProcess( const std::vector<unsigned int> &index)
{
    // start by calling the standard method
    TriangleSetTopologyModifier< DataTypes >::renumberPointsProcess( index );

    // now update the local container structures.
    TetrahedronSetTopology<DataTypes> *topology = dynamic_cast<TetrahedronSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast<TetrahedronSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);

    std::vector< std::vector< unsigned int > > tetrahedronVertexShell_cp = container->m_tetrahedronVertexShell;
    for (unsigned int i = 0; i < index.size(); ++i)
    {
        container->m_tetrahedronVertexShell[i] = tetrahedronVertexShell_cp[ index[i] ];
    }

    for (unsigned int i = 0; i < container->m_tetrahedron.size(); ++i)
    {
        container->m_tetrahedron[i][0]  = index[ container->m_tetrahedron[i][0]  ];
        container->m_tetrahedron[i][1]  = index[ container->m_tetrahedron[i][1]  ];
        container->m_tetrahedron[i][2]  = index[ container->m_tetrahedron[i][2]  ];
    }


}

/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////TriangleSetGeometryAlgorithms//////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////


/// Cross product for 3-elements vectors.
template<typename real>
inline real tripleProduct(const Vec<3,real>& a, const Vec<3,real>& b,const Vec<3,real> &c)
{
    return dot(a,cross(b,c));
}

/// area from 2-elements vectors.
template <typename real>
inline real tripleProduct(const Vec<2,real>& a, const Vec<2,real>& b,const Vec<2,real> &c)
{
    assert(false);
    return (real)0;
}
/// area for 1-elements vectors.
template <typename real>
inline real tripleProduct(const Vec<1,real>& a, const Vec<1,real>& b,const Vec<1,real> &c)
{
    assert(false);
    return (real)0;
}

template< class DataTypes>
typename DataTypes::Real TetrahedronSetGeometryAlgorithms< DataTypes >::computeTetrahedronVolume( const unsigned int i) const
{
    TetrahedronSetTopology< DataTypes > *topology = dynamic_cast<TetrahedronSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast< TetrahedronSetTopologyContainer* >(topology->getTopologyContainer());
    const Tetrahedron &t=container->getTetrahedron(i);
    const VecCoord& p = *topology->getDOF()->getX();
    Real volume=(Real)(tripleProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]],p[t[3]]-p[t[0]])/6.0);
    return volume;
}
template< class DataTypes>
typename DataTypes::Real TetrahedronSetGeometryAlgorithms< DataTypes >::computeRestTetrahedronVolume( const unsigned int i) const
{
    TetrahedronSetTopology< DataTypes > *topology = dynamic_cast<TetrahedronSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast< TetrahedronSetTopologyContainer* >(topology->getTopologyContainer());
    const Tetrahedron &t=container->getTetrahedron(i);
    const VecCoord& p = *topology->getDOF()->getX0();
    Real volume=(Real)(tripleProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]],p[t[3]]-p[t[0]])/6.0);
    return volume;

}

/// computes the edge length of all edges are store in the array interface
template<class DataTypes>
void TetrahedronSetGeometryAlgorithms<DataTypes>::computeTetrahedronVolume( BasicArrayInterface<Real> &ai) const
{
    TetrahedronSetTopology< DataTypes > *topology = dynamic_cast<TetrahedronSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    TetrahedronSetTopologyContainer * container = static_cast< TetrahedronSetTopologyContainer* >(topology->getTopologyContainer());
    const std::vector<Tetrahedron> &ta=container->getTetrahedronArray();
    const typename DataTypes::VecCoord& p = *topology->getDOF()->getX();
    unsigned int i;
    for (i=0; i<ta.size(); ++i)
    {
        const Tetrahedron &t=ta[i];
        ai[i]=(Real)(tripleProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]],p[t[3]]-p[t[0]])/6.0);
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////TetrahedronSetTopology//////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
template<class DataTypes>
void TetrahedronSetTopology<DataTypes>::init()
{
}
template<class DataTypes>
TetrahedronSetTopology<DataTypes>::TetrahedronSetTopology(MechanicalObject<DataTypes> *obj) : PointSetTopology<DataTypes>( obj,(PointSetTopology<DataTypes> *)0)
{
    this->m_topologyContainer= new TetrahedronSetTopologyContainer(this);
    this->m_topologyModifier= new TetrahedronSetTopologyModifier<DataTypes>(this);
    this->m_topologyAlgorithms= new TetrahedronSetTopologyAlgorithms<DataTypes>(this);
    this->m_geometryAlgorithms= new TetrahedronSetGeometryAlgorithms<DataTypes>(this);
}


} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_TetrahedronSetTOPOLOGY_INL
