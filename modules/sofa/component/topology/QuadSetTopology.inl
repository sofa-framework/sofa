#ifndef SOFA_COMPONENT_TOPOLOGY_QUADSETTOPOLOGY_INL
#define SOFA_COMPONENT_TOPOLOGY_QUADSETTOPOLOGY_INL

#include <sofa/component/topology/QuadSetTopology.h>
#include <sofa/component/topology/PointSetTopology.inl>
#include <sofa/component/topology/TopologyChangedEvent.h>
#include <sofa/simulation/tree/PropagateEventVisitor.h>
#include <sofa/simulation/tree/GNode.h>
#include <algorithm>
#include <functional>


#include <sofa/defaulttype/Vec.h> // typing "Vec"

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::behavior;



/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////QuadSetTopologyModifier//////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////



template<class DataTypes>
class QuadSetTopologyLoader : public PointSetTopologyLoader<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;

    VecCoord pointArray;
    QuadSetTopologyModifier<DataTypes> *tstm;

    QuadSetTopologyLoader(QuadSetTopologyModifier<DataTypes> *tm) :PointSetTopologyLoader<DataTypes>(), tstm(tm)
    {
    }

    virtual void addQuad(int p1, int p2, int p3, int p4)
    {
        tstm->addQuad(Quad(helper::make_array<unsigned int>((unsigned int)p1,(unsigned int)p2,(unsigned int) p3,(unsigned int) p4)));
    }
};
template<class DataTypes>
void QuadSetTopologyModifier<DataTypes>::addQuad(Quad t)
{

    QuadSetTopology<DataTypes> *topology = dynamic_cast<QuadSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    QuadSetTopologyContainer * container = static_cast<QuadSetTopologyContainer *>( topology->getTopologyContainer() );
    assert (container != 0);
    container->m_quad.push_back(t);

}
template<class DataTypes>
bool QuadSetTopologyModifier<DataTypes>::load(const char *filename)
{

    QuadSetTopologyLoader<DataTypes> loader(this);
    if (!loader.load(filename))
        return false;
    else
    {
        loadPointSet(&loader);
        return true;
    }
}

template<class DataTypes>
void QuadSetTopologyModifier<DataTypes>::addQuadsProcess(const sofa::helper::vector< Quad > &quads)
{
    QuadSetTopology<DataTypes> *topology = dynamic_cast<QuadSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    QuadSetTopologyContainer * container = static_cast<QuadSetTopologyContainer *>( topology->getTopologyContainer() );
    assert (container != 0);

    if (container->m_quad.size()>0)
    {

        unsigned int quadIndex;
        unsigned int j;

        for (unsigned int i = 0; i < quads.size(); ++i)
        {
            const Quad &t = quads[i];
            // check if the 3 vertices are different
            assert(t[0]!=t[1]);
            assert(t[0]!=t[2]);
            assert(t[1]!=t[2]);
            assert(t[1]!=t[3]);
            // check if there already exists a quad with the same indices
            assert(container->getQuadIndex(t[0],t[1],t[2],t[3])== -1);
            container->m_quad.push_back(t);
            quadIndex=container->m_quad.size() - 1 ;

            const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tvsa=container->getQuadVertexShellArray();
            if (tvsa.size()>0)
            {

                container->getQuadVertexShellForModification( t[0] ).push_back( quadIndex );

                container->getQuadVertexShellForModification( t[1] ).push_back( quadIndex );

                container->getQuadVertexShellForModification( t[2] ).push_back( quadIndex );

                container->getQuadVertexShellForModification( t[3] ).push_back( quadIndex );

            }
            if (container->m_quadEdge.size()>0)
            {

                int edgeIndex;

                for (j=0; j<4; ++j)
                {

                    edgeIndex=container->getEdgeIndex(t[(j+1)%4],t[(j+2)%4]);

                    if(edgeIndex == -1)
                    {

                        // first create the edges
                        sofa::helper::vector< Edge > v;
                        Edge e1 (t[(j+1)%4], t[(j+2)%4]);
                        v.push_back(e1);

                        addEdgesProcess((const sofa::helper::vector< Edge > &) v);

                        edgeIndex=container->getEdgeIndex(t[(j+1)%4],t[(j+2)%4]);
                        sofa::helper::vector< unsigned int > edgeIndexList;
                        edgeIndexList.push_back(edgeIndex);
                        addEdgesWarning( v.size(), v,edgeIndexList);
                    }

                    container->m_quadEdge.resize(quadIndex+1);

                    container->m_quadEdge[quadIndex][j]= edgeIndex;
                }
            }

            const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tesa=container->getQuadEdgeShellArray();
            if (tesa.size()>0)
            {

                for (j=0; j<4; ++j)
                {
                    container->m_quadEdgeShell[container->m_quadEdge[quadIndex][j]].push_back( quadIndex );
                }

                sofa::helper::vector< Quad > current_quad;
                current_quad.push_back(t);
                sofa::helper::vector< unsigned int > quadsIndexList;
                quadsIndexList.push_back((unsigned int) quadIndex);
                addQuadsWarning((const unsigned int) 1, current_quad, quadsIndexList);
            }

        }
    }
}



template<class DataTypes>
void QuadSetTopologyModifier<DataTypes>::addQuadsWarning(const unsigned int nQuads, const sofa::helper::vector< Quad >& quadsList,
        const sofa::helper::vector< unsigned int >& quadsIndexList,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
{
    // Warning that quads just got created
    QuadsAdded *e=new QuadsAdded(nQuads, quadsList,quadsIndexList,ancestors,baryCoefs);
    this->addTopologyChange(e);
}




template<class DataTypes>
void QuadSetTopologyModifier<DataTypes>::removeQuadsWarning( sofa::helper::vector<unsigned int> &quads)
{
    /// sort vertices to remove in a descendent order
    std::sort( quads.begin(), quads.end(), std::greater<unsigned int>() );

    // Warning that these quads will be deleted
    QuadsRemoved *e=new QuadsRemoved(quads);
    this->addTopologyChange(e);

}



template<class DataTypes>
void QuadSetTopologyModifier<DataTypes>::removeQuadsProcess(const sofa::helper::vector<unsigned int> &indices,const bool removeIsolatedItems)
{
    QuadSetTopology<DataTypes> *topology = dynamic_cast<QuadSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    QuadSetTopologyContainer * container = static_cast<QuadSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);


    /// only remove isolated edges if the structures exists since removeEdges
    /// will remove isolated vertices
    if (removeIsolatedItems)
    {
        /// force the creation of the Quad Edge Shell array to detect isolated edges
        if (container->m_quadEdge.size()>0)
            container->getQuadEdgeShellArray();
        /// force the creation of the Quad Shell array to detect isolated vertices
        container->getQuadVertexShellArray();
    }


    if (container->m_quad.size()>0)
    {
        sofa::helper::vector<unsigned int> edgeToBeRemoved;
        sofa::helper::vector<unsigned int> vertexToBeRemoved;



        for (unsigned int i = 0; i < indices.size(); ++i)
        {
            Quad &t = container->m_quad[ indices[i] ];
            // first check that the quad vertex shell array has been initialized
            if (container->m_quadVertexShell.size()>0)
            {

                sofa::helper::vector< unsigned int > &shell0 = container->m_quadVertexShell[ t[0] ];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell0.begin(), shell0.end(), indices[i] ) !=shell0.end());
                shell0.erase( std::find( shell0.begin(), shell0.end(), indices[i] ) );
                if ((removeIsolatedItems) && (shell0.size()==0))
                {
                    vertexToBeRemoved.push_back(t[0]);
                }


                sofa::helper::vector< unsigned int > &shell1 = container->m_quadVertexShell[ t[1] ];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell1.begin(), shell1.end(), indices[i] ) !=shell1.end());
                shell1.erase( std::find( shell1.begin(), shell1.end(), indices[i] ) );
                if ((removeIsolatedItems) && (shell1.size()==0))
                {
                    vertexToBeRemoved.push_back(t[1]);
                }


                sofa::helper::vector< unsigned int > &shell2 = container->m_quadVertexShell[ t[2] ];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell2.begin(), shell2.end(), indices[i] ) !=shell2.end());
                shell2.erase( std::find( shell2.begin(), shell2.end(), indices[i] ) );
                if ((removeIsolatedItems) && (shell2.size()==0))
                {
                    vertexToBeRemoved.push_back(t[2]);
                }

                sofa::helper::vector< unsigned int > &shell3 = container->m_quadVertexShell[ t[3] ];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell3.begin(), shell3.end(), indices[i] ) !=shell3.end());
                shell3.erase( std::find( shell3.begin(), shell3.end(), indices[i] ) );
                if ((removeIsolatedItems) && (shell3.size()==0))
                {
                    vertexToBeRemoved.push_back(t[3]);
                }

            }

            /** first check that the quad edge shell array has been initialized */
            if (container->m_quadEdgeShell.size()>0)
            {
                sofa::helper::vector< unsigned int > &shell0 = container->m_quadEdgeShell[ container->m_quadEdge[indices[i]][0]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell0.begin(), shell0.end(), indices[i] ) !=shell0.end());
                shell0.erase( std::find( shell0.begin(), shell0.end(), indices[i] ) );
                if ((removeIsolatedItems) && (shell0.size()==0))
                    edgeToBeRemoved.push_back(container->m_quadEdge[indices[i]][0]);

                sofa::helper::vector< unsigned int > &shell1 = container->m_quadEdgeShell[ container->m_quadEdge[indices[i]][1]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell1.begin(), shell1.end(), indices[i] ) !=shell1.end());
                shell1.erase( std::find( shell1.begin(), shell1.end(), indices[i] ) );
                if ((removeIsolatedItems) && (shell1.size()==0))
                    edgeToBeRemoved.push_back(container->m_quadEdge[indices[i]][1]);


                sofa::helper::vector< unsigned int > &shell2 = container->m_quadEdgeShell[ container->m_quadEdge[indices[i]][2]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell2.begin(), shell2.end(), indices[i] ) !=shell2.end());
                shell2.erase( std::find( shell2.begin(), shell2.end(), indices[i] ) );
                if ((removeIsolatedItems) && (shell2.size()==0))
                    edgeToBeRemoved.push_back(container->m_quadEdge[indices[i]][2]);


                sofa::helper::vector< unsigned int > &shell3 = container->m_quadEdgeShell[ container->m_quadEdge[indices[i]][3]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell3.begin(), shell3.end(), indices[i] ) !=shell3.end());
                shell3.erase( std::find( shell3.begin(), shell3.end(), indices[i] ) );
                if ((removeIsolatedItems) && (shell3.size()==0))
                    edgeToBeRemoved.push_back(container->m_quadEdge[indices[i]][3]);

            }


            // removes the quad from the quadArray
            container->m_quad[ indices[i] ] = container->m_quad[ container->m_quad.size() - 1 ]; // overwriting with last valid value.

            if (container->m_quadEdge.size()>0)
            {
                // removes the quadEdges from the quadEdgesArray
                container->m_quadEdge[ indices[i] ] = container->m_quadEdge[ container->m_quad.size() - 1 ]; // overwriting with last valid value.
                container->m_quadEdge.resize( container->m_quadEdge.size() - 1 ); // resizing to erase multiple occurence of the edge.
            }
            container->m_quad.resize( container->m_quad.size() - 1 ); // resizing to erase multiple occurence of the edge.

            // now updates the shell information of the edge formely at the end of the array
            // first check that the edge shell array has been initialized
            if ( indices[i] < container->m_quad.size() )
            {
                unsigned int oldQuadIndex=container->m_quad.size();
                t = container->m_quad[ indices[i] ];
                if (container->m_quadVertexShell.size()>0)
                {

                    sofa::helper::vector< unsigned int > &shell0 = container->m_quadVertexShell[ t[0] ];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell0.begin(), shell0.end(), oldQuadIndex ) !=shell0.end());
                    sofa::helper::vector< unsigned int >::iterator it=std::find( shell0.begin(), shell0.end(), oldQuadIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell1 = container->m_quadVertexShell[ t[1] ];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell1.begin(), shell1.end(), oldQuadIndex ) !=shell1.end());
                    it=std::find( shell1.begin(), shell1.end(), oldQuadIndex );
                    (*it)=indices[i];


                    sofa::helper::vector< unsigned int > &shell2 = container->m_quadVertexShell[ t[2] ];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell2.begin(), shell2.end(), oldQuadIndex ) !=shell2.end());
                    it=std::find( shell2.begin(), shell2.end(), oldQuadIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell3 = container->m_quadVertexShell[ t[3] ];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell3.begin(), shell3.end(), oldQuadIndex ) !=shell3.end());
                    it=std::find( shell3.begin(), shell3.end(), oldQuadIndex );
                    (*it)=indices[i];


                }
                if (container->m_quadEdgeShell.size()>0)
                {

                    sofa::helper::vector< unsigned int > &shell0 =  container->m_quadEdgeShell[ container->m_quadEdge[indices[i]][0]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell0.begin(), shell0.end(), oldQuadIndex ) !=shell0.end());
                    sofa::helper::vector< unsigned int >::iterator it=std::find( shell0.begin(), shell0.end(), oldQuadIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell1 =  container->m_quadEdgeShell[ container->m_quadEdge[indices[i]][1]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell1.begin(), shell1.end(), oldQuadIndex ) !=shell1.end());
                    it=std::find( shell1.begin(), shell1.end(), oldQuadIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell2 =  container->m_quadEdgeShell[ container->m_quadEdge[indices[i]][2]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell2.begin(), shell2.end(), oldQuadIndex) !=shell2.end());
                    it=std::find( shell2.begin(), shell2.end(), oldQuadIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell3 =  container->m_quadEdgeShell[ container->m_quadEdge[indices[i]][3]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell3.begin(), shell3.end(), oldQuadIndex) !=shell3.end());
                    it=std::find( shell3.begin(), shell3.end(), oldQuadIndex );
                    (*it)=indices[i];

                }

            }
        }
        if ( (edgeToBeRemoved.size()>0) || (vertexToBeRemoved.size()>0))
        {

            if (edgeToBeRemoved.size()>0)
                /// warn that edges will be deleted
                this->removeEdgesWarning(edgeToBeRemoved);
            if (vertexToBeRemoved.size()>0)
                this->removePointsWarning(vertexToBeRemoved);
            /// propagate to all components
            topology->propagateTopologicalChanges();
            if (edgeToBeRemoved.size()>0)
                /// actually remove edges without looking for isolated vertices
                this->removeEdgesProcess(edgeToBeRemoved,false);

            if (vertexToBeRemoved.size()>0)
            {
                this->removePointsProcess(vertexToBeRemoved);
            }
        }
    }
}



template<class DataTypes >
void QuadSetTopologyModifier< DataTypes >::addPointsProcess(const unsigned int nPoints,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
{
    // start by calling the standard method.
    EdgeSetTopologyModifier< DataTypes >::addPointsProcess( nPoints, ancestors, baryCoefs );

    // now update the local container structures.
    QuadSetTopology<DataTypes> *topology = dynamic_cast<QuadSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    QuadSetTopologyContainer * container = static_cast<QuadSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    container->m_quadVertexShell.resize( container->m_quadVertexShell.size() + nPoints );
}


template<class DataTypes >
void QuadSetTopologyModifier< DataTypes >::addEdgesProcess(const sofa::helper::vector< Edge > &edges)
{
    // start by calling the standard method.
    EdgeSetTopologyModifier< DataTypes >::addEdgesProcess( edges );

    // now update the local container structures.
    QuadSetTopology<DataTypes> *topology = dynamic_cast<QuadSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    QuadSetTopologyContainer * container = static_cast<QuadSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    container->m_quadEdgeShell.resize( container->m_quadEdgeShell.size() + edges.size() );
}



template< class DataTypes >
void QuadSetTopologyModifier< DataTypes >::removePointsProcess( sofa::helper::vector<unsigned int> &indices, const bool removeDOF)
{
    // Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)

    // now update the local container structures
    QuadSetTopology<DataTypes> *topology = dynamic_cast<QuadSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    QuadSetTopologyContainer * container = static_cast<QuadSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);

    // force the creation of the quad vertex shell array before any point is deleted
    container->getQuadVertexShellArray();

    // start by calling the standard method.
    EdgeSetTopologyModifier< DataTypes >::removePointsProcess( indices, removeDOF );

    int vertexIndex;

    unsigned int lastPoint = container->m_quadVertexShell.size() - 1;

    for (unsigned int i = 0; i < indices.size(); ++i)
    {
        // updating the quads connected to the point replacing the removed one:
        // for all quads connected to the last point
        sofa::helper::vector<unsigned int>::iterator itt=container->m_quadVertexShell[lastPoint].begin();
        for (; itt!=container->m_quadVertexShell[lastPoint].end(); ++itt)
        {

            vertexIndex=container->getVertexIndexInQuad(container->m_quad[(*itt)],lastPoint);
            assert(vertexIndex!= -1);
            container->m_quad[(*itt)][(unsigned int)vertexIndex]=indices[i];
        }

        // updating the edge shell itself (change the old index for the new one)
        container->m_quadVertexShell[ indices[i] ] = container->m_quadVertexShell[ lastPoint ];

        --lastPoint;
    }

    container->m_quadVertexShell.resize( container->m_quadVertexShell.size() - indices.size() );
}

template< class DataTypes >
void QuadSetTopologyModifier< DataTypes >::removeEdgesProcess( const sofa::helper::vector<unsigned int> &indices,const bool removeIsolatedItems)
{
    // now update the local container structures
    QuadSetTopology<DataTypes> *topology = dynamic_cast<QuadSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    QuadSetTopologyContainer * container = static_cast<QuadSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);

    if (container->m_quadEdge.size()>0)
        container->getQuadEdgeShellArray();

    // start by calling the standard method.
    EdgeSetTopologyModifier< DataTypes >::removeEdgesProcess(indices,removeIsolatedItems);

    if (container->m_quadEdge.size()>0)
    {
        unsigned int edgeIndex;
        unsigned int lastEdge = container->m_quadEdgeShell.size() - 1;

        for (unsigned int i = 0; i < indices.size(); ++i)
        {
            // updating the quads connected to the edge replacing the removed one:
            // for all quads connected to the last point
            sofa::helper::vector<unsigned int>::iterator itt=container->m_quadEdgeShell[lastEdge].begin();
            for (; itt!=container->m_quadEdgeShell[lastEdge].end(); ++itt)
            {

                edgeIndex=container->getEdgeIndexInQuad(container->m_quadEdge[(*itt)],lastEdge);
                assert((int)edgeIndex!= -1);
                container->m_quadEdge[(*itt)][(unsigned int)edgeIndex]=indices[i];
            }

            // updating the edge shell itself (change the old index for the new one)
            container->m_quadEdgeShell[ indices[i] ] = container->m_quadEdgeShell[ lastEdge ];

            --lastEdge;
        }

        container->m_quadEdgeShell.resize( container->m_quadEdgeShell.size() - indices.size() );
    }
}



template< class DataTypes >
void QuadSetTopologyModifier< DataTypes >::renumberPointsProcess( const sofa::helper::vector<unsigned int> &index)
{
    // start by calling the standard method
    EdgeSetTopologyModifier< DataTypes >::renumberPointsProcess( index );

    // now update the local container structures.
    QuadSetTopology<DataTypes> *topology = dynamic_cast<QuadSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    QuadSetTopologyContainer * container = static_cast<QuadSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);

    sofa::helper::vector< sofa::helper::vector< unsigned int > > quadVertexShell_cp = container->m_quadVertexShell;
    for (unsigned int i = 0; i < index.size(); ++i)
    {
        container->m_quadVertexShell[i] = quadVertexShell_cp[ index[i] ];
    }

    for (unsigned int i = 0; i < container->m_quad.size(); ++i)
    {
        container->m_quad[i][0]  = index[ container->m_quad[i][0]  ];
        container->m_quad[i][1]  = index[ container->m_quad[i][1]  ];
        container->m_quad[i][2]  = index[ container->m_quad[i][2]  ];
        container->m_quad[i][3]  = index[ container->m_quad[i][3]  ];
    }


}
/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////QuadSetTopologyAlgorithms//////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
template<class DataTypes>
void QuadSetTopologyAlgorithms< DataTypes >::removeQuads(sofa::helper::vector< unsigned int >& quads)
{
    QuadSetTopology< DataTypes > *topology = dynamic_cast<QuadSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    QuadSetTopologyModifier< DataTypes >* modifier  = static_cast< QuadSetTopologyModifier< DataTypes >* >(topology->getTopologyModifier());
    assert(modifier != 0);
    /// add the topological changes in the queue
    modifier->removeQuadsWarning(quads);
    // inform other objects that the quads are going to be removed
    topology->propagateTopologicalChanges();
    // now destroy the old quads.

    modifier->removeQuadsProcess(  quads ,true);

    assert(topology->getQuadSetTopologyContainer()->checkTopology());
}



/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////QuadSetGeometryAlgorithms//////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Cross product for 3-elements vectors.
template< class Real>
Real areaProduct(const Vec<3,Real>& a, const Vec<3,Real>& b)
{
    return Vec<3,Real>(a.y()*b.z() - a.z()*b.y(),
            a.z()*b.x() - a.x()*b.z(),
            a.x()*b.y() - a.y()*b.x()).norm();
}

/// area from 2-elements vectors.
template< class Real>
Real areaProduct(const defaulttype::Vec<2,Real>& a, const defaulttype::Vec<2,Real>& b )
{
    return a[0]*b[1] - a[1]*b[0];
}
/// area for 1-elements vectors.
template< class Real>
Real areaProduct(const defaulttype::Vec<1,Real>& , const defaulttype::Vec<1,Real>&  )
{
    assert(false);
    return (Real)0;
}

template< class DataTypes>
typename DataTypes::Real QuadSetGeometryAlgorithms< DataTypes >::computeQuadArea( const unsigned int i) const
{
    QuadSetTopology< DataTypes > *topology = dynamic_cast<QuadSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    QuadSetTopologyContainer * container = static_cast< QuadSetTopologyContainer* >(topology->getTopologyContainer());
    const Quad &t=container->getQuad(i);
    const VecCoord& p = *topology->getDOF()->getX();
    Real area=(Real)( (areaProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]]) + areaProduct(p[t[3]]-p[t[2]],p[t[0]]-p[t[2]]) )/2.0);
    return area;
}
template< class DataTypes>
typename DataTypes::Real QuadSetGeometryAlgorithms< DataTypes >::computeRestQuadArea( const unsigned int i) const
{
    QuadSetTopology< DataTypes > *topology = dynamic_cast<QuadSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    QuadSetTopologyContainer * container = static_cast< QuadSetTopologyContainer* >(topology->getTopologyContainer());
    const Quad &t=container->getQuad(i);
    const VecCoord& p = *topology->getDOF()->getX0();
    Real area=(Real) ( (areaProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]]) + areaProduct(p[t[3]]-p[t[2]],p[t[0]]-p[t[2]]) )/2.0);
    return area;
}

/// computes the edge length of all edges are store in the array interface
template<class DataTypes>
void QuadSetGeometryAlgorithms<DataTypes>::computeQuadArea( BasicArrayInterface<Real> &ai) const
{
    QuadSetTopology< DataTypes > *topology = dynamic_cast<QuadSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    QuadSetTopologyContainer * container = static_cast< QuadSetTopologyContainer* >(topology->getTopologyContainer());
    //const sofa::helper::vector<Quad> &ta=container->getQuadArray();
    unsigned int nb_quads = container->getNumberOfQuads();
    const typename DataTypes::VecCoord& p = *topology->getDOF()->getX();
    unsigned int i;
    for (i=0; i<nb_quads; ++i) // ta.size()
    {
        const Quad &t=container->getQuad(i);  //ta[i];
        ai[i]=(Real)( (areaProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]]) + areaProduct(p[t[3]]-p[t[2]],p[t[0]]-p[t[2]]) )/2.0);
    }
}



// Computes the normal vector of a quad indexed by ind_q (not normed)
template<class DataTypes>
Vec<3,double> QuadSetGeometryAlgorithms< DataTypes >::computeQuadNormal(const unsigned int ind_q)
{

    // HYP :  The quad indexed by ind_q is planar

    QuadSetTopology< DataTypes > *topology = dynamic_cast<QuadSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    QuadSetTopologyContainer * container = static_cast< QuadSetTopologyContainer* >(topology->getTopologyContainer());
    const Quad &q=container->getQuad(ind_q);
    const typename DataTypes::VecCoord& vect_c = *topology->getDOF()->getX();

    const typename DataTypes::Coord& c0=vect_c[q[0]];
    const typename DataTypes::Coord& c1=vect_c[q[1]];
    const typename DataTypes::Coord& c2=vect_c[q[2]];
    //const typename DataTypes::Coord& c3=vect_c[q[3]];

    Vec<3,Real> p0;
    p0[0] = (Real) (c0[0]); p0[1] = (Real) (c0[1]); p0[2] = (Real) (c0[2]);
    Vec<3,Real> p1;
    p1[0] = (Real) (c1[0]); p1[1] = (Real) (c1[1]); p1[2] = (Real) (c1[2]);
    Vec<3,Real> p2;
    p2[0] = (Real) (c2[0]); p2[1] = (Real) (c2[1]); p2[2] = (Real) (c2[2]);
    //Vec<3,Real> p3;
    //p3[0] = (Real) (c3[0]); p3[1] = (Real) (c3[1]); p3[2] = (Real) (c3[2]);

    Vec<3,Real> normal_q=(p1-p0).cross( p2-p0);

    return ((Vec<3,double>) normal_q);

}


// Test if a quad indexed by ind_quad (and incident to the vertex indexed by ind_p) is included or not in the plane defined by (ind_p, plane_vect)
template<class DataTypes>
bool QuadSetGeometryAlgorithms< DataTypes >::is_quad_in_plane(const unsigned int ind_q, const unsigned int ind_p,  const Vec<3,Real>&plane_vect)
{


    QuadSetTopology< DataTypes > *topology = dynamic_cast<QuadSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    QuadSetTopologyContainer * container = static_cast< QuadSetTopologyContainer* >(topology->getTopologyContainer());
    const Quad &q=container->getQuad(ind_q);

    // HYP : ind_p==q[0] or ind_q==t[1] or ind_q==t[2] or ind_q==q[3]

    const typename DataTypes::VecCoord& vect_c = *topology->getDOF()->getX();

    unsigned int ind_1;
    unsigned int ind_2;
    unsigned int ind_3;

    if(ind_p==q[0])
    {
        ind_1=q[1];
        ind_2=q[2];
        ind_3=q[3];
    }
    else
    {
        if(ind_p==q[1])
        {
            ind_1=q[2];
            ind_2=q[3];
            ind_3=q[0];
        }
        else
        {
            if(ind_p==q[2])
            {
                ind_1=q[3];
                ind_2=q[0];
                ind_3=q[1];
            }
            else   // ind_p==q[3]
            {
                ind_1=q[0];
                ind_2=q[1];
                ind_3=q[2];
            }
        }
    }

    const typename DataTypes::Coord& c0=vect_c[ind_p];
    const typename DataTypes::Coord& c1=vect_c[ind_1];
    const typename DataTypes::Coord& c2=vect_c[ind_2];
    const typename DataTypes::Coord& c3=vect_c[ind_3];

    Vec<3,Real> p0;
    p0[0] = (Real) (c0[0]); p0[1] = (Real) (c0[1]); p0[2] = (Real) (c0[2]);
    Vec<3,Real> p1;
    p1[0] = (Real) (c1[0]); p1[1] = (Real) (c1[1]); p1[2] = (Real) (c1[2]);
    Vec<3,Real> p2;
    p2[0] = (Real) (c2[0]); p2[1] = (Real) (c2[1]); p2[2] = (Real) (c2[2]);
    Vec<3,Real> p3;
    p3[0] = (Real) (c3[0]); p3[1] = (Real) (c3[1]); p3[2] = (Real) (c3[2]);

    return((p1-p0)*( plane_vect)>=0.0 && (p2-p0)*( plane_vect)>=0.0 && (p3-p0)*( plane_vect)>=0.0);


}


/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////QuadSetTopology//////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
template<class DataTypes>
void QuadSetTopology<DataTypes>::init()
{
}
template<class DataTypes>
QuadSetTopology<DataTypes>::QuadSetTopology(MechanicalObject<DataTypes> *obj) : EdgeSetTopology<DataTypes>( obj)
{
    this->m_topologyContainer= new QuadSetTopologyContainer(this);
    this->m_topologyModifier= new QuadSetTopologyModifier<DataTypes>(this);
    this->m_topologyAlgorithms= new QuadSetTopologyAlgorithms<DataTypes>(this);
    this->m_geometryAlgorithms= new QuadSetGeometryAlgorithms<DataTypes>(this);
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_QuadSetTOPOLOGY_INL
