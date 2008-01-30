#ifndef SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGY_INL
#define SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGY_INL

#include <sofa/component/topology/TriangleSetTopology.h>
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
////////////////////////////////////TriangleSetTopologyModifier//////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////



template<class DataTypes>
class TriangleSetTopologyLoader : public PointSetTopologyLoader<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;

    VecCoord pointArray;
    TriangleSetTopologyModifier<DataTypes> *tstm;

    TriangleSetTopologyLoader(TriangleSetTopologyModifier<DataTypes> *tm) :PointSetTopologyLoader<DataTypes>(), tstm(tm)
    {
    }

    virtual void addTriangle(int p1, int p2, int p3)
    {
        tstm->addTriangle(Triangle(helper::make_array<unsigned int>((unsigned int)p1,(unsigned int)p2,(unsigned int) p3)));
    }
};
template<class DataTypes>
void TriangleSetTopologyModifier<DataTypes>::addTriangle(Triangle t)
{

    TriangleSetTopology<DataTypes> *topology = dynamic_cast<TriangleSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast<TriangleSetTopologyContainer *>( topology->getTopologyContainer() );
    assert (container != 0);
    container->m_triangle.push_back(t);

}
template<class DataTypes>
bool TriangleSetTopologyModifier<DataTypes>::load(const char *filename)
{

    TriangleSetTopologyLoader<DataTypes> loader(this);
    if (!loader.load(filename))
        return false;
    else
    {
        loadPointSet(&loader);
        return true;
    }
}

template<class DataTypes>
void TriangleSetTopologyModifier<DataTypes>::addTrianglesProcess(const sofa::helper::vector< Triangle > &triangles)
{
    TriangleSetTopology<DataTypes> *topology = dynamic_cast<TriangleSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast<TriangleSetTopologyContainer *>( topology->getTopologyContainer() );
    assert (container != 0);

    if (container->m_triangle.size()>0)
    {

        unsigned int triangleIndex;
        unsigned int j;

        for (unsigned int i = 0; i < triangles.size(); ++i)
        {
            const Triangle &t = triangles[i];
            // check if the 3 vertices are different
            assert(t[0]!=t[1]);
            assert(t[0]!=t[2]);
            assert(t[1]!=t[2]);
            // check if there already exists a triangle with the same indices
            assert(container->getTriangleIndex(t[0],t[1],t[2])== -1);
            container->m_triangle.push_back(t);
            triangleIndex=container->m_triangle.size() - 1 ;

            const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tvsa=container->getTriangleVertexShellArray();
            if (tvsa.size()>0)
            {

                container->getTriangleVertexShellForModification( t[0] ).push_back( triangleIndex );

                container->getTriangleVertexShellForModification( t[1] ).push_back( triangleIndex );

                container->getTriangleVertexShellForModification( t[2] ).push_back( triangleIndex );

            }
            if (container->m_triangleEdge.size()>0)
            {

                int edgeIndex;

                for (j=0; j<3; ++j)
                {

                    edgeIndex=container->getEdgeIndex(t[(j+1)%3],t[(j+2)%3]);

                    if(edgeIndex == -1)
                    {

                        // first create the edges
                        sofa::helper::vector< Edge > v;
                        Edge e1 (t[(j+1)%3], t[(j+2)%3]);
                        v.push_back(e1);

                        addEdgesProcess((const sofa::helper::vector< Edge > &) v);

                        edgeIndex=container->getEdgeIndex(t[(j+1)%3],t[(j+2)%3]);
                        sofa::helper::vector< unsigned int > edgeIndexList;
                        edgeIndexList.push_back(edgeIndex);
                        addEdgesWarning( v.size(), v,edgeIndexList);
                    }

                    container->m_triangleEdge.resize(triangleIndex+1);

                    container->m_triangleEdge[triangleIndex][j]= edgeIndex;
                }
            }

            const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tesa=container->getTriangleEdgeShellArray();
            if (tesa.size()>0)
            {

                for (j=0; j<3; ++j)
                {
                    container->m_triangleEdgeShell[container->m_triangleEdge[triangleIndex][j]].push_back( triangleIndex );
                }

                sofa::helper::vector< Triangle > current_triangle;
                current_triangle.push_back(t);
                sofa::helper::vector< unsigned int > trianglesIndexList;
                trianglesIndexList.push_back((unsigned int) triangleIndex);
                addTrianglesWarning((const unsigned int) 1, current_triangle, trianglesIndexList);
            }

        }
    }
}



template<class DataTypes>
void TriangleSetTopologyModifier<DataTypes>::addTrianglesWarning(const unsigned int nTriangles, const sofa::helper::vector< Triangle >& trianglesList,
        const sofa::helper::vector< unsigned int >& trianglesIndexList,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
{
    // Warning that triangles just got created
    TrianglesAdded *e=new TrianglesAdded(nTriangles, trianglesList,trianglesIndexList,ancestors,baryCoefs);
    this->addTopologyChange(e);
}




template<class DataTypes>
void TriangleSetTopologyModifier<DataTypes>::removeTrianglesWarning( sofa::helper::vector<unsigned int> &triangles)
{
    /// sort vertices to remove in a descendent order
    std::sort( triangles.begin(), triangles.end(), std::greater<unsigned int>() );

    // Warning that these triangles will be deleted
    TrianglesRemoved *e=new TrianglesRemoved(triangles);
    this->addTopologyChange(e);

}



template<class DataTypes>
void TriangleSetTopologyModifier<DataTypes>::removeTrianglesProcess(const sofa::helper::vector<unsigned int> &indices,const bool removeIsolatedEdges, const bool removeIsolatedPoints)
{
    TriangleSetTopology<DataTypes> *topology = dynamic_cast<TriangleSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast<TriangleSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);


    /// only remove isolated edges if the structures exists since removeEdges
    /// will remove isolated vertices
    //if (removeIsolatedItems)
    //{
    /// force the creation of the Triangle Edge Shell array to detect isolated edges
    //if (container->m_triangleEdge.size()>0)
    container->getTriangleEdgeShellArray();
    /// force the creation of the Triangle Shell array to detect isolated vertices
    container->getTriangleVertexShellArray();
    //}


    if (container->m_triangle.size()>0)
    {
        sofa::helper::vector<unsigned int> edgeToBeRemoved;
        sofa::helper::vector<unsigned int> vertexToBeRemoved;



        for (unsigned int i = 0; i < indices.size(); ++i)
        {
            Triangle &t = container->m_triangle[ indices[i] ];
            // first check that the triangle vertex shell array has been initialized
            if (container->m_triangleVertexShell.size()>0)
            {

                sofa::helper::vector< unsigned int > &shell0 = container->m_triangleVertexShell[ t[0] ];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell0.begin(), shell0.end(), indices[i] ) !=shell0.end());
                shell0.erase( std::find( shell0.begin(), shell0.end(), indices[i] ) );
                if ((removeIsolatedPoints) && (shell0.size()==0))
                {
                    vertexToBeRemoved.push_back(t[0]);
                }


                sofa::helper::vector< unsigned int > &shell1 = container->m_triangleVertexShell[ t[1] ];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell1.begin(), shell1.end(), indices[i] ) !=shell1.end());
                shell1.erase( std::find( shell1.begin(), shell1.end(), indices[i] ) );
                if ((removeIsolatedPoints) && (shell1.size()==0))
                {
                    vertexToBeRemoved.push_back(t[1]);
                }


                sofa::helper::vector< unsigned int > &shell2 = container->m_triangleVertexShell[ t[2] ];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell2.begin(), shell2.end(), indices[i] ) !=shell2.end());
                shell2.erase( std::find( shell2.begin(), shell2.end(), indices[i] ) );
                if ((removeIsolatedPoints) && (shell2.size()==0))
                {
                    vertexToBeRemoved.push_back(t[2]);
                }

            }

            /** first check that the triangle edge shell array has been initialized */
            if (container->m_triangleEdgeShell.size()>0)
            {
                sofa::helper::vector< unsigned int > &shell0 = container->m_triangleEdgeShell[ container->m_triangleEdge[indices[i]][0]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell0.begin(), shell0.end(), indices[i] ) !=shell0.end());
                shell0.erase( std::find( shell0.begin(), shell0.end(), indices[i] ) );
                if ((removeIsolatedEdges) && (shell0.size()==0))
                    edgeToBeRemoved.push_back(container->m_triangleEdge[indices[i]][0]);

                sofa::helper::vector< unsigned int > &shell1 = container->m_triangleEdgeShell[ container->m_triangleEdge[indices[i]][1]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell1.begin(), shell1.end(), indices[i] ) !=shell1.end());
                shell1.erase( std::find( shell1.begin(), shell1.end(), indices[i] ) );
                if ((removeIsolatedEdges) && (shell1.size()==0))
                    edgeToBeRemoved.push_back(container->m_triangleEdge[indices[i]][1]);


                sofa::helper::vector< unsigned int > &shell2 = container->m_triangleEdgeShell[ container->m_triangleEdge[indices[i]][2]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell2.begin(), shell2.end(), indices[i] ) !=shell2.end());
                shell2.erase( std::find( shell2.begin(), shell2.end(), indices[i] ) );
                if ((removeIsolatedEdges) && (shell2.size()==0))
                    edgeToBeRemoved.push_back(container->m_triangleEdge[indices[i]][2]);

            }


            // removes the triangle from the triangleArray
            container->m_triangle[ indices[i] ] = container->m_triangle[ container->m_triangle.size() - 1 ]; // overwriting with last valid value.

            if (container->m_triangleEdge.size()>0)
            {
                // removes the triangleEdges from the triangleEdgesArray
                container->m_triangleEdge[ indices[i] ] = container->m_triangleEdge[ container->m_triangle.size() - 1 ]; // overwriting with last valid value.
                container->m_triangleEdge.resize( container->m_triangleEdge.size() - 1 ); // resizing to erase multiple occurence of the edge.
            }
            container->m_triangle.resize( container->m_triangle.size() - 1 ); // resizing to erase multiple occurence of the edge.

            // now updates the shell information of the edge formely at the end of the array
            // first check that the edge shell array has been initialized
            if ( indices[i] < container->m_triangle.size() )
            {
                unsigned int oldTriangleIndex=container->m_triangle.size();
                t = container->m_triangle[ indices[i] ];
                if (container->m_triangleVertexShell.size()>0)
                {

                    sofa::helper::vector< unsigned int > &shell0 = container->m_triangleVertexShell[ t[0] ];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell0.begin(), shell0.end(), oldTriangleIndex ) !=shell0.end());
                    sofa::helper::vector< unsigned int >::iterator it=std::find( shell0.begin(), shell0.end(), oldTriangleIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell1 = container->m_triangleVertexShell[ t[1] ];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell1.begin(), shell1.end(), oldTriangleIndex ) !=shell1.end());
                    it=std::find( shell1.begin(), shell1.end(), oldTriangleIndex );
                    (*it)=indices[i];


                    sofa::helper::vector< unsigned int > &shell2 = container->m_triangleVertexShell[ t[2] ];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell2.begin(), shell2.end(), oldTriangleIndex ) !=shell2.end());
                    it=std::find( shell2.begin(), shell2.end(), oldTriangleIndex );
                    (*it)=indices[i];


                }
                if (container->m_triangleEdgeShell.size()>0)
                {

                    sofa::helper::vector< unsigned int > &shell0 =  container->m_triangleEdgeShell[ container->m_triangleEdge[indices[i]][0]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell0.begin(), shell0.end(), oldTriangleIndex ) !=shell0.end());
                    sofa::helper::vector< unsigned int >::iterator it=std::find( shell0.begin(), shell0.end(), oldTriangleIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell1 =  container->m_triangleEdgeShell[ container->m_triangleEdge[indices[i]][1]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell1.begin(), shell1.end(), oldTriangleIndex ) !=shell1.end());
                    it=std::find( shell1.begin(), shell1.end(), oldTriangleIndex );
                    (*it)=indices[i];

                    sofa::helper::vector< unsigned int > &shell2 =  container->m_triangleEdgeShell[ container->m_triangleEdge[indices[i]][2]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell2.begin(), shell2.end(), oldTriangleIndex) !=shell2.end());
                    it=std::find( shell2.begin(), shell2.end(), oldTriangleIndex );
                    (*it)=indices[i];

                }

            }
        }
        if ( (edgeToBeRemoved.size()>0) || (vertexToBeRemoved.size()>0))
        {

            if (edgeToBeRemoved.size()>0)
                /// warn that edges will be deleted
                this->removeEdgesWarning(edgeToBeRemoved);

            //if (vertexToBeRemoved.size()>0)
            //this->removePointsWarning(vertexToBeRemoved);

            /// propagate to all components
            topology->propagateTopologicalChanges();
            if (edgeToBeRemoved.size()>0)
                /// actually remove edges without looking for isolated vertices
                this->removeEdgesProcess(edgeToBeRemoved,false);


            if (vertexToBeRemoved.size()>0)
            {
                this->removePointsWarning(vertexToBeRemoved);
            }
            topology->propagateTopologicalChanges();

            if (vertexToBeRemoved.size()>0)
            {
                this->removePointsProcess(vertexToBeRemoved);
            }

        }
    }
}

template<class DataTypes >
void TriangleSetTopologyModifier< DataTypes >::addPointsProcess(const unsigned int nPoints,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
{
    // start by calling the standard method.
    EdgeSetTopologyModifier< DataTypes >::addPointsProcess( nPoints, ancestors, baryCoefs );

    // now update the local container structures.
    TriangleSetTopology<DataTypes> *topology = dynamic_cast<TriangleSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast<TriangleSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    container->m_triangleVertexShell.resize( container->m_triangleVertexShell.size() + nPoints );
}


template<class DataTypes >
void TriangleSetTopologyModifier< DataTypes >::addEdgesProcess(const sofa::helper::vector< Edge > &edges)
{
    // start by calling the standard method.
    EdgeSetTopologyModifier< DataTypes >::addEdgesProcess( edges );

    // now update the local container structures.
    TriangleSetTopology<DataTypes> *topology = dynamic_cast<TriangleSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast<TriangleSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    container->m_triangleEdgeShell.resize( container->m_triangleEdgeShell.size() + edges.size() );
}



template< class DataTypes >
void TriangleSetTopologyModifier< DataTypes >::removePointsProcess( sofa::helper::vector<unsigned int> &indices, const bool removeDOF)
{
    // Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
    // now update the local container structures
    TriangleSetTopology<DataTypes> *topology = dynamic_cast<TriangleSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast<TriangleSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);

    // force the creation of the triangle vertex shell array before any point is deleted
    container->getTriangleVertexShellArray();

    // start by calling the standard method.
    EdgeSetTopologyModifier< DataTypes >::removePointsProcess( indices, removeDOF );

    int vertexIndex;

    unsigned int lastPoint = container->m_triangleVertexShell.size() - 1;


    for (unsigned int i = 0; i < indices.size(); ++i)
    {
        // updating the triangles connected to the point replacing the removed one:
        // for all triangles connected to the last point
        sofa::helper::vector<unsigned int>::iterator itt=container->m_triangleVertexShell[lastPoint].begin();
        for (; itt!=container->m_triangleVertexShell[lastPoint].end(); ++itt)
        {

            vertexIndex=container->getVertexIndexInTriangle(container->m_triangle[(*itt)],lastPoint);
            assert(vertexIndex!= -1);
            container->m_triangle[(*itt)][(unsigned int)vertexIndex]=indices[i];
        }

        // updating the edge shell itself (change the old index for the new one)
        container->m_triangleVertexShell[ indices[i] ] = container->m_triangleVertexShell[ lastPoint ];

        --lastPoint;
    }

    container->m_triangleVertexShell.resize( container->m_triangleVertexShell.size() - indices.size() );
}

template< class DataTypes >
void TriangleSetTopologyModifier< DataTypes >::removeEdgesProcess( const sofa::helper::vector<unsigned int> &indices,const bool removeIsolatedItems)
{
    // now update the local container structures
    TriangleSetTopology<DataTypes> *topology = dynamic_cast<TriangleSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast<TriangleSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);

    //if (container->m_triangleEdge.size()>0)
    container->getTriangleEdgeShellArray();

    // start by calling the standard method.
    EdgeSetTopologyModifier< DataTypes >::removeEdgesProcess(indices,removeIsolatedItems);

    if (container->m_triangleEdge.size()>0)
    {
        unsigned int edgeIndex;
        unsigned int lastEdge = container->m_triangleEdgeShell.size() - 1;

        for (unsigned int i = 0; i < indices.size(); ++i)
        {
            // updating the triangles connected to the edge replacing the removed one:
            // for all triangles connected to the last point
            sofa::helper::vector<unsigned int>::iterator itt=container->m_triangleEdgeShell[lastEdge].begin();
            for (; itt!=container->m_triangleEdgeShell[lastEdge].end(); ++itt)
            {

                edgeIndex=container->getEdgeIndexInTriangle(container->m_triangleEdge[(*itt)],lastEdge);
                assert((int)edgeIndex!= -1);
                container->m_triangleEdge[(*itt)][(unsigned int)edgeIndex]=indices[i];
            }

            // updating the edge shell itself (change the old index for the new one)
            container->m_triangleEdgeShell[ indices[i] ] = container->m_triangleEdgeShell[ lastEdge ];

            --lastEdge;
        }

        container->m_triangleEdgeShell.resize( container->m_triangleEdgeShell.size() - indices.size() );
    }
}



template< class DataTypes >
void TriangleSetTopologyModifier< DataTypes >::renumberPointsProcess( const sofa::helper::vector<unsigned int> &index)
{
    // start by calling the standard method
    EdgeSetTopologyModifier< DataTypes >::renumberPointsProcess( index );

    // now update the local container structures.
    TriangleSetTopology<DataTypes> *topology = dynamic_cast<TriangleSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast<TriangleSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);

    sofa::helper::vector< sofa::helper::vector< unsigned int > > triangleVertexShell_cp = container->m_triangleVertexShell;
    for (unsigned int i = 0; i < index.size(); ++i)
    {
        container->m_triangleVertexShell[i] = triangleVertexShell_cp[ index[i] ];
    }

    for (unsigned int i = 0; i < container->m_triangle.size(); ++i)
    {
        container->m_triangle[i][0]  = index[ container->m_triangle[i][0]  ];
        container->m_triangle[i][1]  = index[ container->m_triangle[i][1]  ];
        container->m_triangle[i][2]  = index[ container->m_triangle[i][2]  ];
    }


}
/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////TriangleSetTopologyAlgorithms//////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
template<class DataTypes>
void TriangleSetTopologyAlgorithms< DataTypes >::removeTriangles(sofa::helper::vector< unsigned int >& triangles, const bool removeIsolatedEdges, const bool removeIsolatedPoints)
{
    TriangleSetTopology< DataTypes > *topology = dynamic_cast<TriangleSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyModifier< DataTypes >* modifier  = static_cast< TriangleSetTopologyModifier< DataTypes >* >(topology->getTopologyModifier());
    assert(modifier != 0);
    /// add the topological changes in the queue
    modifier->removeTrianglesWarning(triangles);
    // inform other objects that the triangles are going to be removed
    topology->propagateTopologicalChanges();
    // now destroy the old triangles.

    modifier->removeTrianglesProcess(  triangles ,removeIsolatedEdges, removeIsolatedPoints);

    //assert(topology->getTriangleSetTopologyContainer()->checkTopology());
    topology->getTriangleSetTopologyContainer()->checkTopology();
}

// Preparation of "InciseAlongPointsList" :
// if the input points a and b are equal, then return false
// if a and b are distinct but belong to the same triangle indexed by ind_a (= ind_p), then remesh the input triangle (such that a and b belongs to distinct triangles indexed by new_ind_ta and new_ind_tb) and return true

template<class DataTypes>
double TriangleSetTopologyAlgorithms< DataTypes >::Prepare_InciseAlongPointsList(const Vec<3,double>& a, const Vec<3,double>& b, const unsigned int ind_ta, const unsigned int ind_tb, unsigned int new_ind_ta, unsigned int new_ind_tb)
{

    // Initialization of output variables
    double is_validated=(!(a[0]==b[0] && a[1]==b[1] && a[2]==b[2]));
    new_ind_ta=(unsigned int) ind_ta;
    new_ind_tb=(unsigned int) ind_tb;

    // Access the topology
    TriangleSetTopology<DataTypes> *topology = dynamic_cast<TriangleSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast< TriangleSetTopologyContainer* >(topology->getTopologyContainer());

    TriangleSetTopologyModifier< DataTypes >* modifier  = static_cast< TriangleSetTopologyModifier< DataTypes >* >(topology->getTopologyModifier());
    assert(modifier != 0);

    const Triangle &ta=container->getTriangle(ind_ta);

    const typename DataTypes::VecCoord& vect_c = *topology->getDOF()->getX();
    unsigned int nb_points =  vect_c.size() -1;

    const sofa::helper::vector<Triangle> &vect_t=container->getTriangleArray();
    unsigned int nb_triangles =  vect_t.size() -1;

    // Variables to accumulate the number of elements registered to be created (so as to remember their indices)
    unsigned int acc_nb_points=nb_points;
    unsigned int acc_nb_triangles=nb_triangles;

    // Variables to accumulate the elements registered to be created or to be removed
    sofa::helper::vector< sofa::helper::vector< unsigned int > > p_ancestors;
    sofa::helper::vector< sofa::helper::vector< double > > p_baryCoefs;
    sofa::helper::vector< Triangle > triangles_to_create;
    sofa::helper::vector< unsigned int > triangles_to_remove;

    sofa::helper::vector< double > a_baryCoefs = topology->getTriangleSetGeometryAlgorithms()->computeTriangleBarycoefs((const Vec<3,double> &) a, ind_ta);
    sofa::helper::vector< double > b_baryCoefs = topology->getTriangleSetGeometryAlgorithms()->computeTriangleBarycoefs((const Vec<3,double> &) b, ind_tb);

    /// force the creation of TriangleEdgeShellArray
    container->getTriangleEdgeShellArray();
    /// force the creation of TriangleVertexShellArray
    container->getTriangleVertexShellArray();

    if(ind_ta==ind_tb)  // create the middle point of (a,b), then subdivide the input triangle (indexed by ind_ta = ind_t) into 3 triangles incident to this middle point
    {

        sofa::helper::vector< unsigned int > mid_ancestors;
        mid_ancestors.push_back(ta[0]); mid_ancestors.push_back(ta[1]); mid_ancestors.push_back(ta[2]);
        p_ancestors.push_back(mid_ancestors);

        sofa::helper::vector< double > mid_baryCoefs;
        mid_baryCoefs.push_back((double) (0.5*(a_baryCoefs[0]+b_baryCoefs[0])));
        mid_baryCoefs.push_back((double) (0.5*(a_baryCoefs[1]+b_baryCoefs[1])));
        mid_baryCoefs.push_back((double) (0.5*(a_baryCoefs[2]+b_baryCoefs[2])));
        p_baryCoefs.push_back(mid_baryCoefs);

        acc_nb_points=acc_nb_points+1;

        Triangle t_01 = Triangle(helper::make_array<unsigned int>((unsigned int)acc_nb_points,(unsigned int)ta[0],(unsigned int) ta[1]));
        Triangle t_12 = Triangle(helper::make_array<unsigned int>((unsigned int)acc_nb_points,(unsigned int)ta[1],(unsigned int) ta[2]));
        Triangle t_20 = Triangle(helper::make_array<unsigned int>((unsigned int)acc_nb_points,(unsigned int)ta[2],(unsigned int) ta[0]));
        triangles_to_create.push_back(t_01); triangles_to_create.push_back(t_12); triangles_to_create.push_back(t_20);

        acc_nb_triangles=acc_nb_triangles+3;

        triangles_to_remove.push_back(ind_ta);

        modifier->addPointsProcess( acc_nb_points - nb_points, p_ancestors, p_baryCoefs);

        modifier->addPointsWarning( acc_nb_points - nb_points, p_ancestors, p_baryCoefs);

        modifier->addTrianglesProcess( triangles_to_create) ; // WARNING included after

        topology->propagateTopologicalChanges();

        removeTriangles(triangles_to_remove, true, true); // WARNING and PROPAGATED included before

        topology->propagateTopologicalChanges();

        if(is_validated)  // localize the sub_triangles indexed by new_ind_ta and new_ind_tb
        {

            const Vec<3,double> p_mid(0.5*(a+b));

            /*
              double is_a_0_01 = 0.5*a[0]*a[2]-p_mid[0]*p_mid[2];
              double is_a_1_01 = 0.5*a[1]*a[2]-p_mid[1]*p_mid[2];
              bool is_a_in_01 = (is_a_0_01>=0.0) && (is_a_1_01>=0.0);

              double is_a_1_12 = 0.5*a[1]*a[0]-p_mid[1]*p_mid[0];
              double is_a_2_12 = 0.5*a[2]*a[0]-p_mid[2]*p_mid[0];
              bool is_a_in_12 = (is_a_1_12>=0.0) && (is_a_2_12>=0.0);

              double is_a_2_20 = 0.5*a[2]*a[1]-p_mid[2]*p_mid[1];
              double is_a_0_20 = 0.5*a[0]*a[1]-p_mid[0]*p_mid[1];
              bool is_a_in_20 = (is_a_2_20>=0.0) && (is_a_0_20>=0.0);

              double is_b_0_01 = 0.5*b[0]*b[2]-p_mid[0]*p_mid[2];
              double is_b_1_01 = 0.5*b[1]*b[2]-p_mid[1]*p_mid[2];
              bool is_b_in_01 = (is_b_0_01>=0.0) && (is_b_1_01>=0.0);

              double is_b_1_12 = 0.5*b[1]*b[0]-p_mid[1]*p_mid[0];
              double is_b_2_12 = 0.5*b[2]*b[0]-p_mid[2]*p_mid[0];
              bool is_b_in_12 = (is_b_1_12>=0.0) && (is_b_2_12>=0.0);

              double is_b_2_20 = 0.5*b[2]*b[1]-p_mid[2]*p_mid[1];
              double is_b_0_20 = 0.5*b[0]*b[1]-p_mid[0]*p_mid[1];
              bool is_b_in_20 = (is_b_2_20>=0.0) && (is_b_0_20>=0.0);

            */

            if(is_point_in_triangle(a, p_mid, (const Vec<3,double>& ) ta[0], (const Vec<3,double>& )ta[1])) //is_a_in_01){ // a in (0,1)
            {

                new_ind_ta=acc_nb_triangles-2;

                if(is_point_in_triangle(b, p_mid,(const Vec<3,double>& )ta[1],(const Vec<3,double>& )ta[2])) //is_b_in_12){ // b in (1,2)
                {
                    new_ind_tb=acc_nb_triangles-1;

                }
                else   // b in (2,0)
                {
                    new_ind_tb=acc_nb_triangles;
                }
            }
            else
            {
                if(is_point_in_triangle(a, p_mid, (const Vec<3,double>& ) ta[1], (const Vec<3,double>& )ta[2])) //is_a_in_12){ // a in (1,2)
                {

                    new_ind_ta=acc_nb_triangles-1;

                    if(is_point_in_triangle(b, p_mid,(const Vec<3,double>& )ta[2], (const Vec<3,double>& ) ta[0])) //is_b_in_20){ // b in (2,0)
                    {
                        new_ind_tb=acc_nb_triangles;

                    }
                    else   // b in (0,1)
                    {
                        new_ind_tb=acc_nb_triangles-2;
                    }
                }
                else   // a in (2,0)
                {

                    new_ind_ta=acc_nb_triangles;

                    if(is_point_in_triangle(b, p_mid,(const Vec<3,double>& ) ta[0], (const Vec<3,double>& )ta[1])) //is_b_in_01){ // b in (0,1)
                    {
                        new_ind_tb=acc_nb_triangles-2;

                    }
                    else   // b in (1,2)
                    {
                        new_ind_tb=acc_nb_triangles-1;
                    }
                }
            }

            // Call the method "InciseAlongPointsList" on the new parameters

            //InciseAlongPointsList( a,  b,  new_ind_ta, new_ind_tb);

        }

    }

    return is_validated;

}




// Incises along the list of points (ind_edge,coord) intersected by the segment from point a to point b and the triangular mesh
// Point a belongs to the triangle sindexed by ind_ta
// Point b belongs to the triangle sindexed by ind_tb

template<class DataTypes>
bool TriangleSetTopologyAlgorithms< DataTypes >::InciseAlongPointsList(bool is_first_cut, const Vec<3,double>& a, const Vec<3,double>& b,
        const unsigned int ind_ta, const unsigned int ind_tb,
        unsigned int& a_last, sofa::helper::vector< unsigned int > &a_p12_last, sofa::helper::vector< unsigned int > &a_i123_last,
        unsigned int& b_last, sofa::helper::vector< unsigned int > &b_p12_last, sofa::helper::vector< unsigned int > &b_i123_last, sofa::helper::vector< sofa::helper::vector<unsigned int> > &new_points, sofa::helper::vector< sofa::helper::vector<unsigned int> > &closest_vertices)
{

    double epsilon = 0.2; // INFO : epsilon is a threshold in [0,1] to control the snapping of the extremities to the closest vertex

    unsigned int x_i1 = 0;
    unsigned int x_i2 = 0;
    unsigned int x_i3 = 0;
    unsigned int x_i1_to = 0;
    unsigned int x_i2_to = 0;
    unsigned int x_p1 = 0;
    unsigned int x_p2 = 0;
    unsigned int x_p1_to = 0;
    unsigned int x_p2_to = 0;

    // Access the topology
    TriangleSetTopology<DataTypes> *topology = dynamic_cast<TriangleSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);

    Vec<3,Real> a_new = a;
    unsigned int ind_ta_new = ind_ta;
    Vec<3,Real> b_new = b;
    unsigned int ind_tb_new = ind_tb;

    unsigned int ind_ta_test_init;
    unsigned int ind_tb_test_init;

    unsigned int &ind_ta_test = ind_ta_test_init;
    unsigned int &ind_tb_test = ind_tb_test_init;

    unsigned int ind_tb_final_init;
    unsigned int &ind_tb_final = ind_tb_final_init;

    bool is_on_boundary_init=false;
    bool &is_on_boundary=is_on_boundary_init;

    if(is_first_cut)
    {
        bool is_a_inside = topology->getTriangleSetGeometryAlgorithms()->is_PointinTriangle(true, a_new, ind_ta_new, ind_ta_test);
        if(is_a_inside)
        {
            //std::cout << "a is inside" <<  std::endl;
        }
        else
        {
            //std::cout << "a is NOT inside !!!" <<  std::endl;
            if(ind_ta_new==ind_ta_test)  // fail
            {
                //std::cout << "fail !!!" <<  std::endl;
                return false;
            }
            else
            {
                ind_ta_new=ind_ta_test;
            }
        }
    }

    bool is_b_inside = topology->getTriangleSetGeometryAlgorithms()->is_PointinTriangle(true, b_new, ind_tb_new, ind_tb_test);
    if(is_b_inside)
    {
        //std::cout << "b is inside" <<  std::endl;
    }
    else
    {
        //std::cout << "b is NOT inside !!!" <<  std::endl;
        if(ind_tb_new==ind_tb_test)  // fail
        {
            //std::cout << "fail !!!" <<  std::endl;
            return false;
        }
        else
        {
            ind_tb_new=ind_tb_test;
        }
    }

    if(ind_ta_new==ind_tb_new)
    {
        return false;
    }

    TriangleSetTopologyContainer * container = static_cast< TriangleSetTopologyContainer* >(topology->getTopologyContainer());

    TriangleSetTopologyModifier< DataTypes >* modifier  = static_cast< TriangleSetTopologyModifier< DataTypes >* >(topology->getTopologyModifier());
    assert(modifier != 0);

    const typename DataTypes::VecCoord& vect_c = *topology->getDOF()->getX();
    unsigned int nb_points =  vect_c.size() -1;

    const sofa::helper::vector<Triangle> &vect_t=container->getTriangleArray();
    unsigned int nb_triangles =  vect_t.size() -1;

    // Variables to accumulate the number of elements registered to be created (so as to remember their indices)
    unsigned int acc_nb_points=nb_points;
    unsigned int acc_nb_triangles=nb_triangles;

    // Variables to accumulate the elements registered to be created or to be removed
    sofa::helper::vector< sofa::helper::vector< unsigned int > > p_ancestors;
    sofa::helper::vector< sofa::helper::vector< double > > p_baryCoefs;
    sofa::helper::vector< Triangle > triangles_to_create;
    sofa::helper::vector< unsigned int > triangles_to_remove;

    // Initialization for INTERSECTION method
    sofa::helper::vector< unsigned int > triangles_list_init;
    sofa::helper::vector< unsigned int > &triangles_list = triangles_list_init;

    sofa::helper::vector< sofa::helper::vector<unsigned int> > indices_list_init;
    sofa::helper::vector< sofa::helper::vector<unsigned int> > &indices_list = indices_list_init;

    sofa::helper::vector< double > coords_list_init;
    sofa::helper::vector< double >& coords_list=coords_list_init;

    bool is_intersected=false;

    // Pre-treatment if is_first_cut ==false :

    if(!is_first_cut)
    {

        x_p1 = b_p12_last[0];
        x_p2 = b_p12_last[1];
        x_i1 = b_i123_last[0];
        x_i2 = b_i123_last[1];
        x_i3 = b_i123_last[2];

        const typename DataTypes::Coord& b_point_last=vect_c[b_last];

        a_new[0]= (Real) b_point_last[0];
        a_new[1]= (Real) b_point_last[1];
        a_new[2]= (Real) b_point_last[2];

        bool is_crossed = false;
        double coord_kmin = 0.0;

        const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tvsa=container->getTriangleVertexShellArray();
        if (tvsa.size()>0)
        {

            sofa::helper::vector< unsigned int > shell_b =(sofa::helper::vector< unsigned int >) (tvsa[b_last]);
            unsigned int ind_t_test;
            unsigned int i=0;


            if(shell_b.size()>0)
            {

                while(i < shell_b.size())
                {

                    ind_t_test=shell_b[i];
                    triangles_to_remove.push_back(ind_t_test);

                    sofa::helper::vector<unsigned int> c_indices_init; sofa::helper::vector<unsigned int> &c_indices=c_indices_init;
                    double c_baryCoef_init; double &c_baryCoef=c_baryCoef_init;
                    double c_coord_k_init; double &c_coord_k = c_coord_k_init;

                    bool is_intersection_found = topology->getTriangleSetGeometryAlgorithms()->computeSegmentTriangleIntersection(false, (const Vec<3,double>&) a_new, (const Vec<3,double>&) b, ind_t_test,
                            c_indices, c_baryCoef, c_coord_k);

                    if(is_intersection_found)
                    {
                        is_intersection_found=is_intersection_found && (c_indices[0] != b_last && c_indices[1] != b_last);
                    }

                    if(is_intersection_found && c_coord_k>coord_kmin)
                    {
                        ind_ta_new=ind_t_test;
                        coord_kmin=c_coord_k;
                    }

                    is_crossed = is_crossed || is_intersection_found;

                    i++;
                }

                if(is_crossed)
                {

                    if(ind_ta_new==ind_tb_new)
                    {
                        return false;
                    }

                    ind_tb_final=ind_tb_new;
                    is_intersected = topology->getTriangleSetGeometryAlgorithms()->computeIntersectedPointsList((const Vec<3,double>&) a_new, b, ind_ta_new, ind_tb_final, triangles_list, indices_list, coords_list, is_on_boundary);

                }
                else
                {
                    return false;
                }
            }
        }

    }
    else
    {

        // Call the method "computeIntersectedPointsList" to get the list of points (ind_edge,coord) intersected by the segment from point a to point b and the triangular mesh
        ind_tb_final=ind_tb_new;
        is_intersected = topology->getTriangleSetGeometryAlgorithms()->computeIntersectedPointsList(a, b, ind_ta_new, ind_tb_final, triangles_list, indices_list, coords_list, is_on_boundary);

    }

    unsigned int elem_size = triangles_list.size();

    if(elem_size>0)  // intersection successfull
    {

        /// force the creation of TriangleEdgeShellArray
        container->getTriangleEdgeShellArray();
        /// force the creation of TriangleVertexShellArray
        container->getTriangleVertexShellArray();

        // Initialization for the indices of the previous intersected edge
        unsigned int p1_prev=0;
        unsigned int p2_prev=0;

        // Treatment of particular case for first extremity a

        const Triangle &ta=container->getTriangle(ind_ta_new);
        unsigned int ta_to_remove;
        unsigned int p1_a=indices_list[0][0];
        unsigned int p2_a=indices_list[0][1];

        // Plan to remove triangles indexed by ind_ta_new
        if(is_first_cut)
        {
            triangles_to_remove.push_back(ind_ta_new);
        }

        // Initialization for SNAPPING method for point a

        bool is_snap_a0_init=false; bool is_snap_a1_init=false; bool is_snap_a2_init=false;
        bool& is_snap_a0=is_snap_a0_init;
        bool& is_snap_a1=is_snap_a1_init;
        bool& is_snap_a2=is_snap_a2_init;

        sofa::helper::vector< double > a_baryCoefs = topology->getTriangleSetGeometryAlgorithms()->computeTriangleBarycoefs((const Vec<3,double> &) a_new, ind_ta_new);
        snapping_test_triangle(epsilon, a_baryCoefs[0], a_baryCoefs[1], a_baryCoefs[2], is_snap_a0, is_snap_a1, is_snap_a2);
        double is_snapping_a = is_snap_a0 || is_snap_a1 || is_snap_a2;

        //std::cout << "a_baryCoefs = " << a_baryCoefs[0] << ", " << a_baryCoefs[1] << ", " << a_baryCoefs[2] <<  std::endl;

        sofa::helper::vector< unsigned int > a_first_ancestors;
        sofa::helper::vector< double > a_first_baryCoefs;

        if((is_first_cut) && (!is_snapping_a))
        {

            /// Register the creation of point a

            a_first_ancestors.push_back(ta[0]); a_first_ancestors.push_back(ta[1]); a_first_ancestors.push_back(ta[2]);
            p_ancestors.push_back(a_first_ancestors);
            p_baryCoefs.push_back(a_baryCoefs);

            acc_nb_points=acc_nb_points+1;

            /// Register the creation of triangles incident to point a

            unsigned int ind_a =  acc_nb_points; // last point registered to be created

            a_last=ind_a; // OUPTUT

            a_p12_last.clear();
            a_p12_last.push_back(acc_nb_points+2); // OUPTUT
            a_p12_last.push_back(acc_nb_points+1); // OUPTUT

            sofa::helper::vector< Triangle > a_triangles;
            Triangle t_a01 = Triangle(helper::make_array<unsigned int>((unsigned int)ind_a,(unsigned int)ta[0],(unsigned int) ta[1]));
            Triangle t_a12 = Triangle(helper::make_array<unsigned int>((unsigned int)ind_a,(unsigned int)ta[1],(unsigned int) ta[2]));
            Triangle t_a20 = Triangle(helper::make_array<unsigned int>((unsigned int)ind_a,(unsigned int)ta[2],(unsigned int) ta[0]));
            triangles_to_create.push_back(t_a01); triangles_to_create.push_back(t_a12); triangles_to_create.push_back(t_a20);

            acc_nb_triangles=acc_nb_triangles+3;

            /// Register the removal of triangles incident to point a

            a_i123_last.clear();
            a_i123_last.push_back(p2_a); // OUPTUT
            a_i123_last.push_back(p1_a); // OUPTUT

            if(ta[0]!=p1_a && ta[0]!=p2_a)
            {
                ta_to_remove=acc_nb_triangles-1;
                a_i123_last.push_back(ta[0]); // OUPTUT

            }
            else
            {
                if(ta[1]!=p1_a && ta[1]!=p2_a)
                {
                    ta_to_remove=acc_nb_triangles;
                    a_i123_last.push_back(ta[1]); // OUPTUT

                }
                else   // (ta[2]!=p1_a && ta[2]!=p2_a)
                {
                    ta_to_remove=acc_nb_triangles-2;
                    a_i123_last.push_back(ta[2]); // OUPTUT
                }
            }
            triangles_to_remove.push_back(ta_to_remove);

            Triangle t_pa1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 1,(unsigned int) ind_a,(unsigned int) p1_a));
            Triangle t_pa2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 2,(unsigned int) p2_a, (unsigned int)ind_a));
            triangles_to_create.push_back(t_pa1); triangles_to_create.push_back(t_pa2);

            acc_nb_triangles=acc_nb_triangles+2;

        }
        else   // (is_first_cut == false) or : snapping a to the vertex indexed by ind_a, which is the closest to point a
        {

            x_p1_to = acc_nb_points + 1;
            x_p2_to = acc_nb_points + 2;

            x_i1_to = p1_a;
            x_i2_to = p2_a;

            // localize the closest vertex

            unsigned int ind_a;
            unsigned int p0_a;

            if(ta[0]!=p1_a && ta[0]!=p2_a)
            {
                p0_a=ta[0];
            }
            else
            {
                if(ta[1]!=p1_a && ta[1]!=p2_a)
                {
                    p0_a=ta[1];
                }
                else  // ta[2]!=p1_a && ta[2]!=p2_a
                {
                    p0_a=ta[2];
                }
            }

            if(is_snap_a0)  // is_snap_a1 == false and is_snap_a2 == false
            {
                /// VERTEX 0
                ind_a=ta[0];
            }
            else
            {
                if(is_snap_a1)  // is_snap_a0 == false and is_snap_a2 == false
                {
                    /// VERTEX 1
                    ind_a=ta[1];
                }
                else   // is_snap_a2 == true and (is_snap_a0 == false and is_snap_a1 == false)
                {
                    /// VERTEX 2
                    ind_a=ta[2];
                }
            }

            /// Register the creation of triangles incident to point indexed by ind_a

            Triangle t_pa1;
            Triangle t_pa2;

            if(ind_a==p1_a)
            {
                t_pa1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 2,(unsigned int) p0_a,(unsigned int) p1_a));
                t_pa2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 2,(unsigned int) p2_a, (unsigned int) p0_a));
                triangles_to_create.push_back(t_pa1); triangles_to_create.push_back(t_pa2);

            }
            else
            {

                if(ind_a==p2_a)
                {
                    t_pa1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 1,(unsigned int) p0_a,(unsigned int) p1_a));
                    t_pa2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 1,(unsigned int) p2_a, (unsigned int) p0_a));
                    triangles_to_create.push_back(t_pa1); triangles_to_create.push_back(t_pa2);

                }
                else
                {

                    t_pa1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 1,(unsigned int) ind_a,(unsigned int) p1_a));
                    t_pa2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 2,(unsigned int) p2_a, (unsigned int)ind_a));
                    triangles_to_create.push_back(t_pa1); triangles_to_create.push_back(t_pa2);
                }
            }

            acc_nb_triangles+=2;

            if(!is_first_cut)
            {
                triangles_to_remove.push_back(acc_nb_triangles-1);
                triangles_to_remove.push_back(acc_nb_triangles);
            }

        }

        // Traverse the loop of interected edges

        for (unsigned int i=0; i<indices_list.size(); i++)
        {

            /// Register the creation of the two points (say current "duplicated points") localized on the current interected edge

            unsigned int p1 = indices_list[i][0];
            unsigned int p2 = indices_list[i][1];

            sofa::helper::vector< unsigned int > p_first_ancestors;
            p_first_ancestors.push_back(p1); p_first_ancestors.push_back(p2);
            p_ancestors.push_back(p_first_ancestors); p_ancestors.push_back(p_first_ancestors);

            sofa::helper::vector< double > p_first_baryCoefs;
            p_first_baryCoefs.push_back(1.0 - coords_list[i]); p_first_baryCoefs.push_back(coords_list[i]);
            p_baryCoefs.push_back(p_first_baryCoefs); p_baryCoefs.push_back(p_first_baryCoefs);

            acc_nb_points=acc_nb_points+2;

            sofa::helper::vector<unsigned int> new_points_current;
            new_points_current.push_back(acc_nb_points-1); new_points_current.push_back(acc_nb_points);
            new_points.push_back(new_points_current);
            closest_vertices.push_back(indices_list[i]);

            if(i>0)  // not to treat particular case of first extremitiy
            {

                // SNAPPING TEST

                double gamma = 0.3;
                bool is_snap_p1;
                bool is_snap_p2;

                snapping_test_edge(gamma, 1.0 - coords_list[i], coords_list[i], is_snap_p1, is_snap_p2);
                double is_snapping_p = is_snap_p1 || is_snap_p2;

                unsigned int ind_p;

                if(is_snapping_p && i<indices_list.size()-1)  // not to treat particular case of last extremitiy
                {

                    if(is_snap_p1)
                    {
                        /// VERTEX 0
                        ind_p=p1;
                    }
                    else   // is_snap_p2 == true
                    {
                        /// VERTEX 1
                        ind_p=p2;
                    }

                    // std::cout << "INFO_print : DO is_snapping_p, i = " << i << " on vertex " << ind_p <<  std::endl;

                    sofa::helper::vector< unsigned int > triangles_list_1_init;
                    sofa::helper::vector< unsigned int > &triangles_list_1 = triangles_list_1_init;

                    sofa::helper::vector< unsigned int > triangles_list_2_init;
                    sofa::helper::vector< unsigned int > &triangles_list_2 = triangles_list_2_init;

                    // std::cout << "INFO_print : DO Prepare_VertexDuplication " <<  std::endl;
                    topology->getTriangleSetGeometryAlgorithms()->Prepare_VertexDuplication(ind_p, triangles_list[i], triangles_list[i+1], indices_list[i-1], coords_list[i-1], indices_list[i+1], coords_list[i+1], triangles_list_1, triangles_list_2);
                    // std::cout << "INFO_print : DONE Prepare_VertexDuplication " <<  std::endl;

                    // std::cout << "INFO_print : triangles_list_1.size() = " << triangles_list_1.size() <<  std::endl;
                    //for (unsigned int k=0;k<triangles_list_1.size();k++){
                    // std::cout << "INFO_print : triangles_list_1 number " << k << " = " << triangles_list_1[k] <<  std::endl;
                    //}

                    // std::cout << "INFO_print : triangles_list_2.size() = " << triangles_list_2.size() <<  std::endl;
                    //for (unsigned int k=0;k<triangles_list_2.size();k++){
                    // std::cout << "INFO_print : triangles_list_2 number " << k << " = " << triangles_list_2[k] <<  std::endl;
                    //}

                }

                /// Register the removal of the current triangle

                triangles_to_remove.push_back(triangles_list[i]);

                /// Register the creation of triangles incident to the current "duplicated points" and to the previous "duplicated points"

                unsigned int p1_created=acc_nb_points - 3;
                unsigned int p2_created=acc_nb_points - 2;

                unsigned int p1_to_create=acc_nb_points - 1;
                unsigned int p2_to_create=acc_nb_points;

                unsigned int p0_t = container->getTriangle(triangles_list[i])[0];
                unsigned int p1_t = container->getTriangle(triangles_list[i])[1];
                unsigned int p2_t = container->getTriangle(triangles_list[i])[2];

                Triangle t_p1;
                Triangle t_p2;
                Triangle t_p3;

                unsigned int ind_quad;
                Vec<3,Real> point_created=(Vec<3,double>) topology->getTriangleSetGeometryAlgorithms()->computeBaryEdgePoint(indices_list[i-1], coords_list[i-1]);
                Vec<3,Real> point_to_create=(Vec<3,double>) topology->getTriangleSetGeometryAlgorithms()->computeBaryEdgePoint(indices_list[i], coords_list[i]);

                if(p0_t!=p1_prev && p0_t!=p2_prev)
                {
                    ind_quad=p0_t;
                }
                else
                {
                    if(p1_t!=p1_prev && p1_t!=p2_prev)
                    {
                        ind_quad=p1_t;
                    }
                    else   // (p2_t!=p1_prev && p2_t!=p2_prev)
                    {
                        ind_quad=p2_t;
                    }
                }

                if(ind_quad==p1)  // *** p1_to_create - p1_created - p1_prev - ind_quad
                {

                    t_p1 = Triangle(helper::make_array<unsigned int>(p2_created, p2_to_create, p2_prev));
                    if(topology->getTriangleSetGeometryAlgorithms()->isQuadDeulaunayOriented(point_to_create, point_created, p1_prev, ind_quad))
                    {
                        t_p2 = Triangle(helper::make_array<unsigned int>(p1_to_create, p1_created, p1_prev));
                        t_p3 = Triangle(helper::make_array<unsigned int>(p1_prev, ind_quad, p1_to_create));
                    }
                    else
                    {
                        t_p2 = Triangle(helper::make_array<unsigned int>(p1_created, p1_prev, ind_quad));
                        t_p3 = Triangle(helper::make_array<unsigned int>(ind_quad, p1_to_create, p1_created));
                    }

                }
                else   // ind_quad==p2 // *** p2_created - p2_to_create - ind_quad - p2_prev
                {

                    t_p1 = Triangle(helper::make_array<unsigned int>(p1_to_create, p1_created, p1_prev));
                    if(topology->getTriangleSetGeometryAlgorithms()->isQuadDeulaunayOriented(point_created, point_to_create, ind_quad, p2_prev))
                    {
                        t_p2 = Triangle(helper::make_array<unsigned int>(p2_created, p2_to_create, ind_quad));
                        t_p3 = Triangle(helper::make_array<unsigned int>(ind_quad, p2_prev, p2_created));
                    }
                    else
                    {
                        t_p2 = Triangle(helper::make_array<unsigned int>(p2_to_create, ind_quad, p2_prev));
                        t_p3 = Triangle(helper::make_array<unsigned int>(p2_prev, p2_created, p2_to_create));
                    }
                }

                triangles_to_create.push_back(t_p1); triangles_to_create.push_back(t_p2); triangles_to_create.push_back(t_p3);

                acc_nb_triangles=acc_nb_triangles+3;

            }

            // Update the previous "duplicated points"
            p1_prev=p1;
            p2_prev=p2;

        }

        if(is_intersected || !is_on_boundary)
        {

            ind_tb_new=ind_tb_final;

            b_p12_last.clear();
            b_p12_last.push_back(acc_nb_points-1); // OUPTUT
            b_p12_last.push_back(acc_nb_points); // OUPTUT

            // Treatment of particular case for second extremity b

            const Triangle &tb=container->getTriangle(ind_tb_new);
            unsigned int tb_to_remove;
            unsigned int p1_b=indices_list[indices_list.size()-1][0];
            unsigned int p2_b=indices_list[indices_list.size()-1][1];

            b_i123_last.clear();
            b_i123_last.push_back(p1_b); // OUPTUT
            b_i123_last.push_back(p2_b); // OUPTUT

            // Plan to remove triangles indexed by ind_tb_new
            triangles_to_remove.push_back(ind_tb_new);

            // Initialization for SNAPPING method for point b

            bool is_snap_b0_init=false; bool is_snap_b1_init=false; bool is_snap_b2_init=false;
            bool& is_snap_b0=is_snap_b0_init;
            bool& is_snap_b1=is_snap_b1_init;
            bool& is_snap_b2=is_snap_b2_init;

            sofa::helper::vector< double > b_baryCoefs = topology->getTriangleSetGeometryAlgorithms()->computeTriangleBarycoefs((const Vec<3,double> &) b, ind_tb_new);

            if(!is_intersected && !is_on_boundary)
            {

                b_baryCoefs[0] = (double) (1.0/3.0);
                b_baryCoefs[1] = (double) (1.0/3.0);
                b_baryCoefs[2] = (double) (1.0 - (b_baryCoefs[0] + b_baryCoefs[1]));
            }

            snapping_test_triangle(epsilon, b_baryCoefs[0], b_baryCoefs[1], b_baryCoefs[2], is_snap_b0, is_snap_b1, is_snap_b2);
            double is_snapping_b = is_snap_b0 || is_snap_b1 || is_snap_b2;

            //std::cout << "b_baryCoefs = " << b_baryCoefs[0] << ", " << b_baryCoefs[1] << ", " << b_baryCoefs[2] <<  std::endl;

            sofa::helper::vector< unsigned int > b_first_ancestors;
            sofa::helper::vector< double > b_first_baryCoefs;

            is_snapping_b = false; // COMMENT : point b will not be snapped

            if(!is_snapping_b)
            {

                /// Register the creation of point b

                b_first_ancestors.push_back(tb[0]); b_first_ancestors.push_back(tb[1]); b_first_ancestors.push_back(tb[2]);
                p_ancestors.push_back(b_first_ancestors);
                p_baryCoefs.push_back(b_baryCoefs);

                acc_nb_points=acc_nb_points+1;

                /// Register the creation of triangles incident to point b

                unsigned int ind_b =  acc_nb_points; // last point registered to be created

                b_last=ind_b; // OUPTUT

                sofa::helper::vector< Triangle > b_triangles;
                Triangle t_b01 = Triangle(helper::make_array<unsigned int>((unsigned int)ind_b,(unsigned int)tb[0],(unsigned int) tb[1]));
                Triangle t_b12 = Triangle(helper::make_array<unsigned int>((unsigned int)ind_b,(unsigned int)tb[1],(unsigned int) tb[2]));
                Triangle t_b20 = Triangle(helper::make_array<unsigned int>((unsigned int)ind_b,(unsigned int)tb[2],(unsigned int) tb[0]));
                triangles_to_create.push_back(t_b01); triangles_to_create.push_back(t_b12); triangles_to_create.push_back(t_b20);

                acc_nb_triangles=acc_nb_triangles+3;

                /// Register the removal of triangles incident to point b

                if(tb[0]!=p1_b && tb[0]!=p2_b)
                {
                    tb_to_remove=acc_nb_triangles-1;
                    b_i123_last.push_back(tb[0]); // OUTPUT
                }
                else
                {
                    if(tb[1]!=p1_b && tb[1]!=p2_b)
                    {
                        tb_to_remove=acc_nb_triangles;
                        b_i123_last.push_back(tb[1]); // OUTPUT
                    }
                    else   // (tb[2]!=p1_b && tb[2]!=p2_b)
                    {
                        tb_to_remove=acc_nb_triangles-2;
                        b_i123_last.push_back(tb[2]); // OUTPUT
                    }
                }
                triangles_to_remove.push_back(tb_to_remove);

                Triangle t_pb1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points - 2,(unsigned int) p1_b,(unsigned int)ind_b));
                Triangle t_pb2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points - 1,(unsigned int)ind_b,(unsigned int) p2_b));
                triangles_to_create.push_back(t_pb1); triangles_to_create.push_back(t_pb2);

                acc_nb_triangles=acc_nb_triangles+2;

            }
            else   // snapping b to the vertex indexed by ind_b, which is the closest to point b
            {

                // localize the closest vertex
                unsigned int ind_b;
                unsigned int p0_b;

                if(tb[0]!=p1_b && tb[0]!=p2_b)
                {
                    p0_b=tb[0];
                    b_i123_last.push_back(tb[0]); // OUTPUT
                }
                else
                {
                    if(tb[1]!=p1_b && tb[1]!=p2_b)
                    {
                        p0_b=tb[1];
                        b_i123_last.push_back(tb[1]); // OUTPUT
                    }
                    else  // tb[2]!=p1_b && tb[2]!=p2_b
                    {
                        p0_b=tb[2];
                        b_i123_last.push_back(tb[2]); // OUTPUT
                    }
                }

                if(is_snap_b0)  // is_snap_b1 == false and is_snap_b2 == false
                {
                    /// VERTEX 0
                    ind_b=tb[0];
                }
                else
                {
                    if(is_snap_b1)  // is_snap_b0 == false and is_snap_b2 == false
                    {
                        /// VERTEX 1
                        ind_b=tb[1];
                    }
                    else   // is_snap_b2 == true and (is_snap_b0 == false and is_snap_b1 == false)
                    {
                        /// VERTEX 2
                        ind_b=tb[2];
                    }
                }

                b_last=ind_b; // OUTPUT

                /// Register the creation of triangles incident to point indexed by ind_b

                if(ind_b==p1_b)
                {
                    Triangle t_pb1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points,(unsigned int) p1_b,(unsigned int) p0_b));
                    Triangle t_pb2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points,(unsigned int) p0_b, (unsigned int) p2_b));
                    triangles_to_create.push_back(t_pb1); triangles_to_create.push_back(t_pb2);

                }
                else
                {

                    if(ind_b==p2_b)
                    {
                        Triangle t_pb1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points - 1,(unsigned int) p1_b,(unsigned int) p0_b));
                        Triangle t_pb2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points - 1,(unsigned int) p0_b, (unsigned int) p2_b));
                        triangles_to_create.push_back(t_pb1); triangles_to_create.push_back(t_pb2);

                    }
                    else
                    {

                        Triangle t_pb1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points - 1,(unsigned int) p1_b,(unsigned int) ind_b));
                        Triangle t_pb2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points,(unsigned int) ind_b, (unsigned int)p2_b));
                        triangles_to_create.push_back(t_pb1); triangles_to_create.push_back(t_pb2);
                    }
                }

                acc_nb_triangles+=2;
            }

        }


        // POINT SEPARATING

        if(!is_first_cut)
        {

            sofa::helper::vector< unsigned int > bb_first_ancestors;
            bb_first_ancestors.push_back(x_i1); bb_first_ancestors.push_back(x_i2); bb_first_ancestors.push_back(x_i3);
            sofa::helper::vector< double > bb_baryCoefs = topology->getTriangleSetGeometryAlgorithms()->compute3PointsBarycoefs((const Vec<3,double> &) a_new, x_i1, x_i2, x_i3);

            // Add point B1
            p_ancestors.push_back(bb_first_ancestors);
            p_baryCoefs.push_back(bb_baryCoefs);
            acc_nb_points=acc_nb_points+1;
            unsigned int B1 = acc_nb_points;

            // Add point B2
            p_ancestors.push_back(bb_first_ancestors);
            p_baryCoefs.push_back(bb_baryCoefs);
            acc_nb_points=acc_nb_points+1;
            unsigned int B2 = acc_nb_points;

            Triangle T1 = Triangle(helper::make_array<unsigned int>(B1, x_p1, x_i1));
            Triangle T1_to = Triangle(helper::make_array<unsigned int>(B1, x_i1_to, x_p1_to));

            Triangle T2 = Triangle(helper::make_array<unsigned int>(B2, x_i2, x_p2));
            Triangle T2_to = Triangle(helper::make_array<unsigned int>(B2, x_p2_to, x_i2_to));

            Triangle Ti1 = Triangle(helper::make_array<unsigned int>(B1, x_i1, x_i1_to));
            Triangle Ti2 = Triangle(helper::make_array<unsigned int>(B2, x_i2_to, x_i2));

            Triangle Tp1 = Triangle(helper::make_array<unsigned int>(B1, x_p1, x_p1_to));
            Triangle Tp2 = Triangle(helper::make_array<unsigned int>(B2, x_p2_to, x_p2));

            Triangle T1_13 = Triangle(helper::make_array<unsigned int>(B1, x_i1, x_i3));
            Triangle T1_23 = Triangle(helper::make_array<unsigned int>(B1, x_i3, x_i2));
            Triangle T2_13 = Triangle(helper::make_array<unsigned int>(B2, x_i1, x_i3));
            Triangle T2_23 = Triangle(helper::make_array<unsigned int>(B2, x_i3, x_i2));

            if(x_i1 == x_i1_to)
            {

                triangles_to_create.push_back(T1);
                triangles_to_create.push_back(T1_to);
                triangles_to_create.push_back(T2);
                triangles_to_create.push_back(T2_to);
                triangles_to_create.push_back(Ti2);

            }
            else
            {

                if(x_i2 == x_i2_to)
                {

                    triangles_to_create.push_back(T1);
                    triangles_to_create.push_back(T1_to);
                    triangles_to_create.push_back(T2);
                    triangles_to_create.push_back(T2_to);
                    triangles_to_create.push_back(Ti1);
                }
                else   // (x_i1 == x_i2_to) or (x_i2 == x_i1_to)
                {

                    if(x_i1 == x_i2_to)
                    {

                        triangles_to_create.push_back(Tp1);
                        triangles_to_create.push_back(T2_13);
                        triangles_to_create.push_back(T2_23);
                        triangles_to_create.push_back(T2);
                        triangles_to_create.push_back(T2_to);

                    }
                    else   // x_i2 == x_i1_to
                    {

                        triangles_to_create.push_back(Tp2);
                        triangles_to_create.push_back(T1_13);
                        triangles_to_create.push_back(T1_23);
                        triangles_to_create.push_back(T1);
                        triangles_to_create.push_back(T1_to);

                    }
                }
            }

            acc_nb_triangles = acc_nb_triangles + 5;
        }


        // Create all the points registered to be created
        modifier->addPointsProcess((const unsigned int) acc_nb_points - nb_points, p_ancestors, p_baryCoefs);

        // Warn for the creation of all the points registered to be created
        modifier->addPointsWarning((const unsigned int) acc_nb_points - nb_points, p_ancestors, p_baryCoefs);

        // Create all the triangles registered to be created
        modifier->addTrianglesProcess((const sofa::helper::vector< Triangle > &) triangles_to_create) ; // WARNING called after the creation process by the method "addTrianglesProcess"

        // Propagate the topological changes *** not necessary
        //topology->propagateTopologicalChanges();

        // Remove all the triangles registered to be removed
        removeTriangles(triangles_to_remove, true, true); // (WARNING then PROPAGATION) called before the removal process by the method "removeTriangles"

        // Propagate the topological changes *** not necessary
        //topology->propagateTopologicalChanges();

    }

    return is_intersected && (elem_size>0);
}

// Removes triangles along the list of points (ind_edge,coord) intersected by the segment from point a to point b and the triangular mesh

template<class DataTypes>
void TriangleSetTopologyAlgorithms< DataTypes >::RemoveAlongTrianglesList(const Vec<3,double>& a, const Vec<3,double>& b, const unsigned int ind_ta, const unsigned int ind_tb)
{

    TriangleSetTopology<DataTypes> *topology = dynamic_cast<TriangleSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);

    sofa::helper::vector< unsigned int > triangles_list_init;
    sofa::helper::vector< unsigned int > &triangles_list = triangles_list_init;

    sofa::helper::vector< sofa::helper::vector< unsigned int> > indices_list_init;
    sofa::helper::vector< sofa::helper::vector< unsigned int> > &indices_list = indices_list_init;

    sofa::helper::vector< double > coords_list_init;
    sofa::helper::vector< double >& coords_list=coords_list_init;

    bool is_intersected=false;

    unsigned int ind_tb_final_init;
    unsigned int& ind_tb_final=ind_tb_final_init;

    bool is_on_boundary_init=false;
    bool &is_on_boundary=is_on_boundary_init;

    ind_tb_final=ind_tb;
    is_intersected = topology->getTriangleSetGeometryAlgorithms()->computeIntersectedPointsList(a, b, ind_ta, ind_tb_final, triangles_list, indices_list, coords_list, is_on_boundary);

    if(is_intersected)
    {

        sofa::helper::vector< unsigned int > triangles;

        for (unsigned int i=0; i<triangles_list.size(); i++)
        {

            triangles.push_back(triangles_list[i]);

        }
        removeTriangles(triangles, true, true);

    }

}


// Incises along the list of points (ind_edge,coord) intersected by the sequence of input segments (list of input points) and the triangular mesh

template<class DataTypes>
void TriangleSetTopologyAlgorithms< DataTypes >::InciseAlongLinesList(const sofa::helper::vector< Vec<3,double> >& input_points, const sofa::helper::vector< unsigned int > &input_triangles)
{

    // HYP : input_points.size() == input_triangles.size()

    // Access the topology
    TriangleSetTopology<DataTypes> *topology = dynamic_cast<TriangleSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast< TriangleSetTopologyContainer* >(topology->getTopologyContainer());

    TriangleSetTopologyModifier< DataTypes >* modifier  = static_cast< TriangleSetTopologyModifier< DataTypes >* >(topology->getTopologyModifier());
    assert(modifier != 0);

    unsigned int points_size = input_points.size();

    // Initialization for INTERSECTION method
    sofa::helper::vector< unsigned int > triangles_list_init;
    sofa::helper::vector< unsigned int > &triangles_list = triangles_list_init;

    sofa::helper::vector< sofa::helper::vector<unsigned int> > indices_list_init;
    sofa::helper::vector< sofa::helper::vector<unsigned int> > &indices_list = indices_list_init;

    sofa::helper::vector< double > coords_list_init;
    sofa::helper::vector< double >& coords_list=coords_list_init;

    unsigned int ind_tb_final_init;
    unsigned int &ind_tb_final=ind_tb_final_init;

    bool is_on_boundary_init=false;
    bool &is_on_boundary=is_on_boundary_init;

    bool is_validated=true;
    unsigned int j = 0;

    const Vec<3,double> a = input_points[0];
    unsigned int ind_ta = input_triangles[0];

    while(is_validated &&  j <  points_size - 1)
    {

        const Vec<3,double> pa = input_points[j];
        const Vec<3,double> pb = input_points[j+1];
        unsigned int ind_tpa = input_triangles[j];
        unsigned int ind_tpb = input_triangles[j+1];

        bool is_distinct = (pa!=pb && ind_tpa!=ind_tpb);

        if(is_distinct)
        {
            // Call the method "computeIntersectedPointsList" to get the list of points (ind_edge,coord) intersected by the segment from point a to point b and the triangular mesh

            ind_tb_final=ind_tpb;
            bool is_intersected = topology->getTriangleSetGeometryAlgorithms()->computeIntersectedPointsList(pa, pb, ind_tpa, ind_tb_final, triangles_list, indices_list, coords_list, is_on_boundary);
            is_validated=is_intersected;
        }
        else
        {
            is_validated=false;
        }

        j++;
    }

    const Vec<3,double> b = input_points[j];
    unsigned int ind_tb = input_triangles[j];

    const Triangle &ta=container->getTriangle(ind_ta);
    const Triangle &tb=container->getTriangle(ind_tb);

    const typename DataTypes::VecCoord& vect_c = *topology->getDOF()->getX();
    unsigned int nb_points =  vect_c.size() -1;

    const sofa::helper::vector<Triangle> &vect_t=container->getTriangleArray();
    unsigned int nb_triangles =  vect_t.size() -1;

    // Variables to accumulate the number of elements registered to be created (so as to remember their indices)
    unsigned int acc_nb_points=nb_points;
    unsigned int acc_nb_triangles=nb_triangles;

    // Variables to accumulate the elements registered to be created or to be removed
    sofa::helper::vector< sofa::helper::vector< unsigned int > > p_ancestors;
    sofa::helper::vector< sofa::helper::vector< double > > p_baryCoefs;
    sofa::helper::vector< Triangle > triangles_to_create;
    sofa::helper::vector< unsigned int > triangles_to_remove;

    unsigned int ta_to_remove;
    unsigned int tb_to_remove;

    // Initialization for SNAPPING method

    bool is_snap_a0_init=false; bool is_snap_a1_init=false; bool is_snap_a2_init=false;
    bool& is_snap_a0=is_snap_a0_init;
    bool& is_snap_a1=is_snap_a1_init;
    bool& is_snap_a2=is_snap_a2_init;

    bool is_snap_b0_init=false; bool is_snap_b1_init=false; bool is_snap_b2_init=false;
    bool& is_snap_b0=is_snap_b0_init;
    bool& is_snap_b1=is_snap_b1_init;
    bool& is_snap_b2=is_snap_b2_init;

    double epsilon = 0.2; // INFO : epsilon is a threshold in [0,1] to control the snapping of the extremities to the closest vertex

    sofa::helper::vector< double > a_baryCoefs = topology->getTriangleSetGeometryAlgorithms()->computeTriangleBarycoefs((const Vec<3,double> &) a, ind_ta);
    snapping_test_triangle(epsilon, a_baryCoefs[0], a_baryCoefs[1], a_baryCoefs[2], is_snap_a0, is_snap_a1, is_snap_a2);
    double is_snapping_a = is_snap_a0 || is_snap_a1 || is_snap_a2;

    sofa::helper::vector< double > b_baryCoefs = topology->getTriangleSetGeometryAlgorithms()->computeTriangleBarycoefs((const Vec<3,double> &) b, ind_tb);
    snapping_test_triangle(epsilon, b_baryCoefs[0], b_baryCoefs[1], b_baryCoefs[2], is_snap_b0, is_snap_b1, is_snap_b2);
    double is_snapping_b = is_snap_b0 || is_snap_b1 || is_snap_b2;

    /*
      if(is_snapping_a){
      std::cout << "INFO_print : is_snapping_a" <<  std::endl;
      }
      if(is_snapping_b){
      std::cout << "INFO_print : is_snapping_b" <<  std::endl;
      }
    */

    if(is_validated)  // intersection successfull
    {

        /// force the creation of TriangleEdgeShellArray
        container->getTriangleEdgeShellArray();
        /// force the creation of TriangleVertexShellArray
        container->getTriangleVertexShellArray();

        // Initialization for the indices of the previous intersected edge
        unsigned int p1_prev=0;
        unsigned int p2_prev=0;

        unsigned int p1_a=indices_list[0][0];
        unsigned int p2_a=indices_list[0][1];
        unsigned int p1_b=indices_list[indices_list.size()-1][0];
        unsigned int p2_b=indices_list[indices_list.size()-1][1];

        // Plan to remove triangles indexed by ind_ta and ind_tb
        triangles_to_remove.push_back(ind_ta); triangles_to_remove.push_back(ind_tb);

        // Treatment of particular case for first extremity a

        sofa::helper::vector< unsigned int > a_first_ancestors;
        sofa::helper::vector< double > a_first_baryCoefs;

        if(!is_snapping_a)
        {

            /// Register the creation of point a

            a_first_ancestors.push_back(ta[0]); a_first_ancestors.push_back(ta[1]); a_first_ancestors.push_back(ta[2]);
            p_ancestors.push_back(a_first_ancestors);
            p_baryCoefs.push_back(a_baryCoefs);

            acc_nb_points=acc_nb_points+1;

            /// Register the creation of triangles incident to point a

            unsigned int ind_a =  acc_nb_points; // last point registered to be created

            sofa::helper::vector< Triangle > a_triangles;
            Triangle t_a01 = Triangle(helper::make_array<unsigned int>((unsigned int)ind_a,(unsigned int)ta[0],(unsigned int) ta[1]));
            Triangle t_a12 = Triangle(helper::make_array<unsigned int>((unsigned int)ind_a,(unsigned int)ta[1],(unsigned int) ta[2]));
            Triangle t_a20 = Triangle(helper::make_array<unsigned int>((unsigned int)ind_a,(unsigned int)ta[2],(unsigned int) ta[0]));
            triangles_to_create.push_back(t_a01); triangles_to_create.push_back(t_a12); triangles_to_create.push_back(t_a20);

            acc_nb_triangles=acc_nb_triangles+3;

            /// Register the removal of triangles incident to point a

            if(ta[0]!=p1_a && ta[0]!=p2_a)
            {
                ta_to_remove=acc_nb_triangles-1;
            }
            else
            {
                if(ta[1]!=p1_a && ta[1]!=p2_a)
                {
                    ta_to_remove=acc_nb_triangles;
                }
                else   // (ta[2]!=p1_a && ta[2]!=p2_a)
                {
                    ta_to_remove=acc_nb_triangles-2;
                }
            }
            triangles_to_remove.push_back(ta_to_remove);

            Triangle t_pa1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 1,(unsigned int) ind_a,(unsigned int) p1_a));
            Triangle t_pa2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 2,(unsigned int) p2_a, (unsigned int)ind_a));
            triangles_to_create.push_back(t_pa1); triangles_to_create.push_back(t_pa2);

            acc_nb_triangles=acc_nb_triangles+2;

        }
        else   // snapping a to the vertex indexed by ind_a, which is the closest to point a
        {

            // localize the closest vertex

            unsigned int ind_a;
            unsigned int p0_a;

            if(ta[0]!=p1_a && ta[0]!=p2_a)
            {
                p0_a=ta[0];
            }
            else
            {
                if(ta[1]!=p1_a && ta[1]!=p2_a)
                {
                    p0_a=ta[1];
                }
                else  // ta[2]!=p1_a && ta[2]!=p2_a
                {
                    p0_a=ta[2];
                }
            }

            if(is_snap_a0)  // is_snap_a1 == false and is_snap_a2 == false
            {
                /// VERTEX 0
                ind_a=ta[0];
            }
            else
            {
                if(is_snap_a1)  // is_snap_a0 == false and is_snap_a2 == false
                {
                    /// VERTEX 1
                    ind_a=ta[1];
                }
                else   // is_snap_a2 == true and (is_snap_a0 == false and is_snap_a1 == false)
                {
                    /// VERTEX 2
                    ind_a=ta[2];
                }
            }

            /// Register the creation of triangles incident to point indexed by ind_a

            if(ind_a==p1_a)
            {
                Triangle t_pa1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 2,(unsigned int) p0_a,(unsigned int) p1_a));
                Triangle t_pa2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 2,(unsigned int) p2_a, (unsigned int) p0_a));
                triangles_to_create.push_back(t_pa1); triangles_to_create.push_back(t_pa2);

            }
            else
            {

                if(ind_a==p2_a)
                {
                    Triangle t_pa1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 1,(unsigned int) p0_a,(unsigned int) p1_a));
                    Triangle t_pa2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 1,(unsigned int) p2_a, (unsigned int) p0_a));
                    triangles_to_create.push_back(t_pa1); triangles_to_create.push_back(t_pa2);

                }
                else
                {

                    Triangle t_pa1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 1,(unsigned int) ind_a,(unsigned int) p1_a));
                    Triangle t_pa2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points + 2,(unsigned int) p2_a, (unsigned int)ind_a));
                    triangles_to_create.push_back(t_pa1); triangles_to_create.push_back(t_pa2);
                }
            }

            acc_nb_triangles+=2;

        }

        // Traverse the loop of interected edges

        for (unsigned int i=0; i<indices_list.size(); i++)
        {

            /// Register the creation of the two points (say current "duplicated points") localized on the current interected edge

            unsigned int p1 = indices_list[i][0];
            unsigned int p2 = indices_list[i][1];

            sofa::helper::vector< unsigned int > p_first_ancestors;
            p_first_ancestors.push_back(p1); p_first_ancestors.push_back(p2);
            p_ancestors.push_back(p_first_ancestors); p_ancestors.push_back(p_first_ancestors);

            sofa::helper::vector< double > p_first_baryCoefs;
            p_first_baryCoefs.push_back(1.0 - coords_list[i]); p_first_baryCoefs.push_back(coords_list[i]);
            p_baryCoefs.push_back(p_first_baryCoefs); p_baryCoefs.push_back(p_first_baryCoefs);

            acc_nb_points=acc_nb_points+2;

            if(i>0)  // not to treat particular case of first extremitiy
            {

                // SNAPPING TEST

                double gamma = 0.3;
                bool is_snap_p1;
                bool is_snap_p2;

                snapping_test_edge(gamma, 1.0 - coords_list[i], coords_list[i], is_snap_p1, is_snap_p2);
                double is_snapping_p = is_snap_p1 || is_snap_p2;

                unsigned int ind_p;

                if(is_snapping_p && i<indices_list.size()-1)  // not to treat particular case of last extremitiy
                {

                    if(is_snap_p1)
                    {
                        /// VERTEX 0
                        ind_p=p1;
                    }
                    else   // is_snap_p2 == true
                    {
                        /// VERTEX 1
                        ind_p=p2;
                    }

                    //std::cout << "INFO_print : is_snapping_p, i = " << i << " on vertex " << ind_p <<  std::endl;

                    sofa::helper::vector< unsigned int > triangles_list_1_init;
                    sofa::helper::vector< unsigned int > &triangles_list_1 = triangles_list_1_init;

                    sofa::helper::vector< unsigned int > triangles_list_2_init;
                    sofa::helper::vector< unsigned int > &triangles_list_2 = triangles_list_2_init;

                    //std::cout << "INFO_print : DO Prepare_VertexDuplication " <<  std::endl;
                    topology->getTriangleSetGeometryAlgorithms()->Prepare_VertexDuplication(ind_p, triangles_list[i], triangles_list[i+1], indices_list[i-1], coords_list[i-1], indices_list[i+1], coords_list[i+1], triangles_list_1, triangles_list_2);
                    //std::cout << "INFO_print : DONE Prepare_VertexDuplication " <<  std::endl;

                    //std::cout << "INFO_print : triangles_list_1.size() = " << triangles_list_1.size() <<  std::endl;
                    //for (unsigned int k=0;k<triangles_list_1.size();k++){
                    //		std::cout << "INFO_print : triangles_list_1 number " << k << " = " << triangles_list_1[k] <<  std::endl;
                    //}

                    //std::cout << "INFO_print : triangles_list_2.size() = " << triangles_list_2.size() <<  std::endl;
                    //for (unsigned int k=0;k<triangles_list_2.size();k++){
                    //		std::cout << "INFO_print : triangles_list_2 number " << k << " = " << triangles_list_2[k] <<  std::endl;
                    //}

                }

                /// Register the removal of the current triangle

                triangles_to_remove.push_back(triangles_list[i]);

                /// Register the creation of triangles incident to the current "duplicated points" and to the previous "duplicated points"

                unsigned int p1_created=acc_nb_points - 3;
                unsigned int p2_created=acc_nb_points - 2;

                unsigned int p1_to_create=acc_nb_points - 1;
                unsigned int p2_to_create=acc_nb_points;

                unsigned int p0_t = container->getTriangle(triangles_list[i])[0];
                unsigned int p1_t = container->getTriangle(triangles_list[i])[1];
                unsigned int p2_t = container->getTriangle(triangles_list[i])[2];

                Triangle t_p1 = Triangle(helper::make_array<unsigned int>((unsigned int) p1_created,(unsigned int) p1_prev,(unsigned int) p1_to_create));
                Triangle t_p2 = Triangle(helper::make_array<unsigned int>((unsigned int) p2_created,(unsigned int) p2_to_create,(unsigned int) p2_prev));

                Triangle t_p3;

                if(p0_t!=p1_prev && p0_t!=p2_prev)
                {

                    if(p0_t==p1)
                    {

                        t_p3=Triangle(helper::make_array<unsigned int>((unsigned int) p0_t,(unsigned int) p1_to_create,(unsigned int) p1_prev));

                    }
                    else   // p0_t==p2
                    {

                        t_p3=Triangle(helper::make_array<unsigned int>((unsigned int) p0_t,(unsigned int) p2_prev,(unsigned int) p2_to_create));

                    }

                }
                else
                {
                    if(p1_t!=p1_prev && p1_t!=p2_prev)
                    {

                        if(p1_t==p1)
                        {

                            t_p3=Triangle(helper::make_array<unsigned int>((unsigned int) p1_t,(unsigned int) p1_to_create,(unsigned int) p1_prev));

                        }
                        else   // p1_t==p2
                        {

                            t_p3=Triangle(helper::make_array<unsigned int>((unsigned int) p1_t,(unsigned int) p2_prev,(unsigned int) p2_to_create));

                        }

                    }
                    else   // (p2_t!=p1_prev && p2_t!=p2_prev)
                    {

                        if(p2_t==p1)
                        {

                            t_p3=Triangle(helper::make_array<unsigned int>((unsigned int) p2_t,(unsigned int) p1_to_create,(unsigned int) p1_prev));

                        }
                        else   // p2_t==p2
                        {

                            t_p3=Triangle(helper::make_array<unsigned int>((unsigned int) p2_t,(unsigned int) p2_prev,(unsigned int) p2_to_create));

                        }
                    }
                }

                triangles_to_create.push_back(t_p1); triangles_to_create.push_back(t_p2); triangles_to_create.push_back(t_p3);

                acc_nb_triangles=acc_nb_triangles+3;

            }

            // Update the previous "duplicated points"
            p1_prev=p1;
            p2_prev=p2;

        }

        // Treatment of particular case for second extremity b
        sofa::helper::vector< unsigned int > b_first_ancestors;
        sofa::helper::vector< double > b_first_baryCoefs;

        if(!is_snapping_b)
        {

            /// Register the creation of point b

            b_first_ancestors.push_back(tb[0]); b_first_ancestors.push_back(tb[1]); b_first_ancestors.push_back(tb[2]);
            p_ancestors.push_back(b_first_ancestors);
            p_baryCoefs.push_back(b_baryCoefs);

            acc_nb_points=acc_nb_points+1;

            /// Register the creation of triangles incident to point b

            unsigned int ind_b =  acc_nb_points; // last point registered to be created

            sofa::helper::vector< Triangle > b_triangles;
            Triangle t_b01 = Triangle(helper::make_array<unsigned int>((unsigned int)ind_b,(unsigned int)tb[0],(unsigned int) tb[1]));
            Triangle t_b12 = Triangle(helper::make_array<unsigned int>((unsigned int)ind_b,(unsigned int)tb[1],(unsigned int) tb[2]));
            Triangle t_b20 = Triangle(helper::make_array<unsigned int>((unsigned int)ind_b,(unsigned int)tb[2],(unsigned int) tb[0]));
            triangles_to_create.push_back(t_b01); triangles_to_create.push_back(t_b12); triangles_to_create.push_back(t_b20);

            acc_nb_triangles=acc_nb_triangles+3;

            /// Register the removal of triangles incident to point b

            if(tb[0]!=p1_b && tb[0]!=p2_b)
            {
                tb_to_remove=acc_nb_triangles-1;
            }
            else
            {
                if(tb[1]!=p1_b && tb[1]!=p2_b)
                {
                    tb_to_remove=acc_nb_triangles;
                }
                else   // (tb[2]!=p1_b && tb[2]!=p2_b)
                {
                    tb_to_remove=acc_nb_triangles-2;
                }
            }
            triangles_to_remove.push_back(tb_to_remove);

            Triangle t_pb1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points - 2,(unsigned int) p1_b,(unsigned int)ind_b));
            Triangle t_pb2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points - 1,(unsigned int)ind_b,(unsigned int) p2_b));
            triangles_to_create.push_back(t_pb1); triangles_to_create.push_back(t_pb2);

            acc_nb_triangles=acc_nb_triangles+2;

        }
        else   // snapping b to the vertex indexed by ind_b, which is the closest to point b
        {

            // localize the closest vertex
            unsigned int ind_b;
            unsigned int p0_b;

            if(tb[0]!=p1_b && tb[0]!=p2_b)
            {
                p0_b=tb[0];
            }
            else
            {
                if(tb[1]!=p1_b && tb[1]!=p2_b)
                {
                    p0_b=tb[1];
                }
                else  // tb[2]!=p1_b && tb[2]!=p2_b
                {
                    p0_b=tb[2];
                }
            }

            if(is_snap_b0)  // is_snap_b1 == false and is_snap_b2 == false
            {
                /// VERTEX 0
                ind_b=tb[0];
            }
            else
            {
                if(is_snap_b1)  // is_snap_b0 == false and is_snap_b2 == false
                {
                    /// VERTEX 1
                    ind_b=tb[1];
                }
                else   // is_snap_b2 == true and (is_snap_b0 == false and is_snap_b1 == false)
                {
                    /// VERTEX 2
                    ind_b=tb[2];
                }
            }

            /// Register the creation of triangles incident to point indexed by ind_b

            if(ind_b==p1_b)
            {
                Triangle t_pb1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points,(unsigned int) p1_b,(unsigned int) p0_b));
                Triangle t_pb2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points,(unsigned int) p0_b, (unsigned int) p2_b));
                triangles_to_create.push_back(t_pb1); triangles_to_create.push_back(t_pb2);

            }
            else
            {

                if(ind_b==p2_b)
                {
                    Triangle t_pb1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points - 1,(unsigned int) p1_b,(unsigned int) p0_b));
                    Triangle t_pb2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points - 1,(unsigned int) p0_b, (unsigned int) p2_b));
                    triangles_to_create.push_back(t_pb1); triangles_to_create.push_back(t_pb2);

                }
                else
                {

                    Triangle t_pb1 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points - 1,(unsigned int) p1_b,(unsigned int) ind_b));
                    Triangle t_pb2 = Triangle(helper::make_array<unsigned int>((unsigned int) acc_nb_points,(unsigned int) ind_b, (unsigned int)p2_b));
                    triangles_to_create.push_back(t_pb1); triangles_to_create.push_back(t_pb2);
                }
            }

            acc_nb_triangles+=2;
        }

        // Create all the points registered to be created
        modifier->addPointsProcess((const unsigned int) acc_nb_points - nb_points, p_ancestors, p_baryCoefs);

        // Warn for the creation of all the points registered to be created
        modifier->addPointsWarning((const unsigned int) acc_nb_points - nb_points, p_ancestors, p_baryCoefs);

        // Create all the triangles registered to be created
        modifier->addTrianglesProcess((const sofa::helper::vector< Triangle > &) triangles_to_create) ; // WARNING called after the creation process by the method "addTrianglesProcess"

        // Propagate the topological changes *** not necessary
        //topology->propagateTopologicalChanges();

        // Remove all the triangles registered to be removed
        removeTriangles(triangles_to_remove, true, true); // (WARNING then PROPAGATION) called before the removal process by the method "removeTriangles"

        // Propagate the topological changes *** not necessary
        //topology->propagateTopologicalChanges();
    }
}

// Duplicate the given edge. Only works of at least one of its points is adjacent to a border.
template<class DataTypes>
int TriangleSetTopologyAlgorithms<DataTypes>::InciseAlongEdge(unsigned int ind_edge)
{

    // Access the topology
    TriangleSetTopology<DataTypes> *topology = dynamic_cast<TriangleSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast< TriangleSetTopologyContainer* >(topology->getTopologyContainer());

    TriangleSetTopologyModifier< DataTypes >* modifier  = static_cast< TriangleSetTopologyModifier< DataTypes >* >(topology->getTopologyModifier());
    assert(modifier != 0);

    const Edge & edge0=container->getEdge(ind_edge);
    unsigned ind_pa = edge0.first;
    unsigned ind_pb = edge0.second;

    const helper::vector<unsigned>& triangles0 = container->getTriangleEdgeShell(ind_edge);
    if (triangles0.size() != 2)
    {
        std::cerr << "InciseAlongEdge: ERROR edge "<<ind_edge<<" is not attached to 2 triangles." << std::endl;
        return -1;
    }
    // choose one triangle
    unsigned ind_tri0 = triangles0[0];

    unsigned ind_tria = ind_tri0;
    unsigned ind_trib = ind_tri0;
    unsigned ind_edgea = ind_edge;
    unsigned ind_edgeb = ind_edge;

    helper::vector<unsigned> list_tria;
    helper::vector<unsigned> list_trib;

    for (;;)
    {
        const TriangleEdges& te = container->getTriangleEdge(ind_tria);
        // find the edge adjacent to a that is not ind_edgea
        int j=0;
        for (j=0; j<3; ++j)
        {
            if (te[j] != ind_edgea && (container->getEdge(te[j]).first == ind_pa || container->getEdge(te[j]).second == ind_pa))
                break;
        }
        if (j == 3)
        {
            std::cerr << "InciseAlongEdge: ERROR in triangle "<<ind_tria<<std::endl;
            return -1;
        }
        ind_edgea = te[j];
        if (ind_edgea == ind_edge) break; // full loop
        const helper::vector<unsigned>& tes = container->getTriangleEdgeShell(ind_edgea);
        if(tes.size() < 2) break; // border edge
        if (tes[0] == ind_tria)
            ind_tria = tes[1];
        else
            ind_tria = tes[0];
        list_tria.push_back(ind_tria);
    }

    for (;;)
    {
        const TriangleEdges& te = container->getTriangleEdge(ind_trib);
        // find the edge adjacent to b that is not ind_edgeb
        int j=0;
        for (j=0; j<3; ++j)
        {
            if (te[j] != ind_edgeb && (container->getEdge(te[j]).first == ind_pb || container->getEdge(te[j]).second == ind_pb))
                break;
        }
        if (j == 3)
        {
            std::cerr << "InciseAlongEdge: ERROR in triangle "<<ind_trib<<std::endl;
            return -1;
        }
        ind_edgeb = te[j];
        if (ind_edgeb == ind_edge) break; // full loop
        const helper::vector<unsigned>& tes = container->getTriangleEdgeShell(ind_edgeb);
        if(tes.size() < 2) break; // border edge
        if (tes[0] == ind_trib)
            ind_trib = tes[1];
        else
            ind_trib = tes[0];
        list_trib.push_back(ind_trib);
    }

    bool pa_is_on_border = (ind_edgea != ind_edge);
    bool pb_is_on_border = (ind_edgeb != ind_edge);

    if (!pa_is_on_border && !pb_is_on_border)
    {
        std::cerr << "InciseAlongEdge: ERROR edge "<<ind_edge<<" is not on border." << std::endl;
        return -1;
    }

    // now we can split the edge

    /// force the creation of TriangleEdgeShellArray
    container->getTriangleEdgeShellArray();
    /// force the creation of TriangleVertexShellArray
    container->getTriangleVertexShellArray();

    const typename DataTypes::VecCoord& vect_c = *topology->getDOF()->getX();
    unsigned int nb_points =  vect_c.size();

    // Variables to accumulate the number of elements registered to be created (so as to remember their indices)
    unsigned int acc_nb_points=nb_points;

    // Variables to accumulate the elements registered to be created or to be removed
    sofa::helper::vector< sofa::helper::vector< unsigned int > > p_ancestors;
    sofa::helper::vector< sofa::helper::vector< double > > p_baryCoefs;
    sofa::helper::vector< Triangle > triangles_to_create;
    sofa::helper::vector< unsigned int > triangles_to_remove;

    sofa::helper::vector<double> defaultCoefs; defaultCoefs.push_back(1.0);

    unsigned new_pa, new_pb;

    if (pa_is_on_border)
    {
        sofa::helper::vector<unsigned int> ancestors;
        new_pa = acc_nb_points++;
        ancestors.push_back(ind_pa);
        p_ancestors.push_back(ancestors);
        p_baryCoefs.push_back(defaultCoefs);
    }
    else
        new_pa = ind_pa;

    sofa::helper::vector<unsigned int> ancestors(1);

    if (pb_is_on_border)
    {
        new_pb = acc_nb_points++;
        ancestors[0] = ind_pb;
        p_ancestors.push_back(ancestors);
        p_baryCoefs.push_back(defaultCoefs);
    }
    else
        new_pb = ind_pb;

    // we need to recreate at least tri0
    Triangle new_tri0 = container->getTriangle(ind_tri0);
    for (unsigned i=0; i<3; i++)
        if (new_tri0[i] == ind_pa) new_tri0[i] = new_pa;
        else if (new_tri0[i] == ind_pb) new_tri0[i] = new_pb;
    triangles_to_remove.push_back(ind_tri0);
    ancestors[0] = ind_tri0;
    triangles_to_create.push_back(new_tri0);

    // recreate list_tria iff pa is new
    if (new_pa != ind_pa)
    {
        for (unsigned j=0; j<list_tria.size(); j++)
        {
            unsigned ind_tri = list_tria[j];
            Triangle new_tri = container->getTriangle(ind_tri);
            for (unsigned i=0; i<3; i++)
                if (new_tri[i] == ind_pa) new_tri[i] = new_pa;
            triangles_to_remove.push_back(ind_tri);
            ancestors[0] = ind_tri;
            triangles_to_create.push_back(new_tri);
        }
    }

    // recreate list_trib iff pb is new
    if (new_pb != ind_pb)
    {
        for (unsigned j=0; j<list_trib.size(); j++)
        {
            unsigned ind_tri = list_trib[j];
            Triangle new_tri = container->getTriangle(ind_tri);
            for (unsigned i=0; i<3; i++)
                if (new_tri[i] == ind_pb) new_tri[i] = new_pb;
            triangles_to_remove.push_back(ind_tri);
            ancestors[0] = ind_tri;
            triangles_to_create.push_back(new_tri);
        }
    }

    // Create all the points registered to be created
    modifier->addPointsProcess((const unsigned int) acc_nb_points - nb_points, p_ancestors, p_baryCoefs);

    // Warn for the creation of all the points registered to be created
    modifier->addPointsWarning((const unsigned int) acc_nb_points - nb_points, p_ancestors, p_baryCoefs);

    // Create all the triangles registered to be created
    modifier->addTrianglesProcess((const sofa::helper::vector< Triangle > &) triangles_to_create) ; // WARNING called after the creation process by the method "addTrianglesProcess"

    // Propagate the topological changes *** not necessary
    //topology->propagateTopologicalChanges();

    // Remove all the triangles registered to be removed
    removeTriangles(triangles_to_remove, true, true); // (WARNING then PROPAGATION) called before the removal process by the method "removeTriangles"

    // Propagate the topological changes *** not necessary
    //topology->propagateTopologicalChanges();

    return (pb_is_on_border?1:0)+(pa_is_on_border?1:0); // todo: get new edge indice
}

/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////TriangleSetGeometryAlgorithms//////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test for REAL if a point p is in a triangle indexed by (a,b,c)

template<class Real>
bool is_point_in_triangle(const Vec<3,Real>& p, const Vec<3,Real>& a, const Vec<3,Real>& b, const Vec<3,Real>& c)
{

    Vec<3,Real> ptest = p;

    Vec<3,Real> p0 = a;
    Vec<3,Real> p1 = b;
    Vec<3,Real> p2 = c;

    Vec<3,Real> v_normal = (p2-p0).cross(p1-p0);

    Real norm_v_normal = v_normal*(v_normal);
    if(norm_v_normal != 0.0)
    {

        //v_normal/=norm_v_normal;

        if((ptest-p0)*(v_normal)==0.0)  // p is in the plane defined by the triangle (p0,p1,p2)
        {

            Vec<3,Real> n_01 = (p1-p0).cross(v_normal);
            Vec<3,Real> n_12 = (p2-p1).cross(v_normal);
            Vec<3,Real> n_20 = (p0-p2).cross(v_normal);

            return (((ptest-p0)*(n_01) >= 0.0) && ((ptest-p1)*(n_12) >= 0.0) && ((ptest-p2)*(n_20) >= 0.0));

        }
        else   // p is not in the plane defined by the triangle (p0,p1,p2)
        {
            //std::cout << "INFO_print : p is not in the plane defined by the triangle (p0,p1,p2)" << std::endl;
            return false;
        }

    }
    else   // triangle is flat
    {
        //std::cout << "INFO_print : triangle is flat" << std::endl;
        return false;
    }
}

/// Test if a point p is in the right halfplane

template<class Real>
bool is_point_in_halfplane(const Vec<3,Real>& p, unsigned int e0, unsigned int e1, const Vec<3,Real>& a, const Vec<3,Real>& b, const Vec<3,Real>& c, unsigned int ind_p0, unsigned int ind_p1, unsigned int ind_p2)
{

    Vec<3,Real> ptest = p;

    Vec<3,Real> p0 = a;
    Vec<3,Real> p1 = b;
    Vec<3,Real> p2 = c;

    Vec<3,Real> v_normal = (p2-p0).cross(p1-p0);

    Real norm_v_normal = (v_normal)*(v_normal);
    if(norm_v_normal != 0.0)
    {

        //v_normal/=norm_v_normal;

        if(ind_p0==e0 || ind_p0==e1)
        {
            Vec<3,Real> n_12 = (p2-p1).cross(v_normal);
            return ((ptest-p1)*(n_12) >= 0.0);

        }
        else
        {
            if(ind_p1==e0 || ind_p1==e1)
            {
                Vec<3,Real> n_20 = (p0-p2).cross(v_normal);
                return ((ptest-p2)*(n_20) >= 0.0);

            }
            else
            {
                if(ind_p2==e0 || ind_p2==e1)
                {
                    Vec<3,Real> n_01 = (p1-p0).cross(v_normal);
                    return ((ptest-p0)*(n_01) >= 0.0);

                }
                else
                {
                    return false; // not expected
                }
            }

        }

    }
    else   // triangle is flat
    {
        //std::cout << "INFO_print : triangle is flat" << std::endl;
        return false;
    }

}

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


void snapping_test_triangle(double epsilon, double alpha0, double alpha1, double alpha2, bool& is_snap_0, bool& is_snap_1, bool& is_snap_2)
{

    is_snap_0=false;
    is_snap_1=false;
    is_snap_2=false;

    if(alpha0>=alpha1 && alpha0>=alpha2)
    {

        is_snap_0=(alpha1+alpha2<epsilon);

    }
    else
    {
        if(alpha1>=alpha0 && alpha1>=alpha2)
        {

            is_snap_1=(alpha0+alpha2<epsilon);

        }
        else   // alpha2>=alpha0 && alpha2>=alpha1
        {

            is_snap_2=(alpha0+alpha1<epsilon);
        }

    }

}

void snapping_test_edge(double epsilon, double alpha0, double alpha1, bool& is_snap_0, bool& is_snap_1)
{

    is_snap_0=false;
    is_snap_1=false;

    if(alpha0>=alpha1)
    {

        is_snap_0=(alpha1<epsilon);

    }
    else   // alpha1>=alpha0
    {

        is_snap_1=(alpha0<epsilon);
    }

}


template< class DataTypes>
typename DataTypes::Real TriangleSetGeometryAlgorithms< DataTypes >::computeTriangleArea( const unsigned int i) const
{
    TriangleSetTopology< DataTypes > *topology = static_cast<TriangleSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast< TriangleSetTopologyContainer* >(topology->getTopologyContainer());
    const Triangle &t=container->getTriangle(i);
    const VecCoord& p = *topology->getDOF()->getX();
    Real area=(Real)(areaProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]])/2.0);
    return area;
}
template< class DataTypes>
typename DataTypes::Real TriangleSetGeometryAlgorithms< DataTypes >::computeRestTriangleArea( const unsigned int i) const
{
    TriangleSetTopology< DataTypes > *topology = static_cast<TriangleSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast< TriangleSetTopologyContainer* >(topology->getTopologyContainer());
    const Triangle &t=container->getTriangle(i);
    const VecCoord& p = *topology->getDOF()->getX0();
    Real area=(Real) (areaProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]])/2.0);
    return area;
}

/// computes the edge length of all edges are store in the array interface
template<class DataTypes>
void TriangleSetGeometryAlgorithms<DataTypes>::computeTriangleArea( BasicArrayInterface<Real> &ai) const
{
    TriangleSetTopology< DataTypes > *topology = static_cast<TriangleSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast< TriangleSetTopologyContainer* >(topology->getTopologyContainer());
    const sofa::helper::vector<Triangle> &ta=container->getTriangleArray();
    const typename DataTypes::VecCoord& p = *topology->getDOF()->getX();
    unsigned int i;
    for (i=0; i<ta.size(); ++i)
    {
        const Triangle &t=ta[i];
        ai[i]=(Real)(areaProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]])/2.0);
    }
}


// Computes the point defined by 2 indices of vertex and 1 barycentric coordinate
template<class DataTypes>
Vec<3,double> TriangleSetGeometryAlgorithms< DataTypes >::computeBaryEdgePoint(sofa::helper::vector< unsigned int>& indices, const double &coord_p)
{

    TriangleSetTopology< DataTypes > *topology = static_cast<TriangleSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
//       TriangleSetTopologyContainer * container = static_cast< TriangleSetTopologyContainer* >(topology->getTopologyContainer());
    const typename DataTypes::VecCoord& vect_c = *topology->getDOF()->getX();

    const typename DataTypes::Coord& c1=vect_c[indices[0]];
    const typename DataTypes::Coord& c2=vect_c[indices[1]];

    Vec<3,Real> p;
    p[0]= (Real) ((1.0-coord_p)*c1[0] + coord_p*c2[0]);
    p[1]= (Real) ((1.0-coord_p)*c1[1] + coord_p*c2[1]);
    p[2]= (Real) ((1.0-coord_p)*c1[2] + coord_p*c2[2]);

    return ((Vec<3,double>) p);

}

// Computes the opposite point to ind_p
template<class DataTypes>
Vec<3,double> TriangleSetGeometryAlgorithms< DataTypes >::getOppositePoint(unsigned int ind_p, sofa::helper::vector< unsigned int>& indices, const double &coord_p)
{

    TriangleSetTopology< DataTypes > *topology = static_cast<TriangleSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    const typename DataTypes::VecCoord& vect_c = *topology->getDOF()->getX();

    const typename DataTypes::Coord& c1=vect_c[indices[0]];
    const typename DataTypes::Coord& c2=vect_c[indices[1]];

    Vec<3,Real> p;

    if(ind_p == indices[0])
    {
        p[0]= (Real) c2[0];
        p[1]= (Real) c2[1];
        p[2]= (Real) c2[2];
    }
    else
    {
        if(ind_p == indices[1])
        {
            p[0]= (Real) c1[0];
            p[1]= (Real) c1[1];
            p[2]= (Real) c1[2];
        }
        else
        {
            p[0]= (Real) ((1.0-coord_p)*c1[0] + coord_p*c2[0]);
            p[1]= (Real) ((1.0-coord_p)*c1[1] + coord_p*c2[1]);
            p[2]= (Real) ((1.0-coord_p)*c1[2] + coord_p*c2[2]);
        }
    }

    return ((Vec<3,double>) p);
}

// Computes the normal vector of a triangle indexed by ind_t (not normed)
template<class DataTypes>
Vec<3,double> TriangleSetGeometryAlgorithms< DataTypes >::computeTriangleNormal(const unsigned int ind_t)
{

    TriangleSetTopology< DataTypes > *topology = static_cast<TriangleSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast< TriangleSetTopologyContainer* >(topology->getTopologyContainer());
    const Triangle &t=container->getTriangle(ind_t);
    const typename DataTypes::VecCoord& vect_c = *topology->getDOF()->getX();

    const typename DataTypes::Coord& c0=vect_c[t[0]];
    const typename DataTypes::Coord& c1=vect_c[t[1]];
    const typename DataTypes::Coord& c2=vect_c[t[2]];

    Vec<3,Real> p0;
    p0[0] = (Real) (c0[0]); p0[1] = (Real) (c0[1]); p0[2] = (Real) (c0[2]);
    Vec<3,Real> p1;
    p1[0] = (Real) (c1[0]); p1[1] = (Real) (c1[1]); p1[2] = (Real) (c1[2]);
    Vec<3,Real> p2;
    p2[0] = (Real) (c2[0]); p2[1] = (Real) (c2[1]); p2[2] = (Real) (c2[2]);

    Vec<3,Real> normal_t=(p1-p0).cross( p2-p0);

    return ((Vec<3,double>) normal_t);

}

// barycentric coefficients of point p in triangle (a,b,c) indexed by ind_t
template<class DataTypes>
sofa::helper::vector< double > TriangleSetGeometryAlgorithms< DataTypes >::computeTriangleBarycoefs( const Vec<3,double> &p, unsigned int ind_t)
{

    sofa::helper::vector< double > baryCoefs;

    TriangleSetTopology< DataTypes > *topology = static_cast<TriangleSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast< TriangleSetTopologyContainer* >(topology->getTopologyContainer());
    const Triangle &t=container->getTriangle(ind_t);
    const typename DataTypes::VecCoord& vect_c = *topology->getDOF()->getX();

    const typename DataTypes::Coord& c0=vect_c[t[0]];
    const typename DataTypes::Coord& c1=vect_c[t[1]];
    const typename DataTypes::Coord& c2=vect_c[t[2]];

    Vec<3,Real> a;
    a[0] = (Real) (c0[0]); a[1] = (Real) (c0[1]); a[2] = (Real) (c0[2]);
    Vec<3,Real> b;
    b[0] = (Real) (c1[0]); b[1] = (Real) (c1[1]); b[2] = (Real) (c1[2]);
    Vec<3,Real> c;
    c[0] = (Real) (c2[0]); c[1] = (Real) (c2[1]); c[2] = (Real) (c2[2]);

    Vec<3,double> M = (Vec<3,double>) (b-a).cross(c-a);
    double norm2_M = M*(M);

    double coef_a; double coef_b; double coef_c;

    if(norm2_M==0.0)  // triangle (a,b,c) is flat
    {

        coef_a = (double) (1.0/3.0);
        coef_b = (double) (1.0/3.0);
        coef_c = (double) (1.0 - (coef_a + coef_b));

    }
    else
    {
        Vec<3,Real> N =  M/norm2_M;

        coef_a = N*((b-p).cross(c-p));
        coef_b = N*((c-p).cross(a-p));
        coef_c = (double) (1.0 - (coef_a + coef_b)); //N*((a-p).cross(b-p));

    }

    baryCoefs.push_back(coef_a); baryCoefs.push_back(coef_b); baryCoefs.push_back(coef_c);


    return baryCoefs;
}

// barycentric coefficients of point p in triangle whose vertices are indexed by (ind_p1,ind_p2,ind_p3)
template<class DataTypes>
sofa::helper::vector< double > TriangleSetGeometryAlgorithms< DataTypes >::compute3PointsBarycoefs( const Vec<3,double> &p, unsigned int ind_p1, unsigned int ind_p2, unsigned int ind_p3)
{

    sofa::helper::vector< double > baryCoefs;

    TriangleSetTopology< DataTypes > *topology = static_cast<TriangleSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);

    const typename DataTypes::VecCoord& vect_c = *topology->getDOF()->getX();

    const typename DataTypes::Coord& c0=vect_c[ind_p1];
    const typename DataTypes::Coord& c1=vect_c[ind_p2];
    const typename DataTypes::Coord& c2=vect_c[ind_p3];

    Vec<3,Real> a;
    a[0] = (Real) (c0[0]); a[1] = (Real) (c0[1]); a[2] = (Real) (c0[2]);
    Vec<3,Real> b;
    b[0] = (Real) (c1[0]); b[1] = (Real) (c1[1]); b[2] = (Real) (c1[2]);
    Vec<3,Real> c;
    c[0] = (Real) (c2[0]); c[1] = (Real) (c2[1]); c[2] = (Real) (c2[2]);

    Vec<3,double> M = (Vec<3,double>) (b-a).cross(c-a);
    double norm2_M = M*(M);

    double coef_a; double coef_b; double coef_c;

    if(norm2_M==0.0)  // triangle (a,b,c) is flat
    {

        coef_a = (double) (1.0/3.0);
        coef_b = (double) (1.0/3.0);
        coef_c = (double) (1.0 - (coef_a + coef_b));

    }
    else
    {
        Vec<3,Real> N =  M/norm2_M;

        coef_a = N*((b-p).cross(c-p));
        coef_b = N*((c-p).cross(a-p));
        coef_c = (double) (1.0 - (coef_a + coef_b)); //N*((a-p).cross(b-p));

    }

    baryCoefs.push_back(coef_a); baryCoefs.push_back(coef_b); baryCoefs.push_back(coef_c);

    return baryCoefs;
}

// test if a point is included in the triangle indexed by ind_t
template<class DataTypes>
bool TriangleSetGeometryAlgorithms< DataTypes >::is_PointinTriangle(bool is_tested, const Vec<3,Real>& p, unsigned int ind_t, unsigned int &ind_t_test)
{

    TriangleSetTopology< DataTypes > *topology = static_cast<TriangleSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast< TriangleSetTopologyContainer* >(topology->getTopologyContainer());
    const typename DataTypes::VecCoord& vect_c = *topology->getDOF()->getX();
    const Triangle &t=container->getTriangle(ind_t);

    const typename DataTypes::Coord& c0=vect_c[t[0]];
    const typename DataTypes::Coord& c1=vect_c[t[1]];
    const typename DataTypes::Coord& c2=vect_c[t[2]];

    Vec<3,Real> ptest = p;

    Vec<3,Real> p0;
    p0[0] = (Real) (c0[0]); p0[1] = (Real) (c0[1]); p0[2] = (Real) (c0[2]);
    Vec<3,Real> p1;
    p1[0] = (Real) (c1[0]); p1[1] = (Real) (c1[1]); p1[2] = (Real) (c1[2]);
    Vec<3,Real> p2;
    p2[0] = (Real) (c2[0]); p2[1] = (Real) (c2[1]); p2[2] = (Real) (c2[2]);

    Vec<3,Real> v_normal = (p2-p0).cross(p1-p0);

    Real norm_v_normal = v_normal*(v_normal);
    if(norm_v_normal != 0.0)
    {

        Vec<3,Real> n_01 = (p1-p0).cross(v_normal);
        Vec<3,Real> n_12 = (p2-p1).cross(v_normal);
        Vec<3,Real> n_20 = (p0-p2).cross(v_normal);

        double v_01 = (double) ((ptest-p0)*(n_01));
        double v_12 = (double) ((ptest-p1)*(n_12));
        double v_20 = (double) ((ptest-p2)*(n_20));

        bool is_inside = (v_01 > 0.0) && (v_12 > 0.0) && (v_20 > 0.0);

        if(is_tested && (!is_inside))
        {

            sofa::helper::vector< unsigned int > shell;
            unsigned int ind_edge = 0;
            const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tvsa=container->getTriangleVertexShellArray();
            const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tesa=container->getTriangleEdgeShellArray();

            if(v_01 < 0.0)
            {
                if(v_12 < 0.0)  /// vertex 1
                {
                    shell =(sofa::helper::vector< unsigned int >) (tvsa[t[1]]);

                }
                else
                {
                    if(v_20 < 0.0)  /// vertex 0
                    {
                        shell =(sofa::helper::vector< unsigned int >) (tvsa[t[0]]);

                    }
                    else   // v_01 < 0.0
                    {
                        ind_edge=container->getEdgeIndex(t[0],t[1]);
                        shell =(sofa::helper::vector< unsigned int >) (tesa[ind_edge]);
                    }
                }
            }
            else
            {
                if(v_12 < 0.0)
                {
                    if(v_20 < 0.0)  /// vertex 2
                    {
                        shell =(sofa::helper::vector< unsigned int >) (tvsa[t[2]]);

                    }
                    else   // v_12 < 0.0
                    {
                        ind_edge=container->getEdgeIndex(t[1],t[2]);
                        shell =(sofa::helper::vector< unsigned int >) (tesa[ind_edge]);
                    }
                }
                else   // v_20 < 0.0
                {
                    ind_edge=container->getEdgeIndex(t[2],t[0]);
                    shell =(sofa::helper::vector< unsigned int >) (tesa[ind_edge]);
                }
            }

            unsigned int i =0;
            bool is_in_next_triangle=false;
            unsigned int ind_triangle=0;
            unsigned ind_t_false_init;
            unsigned int &ind_t_false = ind_t_false_init;

            if(shell.size()>1)
            {

                while(i < shell.size() && !is_in_next_triangle)
                {

                    ind_triangle=shell[i];

                    if(ind_triangle != ind_t)
                    {
                        is_in_next_triangle = topology->getTriangleSetGeometryAlgorithms()->is_PointinTriangle(false, p, ind_triangle, ind_t_false);
                    }
                    i++;
                }

                if(is_in_next_triangle)
                {
                    ind_t_test=ind_triangle;
                    //std::cout << "correct to triangle indexed by " << ind_t_test << std::endl;
                }
                else   // not found
                {
                    //std::cout << "not found !!! " << std::endl;
                    ind_t_test=ind_t;
                }
            }
            else
            {
                ind_t_test=ind_t;
            }

        }
        return is_inside;

    }
    else   // triangle is flat
    {
        //std::cout << "INFO_print : triangle is flat" << std::endl;
        return false;
    }

}

// Tests how to triangularize a quad whose vertices are defined by (p_q1, p_q2, ind_q3, ind_q4) according to the Delaunay criterion
template<class DataTypes>
bool TriangleSetGeometryAlgorithms< DataTypes >::isQuadDeulaunayOriented(const Vec<3,double>& p_q1, const Vec<3,double>& p_q2, unsigned int ind_q3, unsigned int ind_q4)
{

    sofa::helper::vector< double > baryCoefs;

    TriangleSetTopology< DataTypes > *topology = static_cast<TriangleSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);

    const typename DataTypes::VecCoord& vect_c = *topology->getDOF()->getX();

    const typename DataTypes::Coord& c3=vect_c[ind_q3];
    const typename DataTypes::Coord& c4=vect_c[ind_q4];

    Vec<3,Real> q1 = (Vec<3,Real>) p_q1;
    Vec<3,Real> q2 = (Vec<3,Real>) p_q2;

    Vec<3,Real> q3;
    q3[0] = (Real) (c3[0]); q3[1] = (Real) (c3[1]); q3[2] = (Real) (c3[2]);
    Vec<3,Real> q4;
    q4[0] = (Real) (c4[0]); q4[1] = (Real) (c4[1]); q4[2] = (Real) (c4[2]);

    Vec<3,Real> G = (q1+q2+q3)/3.0;

    if((G-q2)*(G-q2) <= (G-q4)*(G-q4))
    {
        return true;
    }
    else
    {
        return false;
    }

}

// Test if a triangle indexed by ind_triangle (and incident to the vertex indexed by ind_p) is included or not in the plane defined by (ind_p, plane_vect)
template<class DataTypes>
bool TriangleSetGeometryAlgorithms< DataTypes >::is_triangle_in_plane(const unsigned int ind_t, const unsigned int ind_p,  const Vec<3,Real>&plane_vect)
{


    TriangleSetTopology< DataTypes > *topology = static_cast<TriangleSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast< TriangleSetTopologyContainer* >(topology->getTopologyContainer());
    const Triangle &t=container->getTriangle(ind_t);

    // HYP : ind_p==t[0] or ind_p==t[1] or ind_p==t[2]

    const typename DataTypes::VecCoord& vect_c = *topology->getDOF()->getX();

    unsigned int ind_1;
    unsigned int ind_2;

    if(ind_p==t[0])
    {
        ind_1=t[1];
        ind_2=t[2];
    }
    else
    {
        if(ind_p==t[1])
        {
            ind_1=t[2];
            ind_2=t[0];
        }
        else   // ind_p==t[2]
        {
            ind_1=t[0];
            ind_2=t[1];
        }
    }

    const typename DataTypes::Coord& c0=vect_c[ind_p];
    const typename DataTypes::Coord& c1=vect_c[ind_1];
    const typename DataTypes::Coord& c2=vect_c[ind_2];

    Vec<3,Real> p0;
    p0[0] = (Real) (c0[0]); p0[1] = (Real) (c0[1]); p0[2] = (Real) (c0[2]);
    Vec<3,Real> p1;
    p1[0] = (Real) (c1[0]); p1[1] = (Real) (c1[1]); p1[2] = (Real) (c1[2]);
    Vec<3,Real> p2;
    p2[0] = (Real) (c2[0]); p2[1] = (Real) (c2[1]); p2[2] = (Real) (c2[2]);

    return((p1-p0)*( plane_vect)>=0.0 && (p1-p0)*( plane_vect)>=0.0);


}


// Prepares the duplication of a vertex
template<class DataTypes>
void TriangleSetGeometryAlgorithms< DataTypes >::Prepare_VertexDuplication(const unsigned int ind_p, const unsigned int ind_t_from, const unsigned int ind_t_to,
        const sofa::helper::vector< unsigned int>& indices_from, const double &coord_from, const sofa::helper::vector< unsigned int>& indices_to, const double &coord_to,
        sofa::helper::vector< unsigned int > &triangles_list_1, sofa::helper::vector< unsigned int > &triangles_list_2)
{

    //HYP : if coord_from or coord_to == 0.0 or 1.0, ind_p is distinct from ind_from and from ind_to


    TriangleSetTopology< DataTypes > *topology = static_cast<TriangleSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast< TriangleSetTopologyContainer* >(topology->getTopologyContainer());

    const typename DataTypes::VecCoord& vect_c = *topology->getDOF()->getX();

    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tvsa=container->getTriangleVertexShellArray();
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tesa=container->getTriangleEdgeShellArray();

    const typename DataTypes::Coord& c_p=vect_c[ind_p];
    Vec<3,Real> point_p;
    point_p[0]= (Real) c_p[0]; point_p[1]= (Real) c_p[1]; point_p[2]= (Real) c_p[2];

    Vec<3,Real> point_from=(Vec<3,Real>) getOppositePoint(ind_p, (sofa::helper::vector< unsigned int>&) indices_from, coord_from);
    Vec<3,Real> point_to=(Vec<3,Real>) getOppositePoint(ind_p, (sofa::helper::vector< unsigned int>&) indices_to, coord_to);

    //Vec<3,Real> point_from=(Vec<3,Real>) computeBaryEdgePoint((sofa::helper::vector< unsigned int>&) indices_from, coord_from);
    //Vec<3,Real> point_to=(Vec<3,Real>) computeBaryEdgePoint((sofa::helper::vector< unsigned int>&) indices_to, coord_to);

    Vec<3,Real> vect_from = point_from - point_p;
    Vec<3,Real> vect_to = point_p - point_to;

    //std::cout << "INFO_print : vect_from = " << vect_from <<  std::endl;
    //std::cout << "INFO_print : vect_to = " << vect_to <<  std::endl;

    Vec<3,Real> normal_from;
    Vec<3,Real> normal_to;

    Vec<3,Real> plane_from;
    Vec<3,Real> plane_to;



    if((coord_from!=0.0) && (coord_from!=1.0))
    {

        normal_from=(Vec<3,Real>) computeTriangleNormal(ind_t_from);
        plane_from=vect_from.cross( normal_from); // inverse ??

    }
    else
    {
        // HYP : only 2 edges maximum are adjacent to the same triangle (otherwise : compute the one which minimizes the normed dotProduct and which gives the positive cross)

        unsigned int ind_edge;

        if(coord_from==0.0)
        {
            ind_edge=container->getEdgeIndex(indices_from[0], ind_p);
        }
        else   // coord_from==1.0
        {
            ind_edge=container->getEdgeIndex(indices_from[1], ind_p);
        }


        if (tesa.size()>0)
        {


            sofa::helper::vector< unsigned int > shell =(sofa::helper::vector< unsigned int >) (tesa[ind_edge]);
            unsigned int ind_triangle=shell[0];
            unsigned int i=0;
            bool is_in_next_triangle=false;

            if(shell.size()>1)
            {

                while(i < shell.size() || !is_in_next_triangle)
                {

                    if(shell[i] != ind_t_from)
                    {
                        ind_triangle=shell[i];
                        is_in_next_triangle=true;
                    }
                    i++;
                }

            }
            else
            {
                return;
            }


            if(is_in_next_triangle)
            {

                Vec<3,Real> normal_from_1=(Vec<3,Real>) computeTriangleNormal(ind_triangle);
                Vec<3,Real> normal_from_2=(Vec<3,Real>) computeTriangleNormal(ind_t_from);

                normal_from=(normal_from_1+normal_from_2)/2.0;
                plane_from=vect_from.cross( normal_from);

            }
            else
            {
                return;
            }
        }
    }


    if((coord_to!=0.0) && (coord_to!=1.0))
    {

        normal_to=(Vec<3,Real>) computeTriangleNormal(ind_t_to);

        plane_to=vect_to.cross( normal_to);

    }
    else
    {
        // HYP : only 2 edges maximum are adjacent to the same triangle (otherwise : compute the one which minimizes the normed dotProduct and which gives the positive cross)

        unsigned int ind_edge;

        if(coord_to==0.0)
        {
            ind_edge=container->getEdgeIndex(indices_to[0], ind_p);
        }
        else   // coord_to==1.0
        {
            ind_edge=container->getEdgeIndex(indices_to[1], ind_p);
        }

        if (tesa.size()>0)
        {

            sofa::helper::vector< unsigned int > shell =(sofa::helper::vector< unsigned int >) (tesa[ind_edge]);
            unsigned int ind_triangle=shell[0];
            unsigned int i=0;
            bool is_in_next_triangle=false;

            if(shell.size()>1)
            {

                while(i < shell.size() || !is_in_next_triangle)
                {

                    if(shell[i] != ind_t_to)
                    {
                        ind_triangle=shell[i];
                        is_in_next_triangle=true;
                    }
                    i++;
                }

            }
            else
            {
                return;
            }

            if(is_in_next_triangle)
            {


                Vec<3,Real> normal_to_1=(Vec<3,Real>) computeTriangleNormal(ind_triangle);
                Vec<3,Real> normal_to_2=(Vec<3,Real>) computeTriangleNormal(ind_t_to);

                normal_to=(normal_to_1+normal_to_2)/2.0;
                plane_to=vect_to.cross( normal_to);

            }
            else
            {
                return;
            }
        }
    }


    if (tvsa.size()>0)
    {

        sofa::helper::vector< unsigned int > shell =(sofa::helper::vector< unsigned int >) (tvsa[ind_p]);
        unsigned int ind_triangle=shell[0];
        unsigned int i=0;

        bool is_in_plane_from;
        bool is_in_plane_to;

        if(shell.size()>1)
        {

            Vec<3,Real> normal_test = plane_from.cross( plane_to);
            Real value_test =   normal_test*(normal_from+normal_to);

            if(value_test<=0.0)
            {

                //std::cout << "INFO_print : CONVEXE, value_test = " << value_test <<  std::endl;
                //std::cout << "INFO_print : shell.size() = " << shell.size() << ", ind_t_from = " << ind_t_from << ", ind_t_to = " << ind_t_to <<  std::endl;

                while(i < shell.size())
                {

                    ind_triangle=shell[i];

                    is_in_plane_from=is_triangle_in_plane(ind_triangle,ind_p, (const Vec<3,double>&) plane_from);
                    is_in_plane_to=is_triangle_in_plane(ind_triangle,ind_p, (const Vec<3,double>&) plane_to);

                    if((ind_triangle != ind_t_from) && (ind_triangle != ind_t_to))
                    {
                        if(is_in_plane_from || is_in_plane_to)
                        {
                            triangles_list_1.push_back(ind_triangle);
                        }
                        else
                        {
                            triangles_list_2.push_back(ind_triangle);
                        }

                    }
                    i++;
                }

            }
            else   // value_test>0.0
            {

                //std::cout << "INFO_print : CONCAVE, value_test = " << value_test <<  std::endl;
                //std::cout << "INFO_print : shell.size() = " << shell.size() << ", ind_t_from = " << ind_t_from << ", ind_t_to = " << ind_t_to <<  std::endl;

                while(i < shell.size())
                {

                    ind_triangle=shell[i];

                    is_in_plane_from=is_triangle_in_plane(ind_triangle,ind_p, (const Vec<3,double>&) plane_from);
                    is_in_plane_to=is_triangle_in_plane(ind_triangle,ind_p, (const Vec<3,double>&) plane_to);

                    if((ind_triangle != ind_t_from) && (ind_triangle != ind_t_to))
                    {
                        if(is_in_plane_from && is_in_plane_to)
                        {
                            triangles_list_1.push_back(ind_triangle);
                        }
                        else
                        {
                            triangles_list_2.push_back(ind_triangle);
                        }

                    }
                    i++;
                }

            }
        }
        else
        {
            return;
        }
    }
    else
    {
        return;
    }


}

// Computes the intersection of the segment from point a to point b and the triangle indexed by t
template<class DataTypes>
bool TriangleSetGeometryAlgorithms< DataTypes >::computeSegmentTriangleIntersection(bool is_entered, const Vec<3,double>& a, const Vec<3,double>& b, const unsigned int ind_t,
        sofa::helper::vector<unsigned int> &indices,
        double &baryCoef, double& coord_kmin)
{

    // HYP : point a is in triangle indexed by t
    // is_entered == true => indices.size() == 2

    unsigned int ind_first=0;
    unsigned int ind_second=0;

    if(indices.size()>1)
    {
        ind_first=indices[0];
        ind_second=indices[1];
    }

    indices.clear();

    bool is_validated = false;
    bool is_intersected = false;

    TriangleSetTopology< DataTypes > *topology = static_cast<TriangleSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast< TriangleSetTopologyContainer* >(topology->getTopologyContainer());
    const Triangle &t=container->getTriangle(ind_t);
    const typename DataTypes::VecCoord& vect_c = *topology->getDOF()->getX();

    bool is_full_01=(is_entered && ((t[0] == ind_first && t[1] == ind_second) || (t[1] == ind_first && t[0] == ind_second)));
    bool is_full_12=(is_entered && ((t[1] == ind_first && t[2] == ind_second) || (t[2] == ind_first && t[1] == ind_second)));
    bool is_full_20=(is_entered && ((t[2] == ind_first && t[0] == ind_second) || (t[0] == ind_first && t[2] == ind_second)));


    const typename DataTypes::Coord& c0=vect_c[t[0]];
    const typename DataTypes::Coord& c1=vect_c[t[1]];
    const typename DataTypes::Coord& c2=vect_c[t[2]];

    Vec<3,Real> p0;
    p0[0] = (Real) (c0[0]); p0[1] = (Real) (c0[1]); p0[2] = (Real) (c0[2]);
    Vec<3,Real> p1;
    p1[0] = (Real) (c1[0]); p1[1] = (Real) (c1[1]); p1[2] = (Real) (c1[2]);
    Vec<3,Real> p2;
    p2[0] = (Real) (c2[0]); p2[1] = (Real) (c2[1]); p2[2] = (Real) (c2[2]);

    Vec<3,Real> pa;
    pa[0] = (Real) (a[0]); pa[1] = (Real) (a[1]); pa[2] = (Real) (a[2]);
    Vec<3,Real> pb;
    pb[0] = (Real) (b[0]); pb[1] = (Real) (b[1]); pb[2] = (Real) (b[2]);


    Vec<3,Real> v_normal = (p2-p0).cross(p1-p0);
    //Vec<3,Real> v_normal = (Vec<3,Real>) computeTriangleNormal(ind_t);

    Real norm_v_normal = v_normal.norm(); // WARN : square root COST


    if(norm_v_normal != 0.0)
    {

        v_normal/=norm_v_normal;

        Vec<3,Real> v_ab = pb-pa;
        Vec<3,Real> v_ab_proj = v_ab - v_ab*(v_normal)*v_normal; // projection
        Vec<3,Real> pb_proj = v_ab_proj + pa;

        Vec<3,Real> v_01 = p1-p0;
        Vec<3,Real> v_12 = p2-p1;
        Vec<3,Real> v_20 = p0-p2;

        Vec<3,Real> n_proj =v_ab_proj.cross(v_normal);
        Vec<3,Real> n_01 = v_01.cross(v_normal);
        Vec<3,Real> n_12 = v_12.cross(v_normal);
        Vec<3,Real> n_20 = v_20.cross(v_normal);

        Real norm2_v_ab_proj = v_ab_proj*(v_ab_proj);

        if(norm2_v_ab_proj != 0.0)  // pb_proj != pa
        {

            double init_t=0.0;
            double init_k=0.0;
            double init_kmin=0.0;

            double& coord_t=init_t;
            double& coord_k=init_k;

            double is_initialized=false;
            coord_kmin=init_kmin;

            double coord_test1;
            double coord_test2;

            double s_t;
            double s_k;

            if(!is_full_01)
            {

                /// Test of edge (p0,p1) :

                s_t = (p0-p1)*n_proj;
                s_k = (pa-pb_proj)*n_01;

                // s_t == 0.0 iff s_k == 0.0

                if(s_t==0.0)  // (pa,pb_proj) and (p0,p1) are parallel
                {

                    if((p0-pa)*(n_proj)==0.0)  // (pa,pb_proj) and (p0,p1) are on the same line
                    {

                        coord_test1 = (pa-p0)*(pa-pb_proj)/norm2_v_ab_proj; // HYP : pb_proj != pa
                        coord_test2 = (pa-p1)*(pa-pb_proj)/norm2_v_ab_proj; // HYP : pb_proj != pa

                        if(coord_test1>=0)
                        {
                            coord_k=coord_test1;
                            coord_t=0.0;
                        }
                        else
                        {
                            coord_k=coord_test2;
                            coord_t=1.0;
                        }

                        is_intersected = (coord_k > 0.0 && (coord_t >= 0.0 && coord_t <= 1.0));



                    }
                    else   // (pa,pb_proj) and (p0,p1) are parallel and disjoint
                    {

                        is_intersected=false;

                    }

                }
                else   // s_t != 0.0 and s_k != 0.0
                {

                    coord_k=double((pa-p0)*(n_01))*1.0/double(s_k);
                    coord_t=double((p0-pa)*(n_proj))*1.0/double(s_t);

                    is_intersected = ((coord_k > 0.0) && (coord_t >= 0.0 && coord_t <= 1.0));

                }

                if(is_intersected)
                {

                    if((!is_initialized) || (coord_k > coord_kmin))
                    {

                        indices.clear();
                        indices.push_back(t[0]);
                        indices.push_back(t[1]);
                        baryCoef=coord_t;
                        coord_kmin=coord_k;
                    }

                    is_initialized=true;
                }

                is_validated = is_validated || is_initialized;
            }

            if(!is_full_12)
            {

                /// Test of edge (p1,p2) :

                s_t = (p1-p2)*(n_proj);
                s_k = (pa-pb_proj)*(n_12);

                // s_t == 0.0 iff s_k == 0.0

                if(s_t==0.0)  // (pa,pb_proj) and (p1,p2) are parallel
                {

                    if((p1-pa)*(n_proj)==0.0)  // (pa,pb_proj) and (p1,p2) are on the same line
                    {

                        coord_test1 = (pa-p1)*(pa-pb_proj)/norm2_v_ab_proj; // HYP : pb_proj != pa
                        coord_test2 = (pa-p2)*(pa-pb_proj)/norm2_v_ab_proj; // HYP : pb_proj != pa

                        if(coord_test1>=0)
                        {
                            coord_k=coord_test1;
                            coord_t=0.0;
                        }
                        else
                        {
                            coord_k=coord_test2;
                            coord_t=1.0;
                        }

                        is_intersected = (coord_k > 0.0 && (coord_t >= 0.0 && coord_t <= 1.0));

                    }
                    else   // (pa,pb_proj) and (p1,p2) are parallel and disjoint
                    {

                        is_intersected=false;

                    }

                }
                else   // s_t != 0.0 and s_k != 0.0
                {

                    coord_k=double((pa-p1)*(n_12))*1.0/double(s_k);
                    coord_t=double((p1-pa)*(n_proj))*1.0/double(s_t);

                    is_intersected = ((coord_k > 0.0) && (coord_t >= 0.0 && coord_t <= 1.0));
                }

                if(is_intersected)
                {

                    if((!is_initialized) || (coord_k > coord_kmin))
                    {

                        indices.clear();
                        indices.push_back(t[1]);
                        indices.push_back(t[2]);
                        baryCoef=coord_t;
                        coord_kmin=coord_k;
                    }

                    is_initialized=true;

                }

                is_validated = is_validated || is_initialized;

            }

            if(!is_full_20)
            {

                /// Test of edge (p2,p0) :

                s_t = (p2-p0)*(n_proj);
                s_k = (pa-pb_proj)*(n_20);

                // s_t == 0.0 iff s_k == 0.0

                if(s_t==0.0)  // (pa,pb_proj) and (p2,p0) are parallel
                {

                    if((p2-pa)*(n_proj)==0.0)  // (pa,pb_proj) and (p2,p0) are on the same line
                    {

                        coord_test1 = (pa-p2)*(pa-pb_proj)/norm2_v_ab_proj; // HYP : pb_proj != pa
                        coord_test2 = (pa-p0)*(pa-pb_proj)/norm2_v_ab_proj; // HYP : pb_proj != pa

                        if(coord_test1>=0)
                        {
                            coord_k=coord_test1;
                            coord_t=0.0;
                        }
                        else
                        {
                            coord_k=coord_test2;
                            coord_t=1.0;
                        }

                        is_intersected = (coord_k > 0.0 && (coord_t >= 0.0 && coord_t <= 1.0));

                    }
                    else   // (pa,pb_proj) and (p2,p0) are parallel and disjoint
                    {

                        is_intersected=false;

                    }

                }
                else   // s_t != 0.0 and s_k != 0.0
                {

                    coord_k=double((pa-p2)*(n_20))*1.0/double(s_k);
                    coord_t=double((p2-pa)*(n_proj))*1.0/double(s_t);

                    is_intersected = ((coord_k > 0.0) && (coord_t >= 0.0 && coord_t <= 1.0));
                }

                if(is_intersected)
                {

                    if((!is_initialized) || (coord_k > coord_kmin))
                    {

                        indices.clear();
                        indices.push_back(t[2]);
                        indices.push_back(t[0]);
                        baryCoef=coord_t;
                        coord_kmin=coord_k;
                    }

                    is_initialized=true;

                }

                is_validated = is_validated || is_initialized;
            }

        }
        else
        {
            is_validated=false; // points a and b are projected to the same point on triangle t
        }


    }
    else
    {
        is_validated=false; // triangle t is flat
    }

    return is_validated;

}


// Computes the list of points (edge,coord) intersected by the segment from point a to point b
// and the triangular mesh
template<class DataTypes>
bool TriangleSetGeometryAlgorithms< DataTypes >::computeIntersectedPointsList(const Vec<3,double>& a, const Vec<3,double>& b, const unsigned int ind_ta, unsigned int& ind_tb,
        sofa::helper::vector< unsigned int > &triangles_list, sofa::helper::vector< sofa::helper::vector<unsigned int> > &indices_list, sofa::helper::vector< double >& coords_list, bool& is_on_boundary)
{

    TriangleSetTopology< DataTypes > *topology = static_cast<TriangleSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast< TriangleSetTopologyContainer* >(topology->getTopologyContainer());

    bool is_validated=true;
    bool is_intersected=true;

    Vec<3,double> c_t_test = a;

    is_on_boundary = false;

    sofa::helper::vector<unsigned int> init_indices;
    sofa::helper::vector<unsigned int> &indices=init_indices;

    double init_t=0.0;
    double init_k=0.0;
    double init_k_test=0.0;

    double& coord_t=init_t;
    double& coord_k=init_k;

    Vec<3,double> p_current=a;
    unsigned int ind_t_current=ind_ta;

    unsigned int ind_edge;
    unsigned int ind_index;
    unsigned int ind_triangle = ind_ta;
    double coord_k_test=init_k_test;

    const Vec<3,double>& p_const=p_current;

    double dist_min=0.0;

    is_intersected=computeSegmentTriangleIntersection(false, p_const, b, (const unsigned int) ind_t_current, indices, coord_t, coord_k);

    coord_k_test=coord_k;
    dist_min=(b-a)*(b-a);

    while((coord_k_test<1.0 && is_validated) && is_intersected)
    {

        ind_edge=container->getEdgeIndex(indices[0],indices[1]);
        sofa::helper::vector< unsigned int > indices_first_list; indices_first_list.push_back(indices[0]); indices_first_list.push_back(indices[1]);
        triangles_list.push_back(ind_t_current);
        indices_list.push_back(indices_first_list);
        coords_list.push_back(coord_t);

        const typename DataTypes::VecCoord& vect_c = *topology->getDOF()->getX();

        Vec<3,double> c_t_current; // WARNING : conversion from 'double' to 'float', possible loss of data ! // typename DataTypes::Coord
        c_t_current[0]=(double) ((1.0-coord_t)*((double) (vect_c[indices[0]][0]))+coord_t*((double) (vect_c[indices[1]][0])));
        c_t_current[1]=(double) ((1.0-coord_t)*((double) (vect_c[indices[0]][1]))+coord_t*((double) (vect_c[indices[1]][1])));
        c_t_current[2]=(double) ((1.0-coord_t)*((double) (vect_c[indices[0]][2]))+coord_t*((double) (vect_c[indices[1]][2])));

        p_current=c_t_current;

        Vec<3,Real> p_t_aux;
        p_t_aux[0] = (Real) (c_t_current[0]); p_t_aux[1] = (Real) (c_t_current[1]); p_t_aux[2] = (Real) (c_t_current[2]);

        if(coord_t==0.0 || coord_t==1.0)  // current point indexed by ind_t_current is on a vertex
        {

            //std::cout << "INFO_print : INPUT ON A VERTEX !!!" <<  std::endl;

            if(coord_t==0.0)
            {
                ind_index=indices[0];
            }
            else   // coord_t==1.0
            {
                ind_index=indices[1];
            }

            const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tvsa=container->getTriangleVertexShellArray();

            if (tvsa.size()>0)
            {

                sofa::helper::vector< unsigned int > shell =(sofa::helper::vector< unsigned int >) (tvsa[ind_index]);
                ind_triangle=shell[0];
                unsigned int i=0;
                //bool is_in_next_triangle=false;
                bool is_test_init=false;

                unsigned int ind_from = ind_t_current;

                if(shell.size()>1)  // at leat one neighbor triangle which is not indexed by ind_t_current
                {

                    is_on_boundary=false;

                    while(i < shell.size())
                    {

                        if(shell[i] != ind_from)
                        {

                            ind_triangle=shell[i];

                            const Triangle &t=container->getTriangle(ind_triangle);

                            const typename DataTypes::Coord& c0=vect_c[t[0]];
                            const typename DataTypes::Coord& c1=vect_c[t[1]];
                            const typename DataTypes::Coord& c2=vect_c[t[2]];

                            Vec<3,Real> p0_aux;
                            p0_aux[0] = (Real) (c0[0]); p0_aux[1] = (Real) (c0[1]); p0_aux[2] = (Real) (c0[2]);
                            Vec<3,Real> p1_aux;
                            p1_aux[0] = (Real) (c1[0]); p1_aux[1] = (Real) (c1[1]); p1_aux[2] = (Real) (c1[2]);
                            Vec<3,Real> p2_aux;
                            p2_aux[0] = (Real) (c2[0]); p2_aux[1] = (Real) (c2[1]); p2_aux[2] = (Real) (c2[2]);

                            const Vec<3,double>& p_const_current=p_current;
                            is_intersected=computeSegmentTriangleIntersection(true, p_const_current, b, ind_triangle, indices, coord_t, coord_k);

                            if(is_intersected)
                            {

                                Vec<3,double> c_t_test; // WARNING : conversion from 'double' to 'float', possible loss of data ! // typename DataTypes::Coord
                                c_t_test[0]=(double) ((1.0-coord_t)*((double) (vect_c[indices[0]][0]))+coord_t*((double) (vect_c[indices[1]][0])));
                                c_t_test[1]=(double) ((1.0-coord_t)*((double) (vect_c[indices[0]][1]))+coord_t*((double) (vect_c[indices[1]][1])));
                                c_t_test[2]=(double) ((1.0-coord_t)*((double) (vect_c[indices[0]][2]))+coord_t*((double) (vect_c[indices[1]][2])));

                                double dist_test=(b-c_t_test)*(b-c_t_test);

                                if(is_test_init)
                                {

                                    if(dist_test<dist_min && coord_k<=1)  //dist_test<dist_min
                                    {

                                        coord_k_test=coord_k;
                                        dist_min=dist_test;
                                        ind_t_current=ind_triangle;
                                    }
                                }
                                else
                                {
                                    is_test_init=true;
                                    coord_k_test=coord_k;
                                    dist_min=dist_test;
                                    ind_t_current=ind_triangle;
                                }

                            }

                        }

                        i=i+1;
                    }

                    is_intersected=is_test_init;

                }
                else
                {

                    is_on_boundary=true;
                    is_validated=false;

                }

            }
            else
            {

                is_validated=false;

            }

        }
        else   // current point indexed by ind_t_current is on an edge, but not on a vertex
        {


            ind_edge=container->getEdgeIndex(indices[0],indices[1]);

            const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tesa=container->getTriangleEdgeShellArray();

            if (tesa.size()>0)
            {

                sofa::helper::vector< unsigned int > shell =(sofa::helper::vector< unsigned int >) (tesa[ind_edge]);
                ind_triangle=shell[0];
                unsigned int i=0;

                bool is_test_init=false;

                unsigned int ind_from = ind_t_current;

                if(shell.size()>1)  // at leat one neighbor triangle which is not indexed by ind_t_current
                {

                    is_on_boundary=false;

                    while(i < shell.size())
                    {

                        if(shell[i] != ind_from)
                        {

                            ind_triangle=shell[i];

                            const Triangle &t=container->getTriangle(ind_triangle);

                            const typename DataTypes::Coord& c0=vect_c[t[0]];
                            const typename DataTypes::Coord& c1=vect_c[t[1]];
                            const typename DataTypes::Coord& c2=vect_c[t[2]];

                            Vec<3,Real> p0_aux;
                            p0_aux[0] = (Real) (c0[0]); p0_aux[1] = (Real) (c0[1]); p0_aux[2] = (Real) (c0[2]);
                            Vec<3,Real> p1_aux;
                            p1_aux[0] = (Real) (c1[0]); p1_aux[1] = (Real) (c1[1]); p1_aux[2] = (Real) (c1[2]);
                            Vec<3,Real> p2_aux;
                            p2_aux[0] = (Real) (c2[0]); p2_aux[1] = (Real) (c2[1]); p2_aux[2] = (Real) (c2[2]);


                            const Vec<3,double>& p_const_current=p_current;
                            is_intersected=computeSegmentTriangleIntersection(true, p_const_current, b, ind_triangle, indices, coord_t, coord_k);

                            if(is_intersected)
                            {

                                //Vec<3,double> c_t_test; // WARNING : conversion from 'double' to 'float', possible loss of data ! // typename DataTypes::Coord
                                c_t_test[0]=(double) ((1.0-coord_t)*((double) (vect_c[indices[0]][0]))+coord_t*((double) (vect_c[indices[1]][0])));
                                c_t_test[1]=(double) ((1.0-coord_t)*((double) (vect_c[indices[0]][1]))+coord_t*((double) (vect_c[indices[1]][1])));
                                c_t_test[2]=(double) ((1.0-coord_t)*((double) (vect_c[indices[0]][2]))+coord_t*((double) (vect_c[indices[1]][2])));

                                double dist_test=(b-c_t_test)*(b-c_t_test);

                                if(is_test_init)
                                {

                                    if(dist_test<dist_min && coord_k<=1)  //dist_test<dist_min
                                    {

                                        coord_k_test=coord_k;
                                        dist_min=dist_test;
                                        ind_t_current=ind_triangle;
                                    }
                                }
                                else
                                {
                                    is_test_init=true;
                                    coord_k_test=coord_k;
                                    dist_min=dist_test;
                                    ind_t_current=ind_triangle;
                                }

                            }


                        }

                        i=i+1;
                    }

                    is_intersected=is_test_init;

                }
                else
                {

                    is_on_boundary=true;
                    is_validated=false;

                }

            }
            else
            {

                is_validated=false;

            }

        }

    }

    bool is_reached = (ind_tb==ind_triangle && coord_k_test>=1.0);


    if(is_reached)
    {
        std::cout << "INFO_print - TriangleSetTopology.inl : Cut is reached" << std::endl;
    }

    if(is_on_boundary)
    {
        std::cout << "INFO_print - TriangleSetTopology.inl : Cut meets a mesh boundary" << std::endl;
    }

    if(!is_reached && !is_on_boundary)
    {
        std::cout << "INFO_print - TriangleSetTopology.inl : Cut is not reached" << std::endl;
        ind_tb=ind_triangle;
    }


    return (is_reached && is_validated && is_intersected); // b is in triangle indexed by ind_t_current
}

/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////TriangleSetTopology//////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
template<class DataTypes>
void TriangleSetTopology<DataTypes>::init()
{
}
template<class DataTypes>
TriangleSetTopology<DataTypes>::TriangleSetTopology(MechanicalObject<DataTypes> *obj) : EdgeSetTopology<DataTypes>( obj),f_m_topologyContainer(new DataPtr< TriangleSetTopologyContainer >(new TriangleSetTopologyContainer(), "Triangle Container"))

{
    this->m_topologyContainer=f_m_topologyContainer->beginEdit();
    this->m_topologyContainer->setTopology(this);
    this->m_topologyContainer->setTopology(this);
    this->m_topologyModifier=(new TriangleSetTopologyModifier<DataTypes>(this));
    this->m_topologyAlgorithms=(new TriangleSetTopologyAlgorithms<DataTypes>(this));
    this->m_geometryAlgorithms=(new TriangleSetGeometryAlgorithms<DataTypes>(this));

    this->addField(f_m_topologyContainer, "trianglecontainer");
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_TriangleSetTOPOLOGY_INL

