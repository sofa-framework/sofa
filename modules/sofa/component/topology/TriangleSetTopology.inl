#ifndef SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGY_INL
#define SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGY_INL

#include <sofa/component/topology/TriangleSetTopology.h>
#include <sofa/component/topology/PointSetTopology.inl>
#include <sofa/component/topology/TopologyChangedEvent.h>
#include <sofa/simulation/tree/PropagateEventVisitor.h>
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
        //unsigned int p[3];p[0]=(unsigned int) p1;p[1]=(unsigned int) p2;p[2]=(unsigned int) p;
        tstm->addTriangle(Triangle(make_array<unsigned int>((unsigned int)p1,(unsigned int)p2,(unsigned int) p3)));
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
void TriangleSetTopologyModifier<DataTypes>::addTrianglesProcess(const std::vector< Triangle > &triangles)
{
    TriangleSetTopology<DataTypes> *topology = dynamic_cast<TriangleSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast<TriangleSetTopologyContainer *>( topology->getTopologyContainer() );
    assert (container != 0);
    if (container->m_triangle.size()>0)
    {
        unsigned int triangleIndex;
        const std::vector< std::vector<unsigned int> > &tvsa=container->getTriangleVertexShellArray();
        const std::vector< std::vector<unsigned int> > &tesa=container->getTriangleEdgeShellArray();

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
                    assert(edgeIndex!= -1);
                    container->m_triangleEdge[triangleIndex][j]= edgeIndex;
                }
            }
            if (tesa.size()>0)
            {
                for (j=0; j<3; ++j)
                {
                    container->m_triangleEdgeShell[container->m_triangleEdge[triangleIndex][j]].push_back( triangleIndex );
                }
            }

        }
    }
}



template<class DataTypes>
void TriangleSetTopologyModifier<DataTypes>::addTrianglesWarning(const unsigned int nTriangles, const std::vector< Triangle >& trianglesList,
        const std::vector< unsigned int >& trianglesIndexList,
        const std::vector< std::vector< unsigned int > > & ancestors,
        const std::vector< std::vector< double > >& baryCoefs)
{
    // Warning that triangles just got created
    TrianglesAdded *e=new TrianglesAdded(nTriangles, trianglesList,trianglesIndexList,ancestors,baryCoefs);
    this->addTopologyChange(e);
}




template<class DataTypes>
void TriangleSetTopologyModifier<DataTypes>::removeTrianglesWarning( std::vector<unsigned int> &triangles)
{
    /// sort vertices to remove in a descendent order
    std::sort( triangles.begin(), triangles.end(), std::greater<unsigned int>() );

    // Warning that these triangles will be deleted
    TrianglesRemoved *e=new TrianglesRemoved(triangles);
    this->addTopologyChange(e);

}



template<class DataTypes>
void TriangleSetTopologyModifier<DataTypes>::removeTrianglesProcess(const std::vector<unsigned int> &indices,const bool removeIsolatedItems)
{
    TriangleSetTopology<DataTypes> *topology = dynamic_cast<TriangleSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast<TriangleSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);


    /// only remove isolated edges if the structures exists since removeEdges
    /// will remove isolated vertices
    if (removeIsolatedItems)
    {
        /// force the creation of the Triangle Edge Shell array to detect isolated edges
        if (container->m_triangleEdge.size()>0)
            container->getTriangleEdgeShellArray();
        /// force the creation of the Triangle Shell array to detect isolated vertices
        container->getTriangleVertexShellArray();
    }


    if (container->m_triangle.size()>0)
    {
        std::vector<unsigned int> edgeToBeRemoved;
        std::vector<unsigned int> vertexToBeRemoved;



        for (unsigned int i = 0; i < indices.size(); ++i)
        {
            Triangle &t = container->m_triangle[ indices[i] ];
            // first check that the triangle vertex shell array has been initialized
            if (container->m_triangleVertexShell.size()>0)
            {

                std::vector< unsigned int > &shell0 = container->m_triangleVertexShell[ t[0] ];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell0.begin(), shell0.end(), indices[i] ) !=shell0.end());
                shell0.erase( std::find( shell0.begin(), shell0.end(), indices[i] ) );
                if ((removeIsolatedItems) && (shell0.size()==0))
                    vertexToBeRemoved.push_back(t[0]);


                std::vector< unsigned int > &shell1 = container->m_triangleVertexShell[ t[1] ];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell1.begin(), shell1.end(), indices[i] ) !=shell1.end());
                shell1.erase( std::find( shell1.begin(), shell1.end(), indices[i] ) );
                if ((removeIsolatedItems) && (shell1.size()==0))
                    vertexToBeRemoved.push_back(t[1]);

                std::vector< unsigned int > &shell2 = container->m_triangleVertexShell[ t[2] ];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell2.begin(), shell2.end(), indices[i] ) !=shell2.end());
                shell2.erase( std::find( shell2.begin(), shell2.end(), indices[i] ) );
                if ((removeIsolatedItems) && (shell2.size()==0))
                    vertexToBeRemoved.push_back(t[2]);
            }

            /** first check that the triangle edge shell array has been initialized */
            if (container->m_triangleEdgeShell.size()>0)
            {
                std::vector< unsigned int > &shell0 = container->m_triangleEdgeShell[ container->m_triangleEdge[indices[i]][0]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell0.begin(), shell0.end(), indices[i] ) !=shell0.end());
                shell0.erase( std::find( shell0.begin(), shell0.end(), indices[i] ) );
                if ((removeIsolatedItems) && (shell0.size()==0))
                    edgeToBeRemoved.push_back(container->m_triangleEdge[indices[i]][0]);

                std::vector< unsigned int > &shell1 = container->m_triangleEdgeShell[ container->m_triangleEdge[indices[i]][1]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell1.begin(), shell1.end(), indices[i] ) !=shell1.end());
                shell1.erase( std::find( shell1.begin(), shell1.end(), indices[i] ) );
                if ((removeIsolatedItems) && (shell1.size()==0))
                    edgeToBeRemoved.push_back(container->m_triangleEdge[indices[i]][1]);


                std::vector< unsigned int > &shell2 = container->m_triangleEdgeShell[ container->m_triangleEdge[indices[i]][2]];
                // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                assert(std::find( shell2.begin(), shell2.end(), indices[i] ) !=shell2.end());
                shell2.erase( std::find( shell2.begin(), shell2.end(), indices[i] ) );
                if ((removeIsolatedItems) && (shell2.size()==0))
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

                    std::vector< unsigned int > &shell0 = container->m_triangleVertexShell[ t[0] ];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell0.begin(), shell0.end(), oldTriangleIndex ) !=shell0.end());
                    std::vector< unsigned int >::iterator it=std::find( shell0.begin(), shell0.end(), oldTriangleIndex );
                    (*it)=indices[i];

                    std::vector< unsigned int > &shell1 = container->m_triangleVertexShell[ t[1] ];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell1.begin(), shell1.end(), oldTriangleIndex ) !=shell1.end());
                    it=std::find( shell1.begin(), shell1.end(), oldTriangleIndex );
                    (*it)=indices[i];


                    std::vector< unsigned int > &shell2 = container->m_triangleVertexShell[ t[2] ];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell2.begin(), shell2.end(), oldTriangleIndex ) !=shell2.end());
                    it=std::find( shell2.begin(), shell2.end(), oldTriangleIndex );
                    (*it)=indices[i];


                }
                if (container->m_triangleEdgeShell.size()>0)
                {

                    std::vector< unsigned int > &shell0 =  container->m_triangleEdgeShell[ container->m_triangleEdge[indices[i]][0]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell0.begin(), shell0.end(), oldTriangleIndex ) !=shell0.end());
                    std::vector< unsigned int >::iterator it=std::find( shell0.begin(), shell0.end(), oldTriangleIndex );
                    (*it)=indices[i];

                    std::vector< unsigned int > &shell1 =  container->m_triangleEdgeShell[ container->m_triangleEdge[indices[i]][1]];
                    // removes the first occurence (should be the only one) of the edge in the edge shell of the point
                    assert(std::find( shell1.begin(), shell1.end(), oldTriangleIndex ) !=shell1.end());
                    it=std::find( shell1.begin(), shell1.end(), oldTriangleIndex );
                    (*it)=indices[i];

                    std::vector< unsigned int > &shell2 =  container->m_triangleEdgeShell[ container->m_triangleEdge[indices[i]][2]];
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
void TriangleSetTopologyModifier< DataTypes >::addPointsProcess(const unsigned int nPoints,
        const std::vector< std::vector< unsigned int > >& ancestors,
        const std::vector< std::vector< double > >& baryCoefs)
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
void TriangleSetTopologyModifier< DataTypes >::addEdgesProcess(const std::vector< Edge > &edges)
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
void TriangleSetTopologyModifier< DataTypes >::removePointsProcess( std::vector<unsigned int> &indices)
{
    // now update the local container structures
    TriangleSetTopology<DataTypes> *topology = dynamic_cast<TriangleSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast<TriangleSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);

    // force the creation of the triangle vertex shell array before any point is deleted
    container->getTriangleVertexShellArray();

    // start by calling the standard method.
    EdgeSetTopologyModifier< DataTypes >::removePointsProcess( indices );

    int vertexIndex;

    unsigned int lastPoint = container->m_triangleVertexShell.size() - 1;

    for (unsigned int i = 0; i < indices.size(); ++i)
    {
        // updating the triangles connected to the point replacing the removed one:
        // for all triangles connected to the last point
        std::vector<unsigned int>::iterator itt=container->m_triangleVertexShell[lastPoint].begin();
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
void TriangleSetTopologyModifier< DataTypes >::removeEdgesProcess( const std::vector<unsigned int> &indices,const bool removeIsolatedItems)
{
    // now update the local container structures
    TriangleSetTopology<DataTypes> *topology = dynamic_cast<TriangleSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast<TriangleSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);

    if (container->m_triangleEdge.size()>0)
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
            std::vector<unsigned int>::iterator itt=container->m_triangleEdgeShell[lastEdge].begin();
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
void TriangleSetTopologyModifier< DataTypes >::renumberPointsProcess( const std::vector<unsigned int> &index)
{
    // start by calling the standard method
    EdgeSetTopologyModifier< DataTypes >::renumberPointsProcess( index );

    // now update the local container structures.
    TriangleSetTopology<DataTypes> *topology = dynamic_cast<TriangleSetTopology<DataTypes> *>(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast<TriangleSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);

    std::vector< std::vector< unsigned int > > triangleVertexShell_cp = container->m_triangleVertexShell;
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
void TriangleSetTopologyAlgorithms< DataTypes >::removeTriangles(std::vector< unsigned int >& triangles)
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

    modifier->removeTrianglesProcess(  triangles ,true);

    assert(topology->getTriangleSetTopologyContainer()->checkTopology());
}
/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////TriangleSetGeometryAlgorithms//////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Dot product for 3-elements vectors.
template<typename real>
inline real dotProduct(const Vec<3,real>& a, const Vec<3,real>& b)
{
    return a.x()*b.x()+a.y()*b.y()+a.z()*b.z();
}

/// Cross product for 3-elements vectors.
template<typename real>
inline Vec<3,real> crossProduct(const Vec<3,real>& a, const Vec<3,real>& b)
{
    return Vec<3,real>(a.y()*b.z() - a.z()*b.y(),
            a.z()*b.x() - a.x()*b.z(),
            a.x()*b.y() - a.y()*b.x());
}

/// Test intersection of the segment defined by (pa, vproj_ab) and the edge (p1,p2)
template<typename real>
bool test_intersect(const Vec<3,real>& p1, const Vec<3,real>& p2, const Vec<3,real>& pa, const Vec<3,real>& vproj_ab, double &coord_p, double &coord_k)
{

    // HYP : vproj_ab not equal to zero

    bool is_validated=false;
    Vec<3,real> proj_b=vproj_ab+pa;

    Vec<3,real> vi=crossProduct(p2-p1,vproj_ab);
    Vec<3,real> vj=crossProduct(pa-p1,vproj_ab);

    if(vi.x() != 0.0)
    {

        coord_p=vj.x()/vi.x();
        coord_k=(((real) coord_p)*(p2-p1)+(p1-pa)).x()/vproj_ab.x();
        is_validated=(coord_k>0.0 && (coord_p>=0.0 && coord_p<=1.0));

    }
    else
    {
        if(vi.y() != 0.0)
        {

            coord_p=vj.y()/vi.y();
            coord_k=(((real) coord_p)*(p2-p1)+(p1-pa)).x()/vproj_ab.x();
            is_validated=(coord_k>0.0 && (coord_p>=0.0 && coord_p<=1.0));
        }
        else
        {
            if(vi.z() != 0.0)
            {

                coord_p=vj.z()/vi.z();
                coord_k=(((real) coord_p)*(p2-p1)+(p1-pa)).x()/vproj_ab.x();
                is_validated=(coord_k>0.0 && (coord_p>=0.0 && coord_p<=1.0));

            }
            else   // p1-p2 and vproj_ab are parallel
            {

                if(vj.x()==0.0 && vj.y()==0.0 && vj.z()==0.0)  // p1-p2 and vproj_ab are on the same line
                {
                    coord_p=0.0;
                    coord_k=(((real) coord_p)*(p2-p1)+(p1-pa)).x()/vproj_ab.x();
                    if(coord_k<=0)
                    {
                        coord_p=1.0;
                    }

                    is_validated=true;

                }
                else   // p1-p2 and vproj_ab are disjoint
                {
                    is_validated=false;
                }

            }
        }
    }

    return is_validated;

}

/// Cross product for 3-elements vectors.
template<typename real>
inline real areaProduct(const Vec<3,real>& a, const Vec<3,real>& b)
{
    return Vec<3,real>(a.y()*b.z() - a.z()*b.y(),
            a.z()*b.x() - a.x()*b.z(),
            a.x()*b.y() - a.y()*b.x()).norm();
}

/// area from 2-elements vectors.
template <typename real>
inline real areaProduct(const defaulttype::Vec<2,real>& a, const defaulttype::Vec<2,real>& b )
{
    return a[0]*b[1] - a[1]*b[0];
}
/// area for 1-elements vectors.
template <typename real>
inline real areaProduct(const defaulttype::Vec<1,real>& , const defaulttype::Vec<1,real>&  )
{
    assert(false);
    return (real)0;
}

template< class DataTypes>
typename DataTypes::Real TriangleSetGeometryAlgorithms< DataTypes >::computeTriangleArea( const unsigned int i) const
{
    TriangleSetTopology< DataTypes > *topology = dynamic_cast<TriangleSetTopology< DataTypes >* >(this->m_basicTopology);
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
    TriangleSetTopology< DataTypes > *topology = dynamic_cast<TriangleSetTopology< DataTypes >* >(this->m_basicTopology);
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
    TriangleSetTopology< DataTypes > *topology = dynamic_cast<TriangleSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast< TriangleSetTopologyContainer* >(topology->getTopologyContainer());
    const std::vector<Triangle> &ta=container->getTriangleArray();
    const typename DataTypes::VecCoord& p = *topology->getDOF()->getX();
    unsigned int i;
    for (i=0; i<ta.size(); ++i)
    {
        const Triangle &t=ta[i];
        ai[i]=(Real)(areaProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]])/2.0);
    }
}


// Computes the intersection of the segment from point a to point b and the triangle indexed by t
template<class DataTypes>
bool TriangleSetGeometryAlgorithms< DataTypes >::computeSegmentTriangleIntersection(const Coord& a, const Coord& b, const unsigned int ind_t,
        std::vector<unsigned int> &indices,
        double &baryCoef, double& coord_kmin)
{

    // HYP : indices.size == 2

    bool is_validated = false;

    TriangleSetTopology< DataTypes > *topology = dynamic_cast<TriangleSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    TriangleSetTopologyContainer * container = static_cast< TriangleSetTopologyContainer* >(topology->getTopologyContainer());
    const Triangle &t=container->getTriangle(ind_t);
    const typename DataTypes::VecCoord& p = *topology->getDOF()->getX();

    const typename DataTypes::Coord& c0=p[t[0]];
    const typename DataTypes::Coord& c1=p[t[1]];
    const typename DataTypes::Coord& c2=p[t[2]];

    const Vec<3,Real> p0 (c0[0],c0[1],c0[2]);
    const Vec<3,Real> p1 (c1[0],c1[1],c1[2]);
    const Vec<3,Real> p2 (c2[0],c2[1],c2[2]);

    const Vec<3,Real> pa (a[0],a[1],a[2]);
    const Vec<3,Real> pb (b[0],b[1],b[2]);


    Vec<3,Real> v_normal = crossProduct(p2-p0,p1-p0);

    Real norm_v_normal = v_normal.norm();
    if(norm_v_normal != 0.0)
    {

        v_normal/=norm_v_normal;

        Vec<3,Real> v_ab = pb-pa;
        const Vec<3,Real> &vproj_ab = v_ab - dotProduct(v_ab,v_normal)*v_normal; // projection

        Real norm_v_ab = v_ab.norm();
        if(norm_v_ab != 0.0)
        {

            double init_coord=0.0;
            double& coord_p=init_coord;
            double& coord_k=init_coord;

            coord_kmin=init_coord;

            // Test of edge (p0,p1) :
            if(test_intersect(p0,p1,pa,vproj_ab,coord_p,coord_k))
            {
                is_validated=true;

                indices[0]=t[0];
                indices[1]=t[1];
                baryCoef=coord_p;
                coord_kmin=coord_k;
            }

            // Test of edge (p1,p2) :
            if(test_intersect(p1,p2,pa,vproj_ab,coord_p,coord_k))
            {
                is_validated=true;

                if(coord_k<coord_kmin)
                {
                    indices[0]=t[1];
                    indices[1]=t[2];
                    baryCoef=coord_p;
                    coord_kmin=coord_k;
                }
            }

            // Test of edge (p2,p0) :
            if(test_intersect(p1,p2,pa,vproj_ab,coord_p,coord_k))
            {
                is_validated=true;

                if(coord_k<coord_kmin)
                {

                    indices[0]=t[2];
                    indices[1]=t[0];
                    baryCoef=coord_p;
                    coord_kmin=coord_k;
                }
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


/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////TriangleSetTopology//////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
template<class DataTypes>
void TriangleSetTopology<DataTypes>::init()
{
}
template<class DataTypes>
TriangleSetTopology<DataTypes>::TriangleSetTopology(MechanicalObject<DataTypes> *obj) : EdgeSetTopology<DataTypes>( obj)
{
    this->m_topologyContainer= new TriangleSetTopologyContainer(this);
    this->m_topologyModifier= new TriangleSetTopologyModifier<DataTypes>(this);
    this->m_topologyAlgorithms= new TriangleSetTopologyAlgorithms<DataTypes>(this);
    this->m_geometryAlgorithms= new TriangleSetGeometryAlgorithms<DataTypes>(this);
}


} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_TriangleSetTOPOLOGY_INL
