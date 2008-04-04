#include <sofa/component/topology/EdgeSetTopology.h>
#include <sofa/component/topology/EdgeSetTopology.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>

// Use BOOST GRAPH LIBRARY :

#include <boost/config.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/bandwidth.hpp>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(EdgeSetTopology)


template class EdgeSetTopology<Vec3dTypes>;
template class EdgeSetTopology<Vec3fTypes>;
template class EdgeSetTopology<Vec2dTypes>;
template class EdgeSetTopology<Vec2fTypes>;
template class EdgeSetTopology<Vec1dTypes>;
template class EdgeSetTopology<Vec1fTypes>;
template class EdgeSetTopology<Rigid3dTypes>;
template class EdgeSetTopology<Rigid3fTypes>;
template class EdgeSetTopology<Rigid2dTypes>;
template class EdgeSetTopology<Rigid2fTypes>;

template class EdgeSetTopologyAlgorithms<Vec3fTypes>;
template class EdgeSetTopologyAlgorithms<Vec3dTypes>;
template class EdgeSetTopologyAlgorithms<Vec2dTypes>;
template class EdgeSetTopologyAlgorithms<Vec2fTypes>;
template class EdgeSetTopologyAlgorithms<Vec1dTypes>;
template class EdgeSetTopologyAlgorithms<Vec1fTypes>;
template class EdgeSetTopologyAlgorithms<Rigid3dTypes>;
template class EdgeSetTopologyAlgorithms<Rigid3fTypes>;
template class EdgeSetTopologyAlgorithms<Rigid2dTypes>;
template class EdgeSetTopologyAlgorithms<Rigid2fTypes>;

template class EdgeSetGeometryAlgorithms<Vec3fTypes>;
template class EdgeSetGeometryAlgorithms<Vec3dTypes>;
template class EdgeSetGeometryAlgorithms<Vec2dTypes>;
template class EdgeSetGeometryAlgorithms<Vec2fTypes>;
template class EdgeSetGeometryAlgorithms<Vec1dTypes>;
template class EdgeSetGeometryAlgorithms<Vec1fTypes>;
template class EdgeSetGeometryAlgorithms<Rigid3dTypes>;
template class EdgeSetGeometryAlgorithms<Rigid3fTypes>;
template class EdgeSetGeometryAlgorithms<Rigid2dTypes>;
template class EdgeSetGeometryAlgorithms<Rigid2fTypes>;

template class EdgeSetTopologyModifier<Vec3fTypes>;
template class EdgeSetTopologyModifier<Vec3dTypes>;
template class EdgeSetTopologyModifier<Vec2fTypes>;
template class EdgeSetTopologyModifier<Vec2dTypes>;
template class EdgeSetTopologyModifier<Vec1fTypes>;
template class EdgeSetTopologyModifier<Vec1dTypes>;

// implementation EdgeSetTopologyContainer

void EdgeSetTopologyContainer::createEdgeVertexShellArray ()
{
    m_edgeVertexShell.resize( m_basicTopology->getDOFNumber() );

    for (unsigned int i = 0; i < m_edge.size(); ++i)
    {
        // adding edge i in the edge shell of both points
        m_edgeVertexShell[ m_edge[i][0]  ].push_back( i );
        m_edgeVertexShell[ m_edge[i][1] ].push_back( i );
    }
}



const sofa::helper::vector<Edge> &EdgeSetTopologyContainer::getEdgeArray()
{
    if (!m_edge.size())
        createEdgeSetArray();
    return m_edge;
}


int EdgeSetTopologyContainer::getEdgeIndex(const unsigned int v1, const unsigned int v2)
{
    const sofa::helper::vector< unsigned int > &es1=getEdgeVertexShell(v1) ;
    const sofa::helper::vector<Edge> &ea=getEdgeArray();
    unsigned int i=0;
    int result= -1;
    while ((i<es1.size()) && (result== -1))
    {
        const Edge &e=ea[es1[i]];
        if ((e[0]==v2)|| (e[1]==v2))
            result=(int) es1[i];

        i++;
    }
    return result;
}

const Edge &EdgeSetTopologyContainer::getEdge(const unsigned int i)
{
    if (!m_edge.size())
        createEdgeSetArray();
    return m_edge[i];
}

// Return the number of connected components from the graph containing all edges and give, for each vertex, which component it belongs to  (use BOOST GRAPH LIBRAIRY)
int EdgeSetTopologyContainer::getNumberConnectedComponents(sofa::helper::vector<int>& components)
{
    using namespace boost;

    typedef adjacency_list <vecS, vecS, undirectedS> Graph;

    Graph G;

    const sofa::helper::vector<Edge> &ea=getEdgeArray();
    for (unsigned int k=0; k<ea.size(); ++k)
    {
        add_edge(ea[k][0], ea[k][1], G);
    }

    components.resize(num_vertices(G));
    int num = connected_components(G, &components[0]);

    //sofa::helper::vector<int>::size_type i;
    std::cout << "Total number of components: " << num << endl;
    //for (i = 0; i != components.size(); ++i)
    //  std::cout << "Vertex " << i <<" is in component " << components[i] << endl;
    std::cout << endl;

    return num;
}



bool EdgeSetTopologyContainer::checkTopology() const
{
    //std::cout << "*** CHECK EdgeSetTopologyContainer ***" << std::endl;

    PointSetTopologyContainer::checkTopology();
    if (m_edgeVertexShell.size()>0)
    {
        unsigned int i,j;
        for (i=0; i<m_edgeVertexShell.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &es=m_edgeVertexShell[i];

            for (j=0; j<es.size(); ++j)
            {
                bool check_edge_vertex_shell = (m_edge[es[j]][0]==i) ||  (m_edge[es[j]][1]==i);
                if(!check_edge_vertex_shell)
                {
                    std::cout << "*** CHECK FAILED : check_edge_vertex_shell, i = " << i << " , j = " << j << std::endl;
                }
                assert(check_edge_vertex_shell);
            }
        }
        //std::cout << "******** DONE : check_edge_vertex_shell" << std::endl;
    }
    return true;
}


unsigned int EdgeSetTopologyContainer::getNumberOfEdges()
{
    if (!m_edge.size())
        createEdgeSetArray();
    return m_edge.size();
}



const sofa::helper::vector< sofa::helper::vector<unsigned int> > &EdgeSetTopologyContainer::getEdgeVertexShellArray()
{
    if (!m_edgeVertexShell.size())
        createEdgeVertexShellArray();
    return m_edgeVertexShell;
}






const sofa::helper::vector< unsigned int > &EdgeSetTopologyContainer::getEdgeVertexShell(const unsigned int i)
{
    if (!m_edgeVertexShell.size())
        createEdgeVertexShellArray();
    return m_edgeVertexShell[i];
}


sofa::helper::vector< unsigned int > &EdgeSetTopologyContainer::getEdgeVertexShellForModification(const unsigned int i)
{
    if (!m_edgeVertexShell.size())
        createEdgeVertexShellArray();
    return m_edgeVertexShell[i];
}



/*EdgeSetTopologyContainer::EdgeSetTopologyContainer(core::componentmodel::topology::BaseTopology *top) : PointSetTopologyContainer( top )
{

}
*/

// Give the optimal vertex permutation according to the Reverse CuthillMckee algorithm (use BOOST GRAPH LIBRAIRY)
template<class DataTypes>
void EdgeSetTopologyAlgorithms< DataTypes >::resortCuthillMckee(sofa::helper::vector<int>& inverse_permutation)
{

    using namespace boost;
    using namespace std;
    typedef adjacency_list<vecS, vecS, undirectedS,
            property<vertex_color_t, default_color_type,
            property<vertex_degree_t,int> > > Graph;
    typedef graph_traits<Graph>::vertex_descriptor Vertex;
    typedef graph_traits<Graph>::vertices_size_type size_type;

    Graph G;

    EdgeSetTopology< DataTypes > *topology = dynamic_cast<EdgeSetTopology< DataTypes >* >(this->m_basicTopology);
    assert (topology != 0);
    EdgeSetTopologyContainer * container = static_cast< EdgeSetTopologyContainer* >(topology->getTopologyContainer());

    const sofa::helper::vector<Edge> &ea=container->getEdgeArray();
    if(ea.size()>0)
    {

        for (unsigned int k=0; k<ea.size(); ++k)
        {
            add_edge(ea[k][0], ea[k][1], G);
        }

        inverse_permutation.resize(num_vertices(G));

        //graph_traits<Graph>::vertex_iterator ui, ui_end;

        //property_map<Graph,vertex_degree_t>::type deg = get(vertex_degree, G);
        //for (boost::tie(ui, ui_end) = vertices(G); ui != ui_end; ++ui)
        //  deg[*ui] = degree(*ui, G);

        property_map<Graph, vertex_index_t>::type
        index_map = get(vertex_index, G);

        std::cout << "original bandwidth: " << bandwidth(G) << std::endl;

        std::vector<Vertex> inv_perm(num_vertices(G));
        std::vector<size_type> perm(num_vertices(G));

        //reverse cuthill_mckee_ordering
        cuthill_mckee_ordering(G, inv_perm.rbegin());

        //std::cout << "Reverse Cuthill-McKee ordering:" << endl;
        //std::cout << "  ";
        unsigned int ind_i = 0;
        for (std::vector<Vertex>::const_iterator i=inv_perm.begin();
                i != inv_perm.end(); ++i)
        {
            //std::cout << index_map[*i] << " ";
            inverse_permutation[ind_i]=index_map[*i];
            ind_i++;
        }
        //std::cout << endl;

        for (size_type c = 0; c != inv_perm.size(); ++c)
            perm[index_map[inv_perm[c]]] = c;
        std::cout << "  bandwidth: "
                << bandwidth(G, make_iterator_property_map(&perm[0], index_map, perm[0]))
                << std::endl;
    }

}


EdgeSetTopologyContainer::EdgeSetTopologyContainer(core::componentmodel::topology::BaseTopology *top, /*const sofa::helper::vector< unsigned int > &DOFIndex, */
        const sofa::helper::vector< Edge >         &edges )
    : PointSetTopologyContainer( top /*, DOFIndex*/ ), m_edge( edges )
{

}



// factory related stuff

int EdgeSetTopologyClass = core::RegisterObject("Dynamic topology handling point sets")
        .add< EdgeSetTopology<Vec3dTypes> >()
        .add< EdgeSetTopology<Vec3fTypes> >()
        .add< EdgeSetTopology<Vec2dTypes> >()
        .add< EdgeSetTopology<Vec2fTypes> >()
        .add< EdgeSetTopology<Vec1dTypes> >()
        .add< EdgeSetTopology<Vec1fTypes> >()
        .add< EdgeSetTopology<Rigid3dTypes> >()
        .add< EdgeSetTopology<Rigid3fTypes> >()
        .add< EdgeSetTopology<Rigid2dTypes> >()
        .add< EdgeSetTopology<Rigid2fTypes> >()
        ;


} // namespace topology

} // namespace component

} // namespace sofa

