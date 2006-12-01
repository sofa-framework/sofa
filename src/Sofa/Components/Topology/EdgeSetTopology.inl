#ifndef SOFA_COMPONENTS_EDGESETTOPOLOGY_INL
#define SOFA_COMPONENTS_EDGESETTOPOLOGY_INL

#include "EdgeSetTopology.h"
#include "TopologyChangedEvent.h"
#include <Sofa/Components/Graph/PropagateEventAction.h>
#include <Sofa/Components/Graph/GNode.h>
#include <algorithm>

namespace Sofa
{
namespace Components
{

using namespace Common;
using namespace Sofa::Core;



/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////EdgeSetTopologyModifier/////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
template<class TDataTypes>
void EdgeSetTopologyModifier<TDataTypes>::addEdgesProcess(const std::vector< Edge > &edges)
{
    EdgeSetTopology<TDataTypes> *topology = dynamic_cast<EdgeSetTopology<TDataTypes> *>m_basicTopology;
    if (topology != 0)
    {
        EdgeSetTopologyContainer * container = dynamic_cast<EdgeSetTopologyContainer *>topology->getTopologyContainer());
        if (container != 0)
    {
        for (unsigned int i = 0; i < edges.size(); ++i)
            {
                Edge &e = edges[i];
                container->m_edge.push_back(e);
                container->getEdgeShell( e.first ).push_back( container->m_edge.size() - 1 );
                container->getEdgeShell( e.second ).push_back( container->m_edge.size() - 1 );
            }
        }
    }
}



template<class TDataTypes>
void EdgeSetTopologyModifier<TDataTypes>::addEdgesProcess(const std::vector< std::vector< Edge > > &ancestors)
{
    EdgeSetTopology<TDataTypes> *topology = dynamic_cast<EdgeSetTopology<TDataTypes> *>m_basicTopology;
    if (topology != 0)
    {
        EdgeSetTopologyContainer * container = dynamic_cast<EdgeSetTopologyContainer *>topology->getTopologyContainer());
        if (container != 0)
    {
        for (unsigned int i = 0; i < edges.size(); ++i)
            {
                // TODO
            }
        }
    }
}



template<class TDataTypes>
void EdgeSetTopologyModifier<TDataTypes>::addEdgesWarning(const unsigned int nEdges, const std::vector< std::vector< Edge > > &ancestors = (std::vector< std::vector< Edge > > )0)
{
    // Warning that edges just got created
    EdgesAdded e(nEdges, ancestors);
    addTopologyChange(e);
}




template<class TDataTypes>
void EdgeSetTopologyModifier<TDataTypes>::removeEdgesWarning( const std::vector<unsigned int> &edges )
{
    // Warning that these edges will be deleted
    EdgesRemoved e(indices);
    addTopologyChange(e);
}



template<class TDataTypes>
void EdgeSetTopologyModifier<TDataTypes>::removeEdgesProcess(const unsigned int nEdges, const std::vector<unsigned int> &indices)
{
    EdgeSetTopology<TDataTypes> *topology = dynamic_cast<EdgeSetTopology<TDataTypes> *>m_basicTopology;
    if (topology != 0)
    {
        EdgeSetTopologyContainer * container = dynamic_cast<EdgeSetTopologyContainer *>topology->getTopologyContainer());
        if (container != 0)
    {
        for (unsigned int i = 0; i < indices.size(); ++i)
            {
                Edge *e = &container->m_edge[ indices[i] ];
                unsigned int point1 = e->first, point2 = e->second;



            }
        }
    }
}



template<class TDataTypes>
void EdgeSetTopologyModifier<TDataTypes>::renumberEdgesProcess( const std::vector<unsigned int> &index )
{

    EdgeSetTopology<TDataTypes> *topology = dynamic_cast<EdgeSetTopology<TDataTypes> *>m_basicTopology;
    if (topology != 0)
    {
        topology->object->renumberValues( index );
    }
}



} // namespace Components

} // namespace Sofa

#endif // SOFA_COMPONENTS_EDGESETTOPOLOGY_INL
