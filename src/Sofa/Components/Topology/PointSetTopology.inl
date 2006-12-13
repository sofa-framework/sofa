#ifndef SOFA_COMPONENTS_POINTSETTOPOLOGY_INL
#define SOFA_COMPONENTS_POINTSETTOPOLOGY_INL


#include "PointSetTopology.h"
#include "TopologyChangedEvent.h"
#include <Sofa/Components/Graph/PropagateEventAction.h>
#include <Sofa/Components/Graph/GNode.h>

namespace Sofa
{
namespace Components
{

using namespace Common;
using namespace Sofa::Core;



/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////PointSetTopologyModifier/////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::swapPoints(const int i1,const int i2)
{
    PointSetTopology<DataTypes> *topology = dynamic_cast<PointSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);
    PointSetTopologyContainer * container = dynamic_cast<PointSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    topology->object->swapValues( container->getDOFIndex(i1), container->getDOFIndex(i2) );

    PointsIndicesSwap e( i1, i2 ); // Indices locaux ou globaux? (exemple de arretes)
    addTopologyChange(e);
}




template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::addPointsProcess(const unsigned int nPoints,
        const std::vector< std::vector< unsigned int > >& ancestors,
        const std::vector< std::vector< double > >& baryCoefs)
{
    PointSetTopology<DataTypes> *topology = dynamic_cast<PointSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);
    PointSetTopologyContainer * container = dynamic_cast<PointSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    unsigned int prevSizeMechObj   = topology->object->getSize();
    unsigned int prevSizeContainer = container->getDOFIndexArray().size();

    // resizing the state vectors
    topology->object->resize( prevSizeMechObj + nPoints );

    if ( ancestors != (const std::vector< std::vector< unsigned int > >)0 )
    {
        assert( baryCoefs == (const std::vector< std::vector< double > >)0 || ancestors.size() == baryCoefs.size() );

        std::vector< std::vector< double > > coefs;

        for (unsigned int i = 0; i < ancestors.size(); ++i)
        {
            assert( baryCoefs == (const std::vector< std::vector< double > >)0 || baryCoefs[i].size() == 0 || ancestors[i].size() == baryCoefs[i].size() );

            for (unsigned int j = 0; j < ancestors[i].size(); ++j)
            {
                // constructng default coefs if none were defined
                if (baryCoefs == (const std::vector< std::vector< double > >)0 || baryCoefs[i].size() == 0)
                    coefs[i][j] = 1.0f / ancestors[i].size();
                else
                    coefs[i][j] = baryCoefs[i][j];
            }
        }

        for ( unsigned int i = 0; i < nPoints; ++i)
        {
            topology->object->computeWeightedValue( prevSizeMechObj + i, ancestors[i], coefs[i] );
        }
    }

    // setting the new indices
    std::vector<unsigned int> DOFIndex = container->getDOFIndexArray();
    DOFIndex.resize(prevSizeContainer + nPoints);
    for (unsigned int i = 0; i < nPoints; ++i)
    {
        DOFIndex[prevSizeContainer + i] = prevSizeMechObj + i;
    }

    //invalidating PointSetIndex, since it is no longer up-to-date
    container->getPointSetIndexArray().resize(0);

}



template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::addPointsWarning(const unsigned int nPoints,
        const std::vector< std::vector< unsigned int > > &ancestors,
        const std::vector< std::vector< double       > >& coefs)
{
    // Warning that vertices just got created
    PointsAdded e(nPoints, ancestors, coefs);
    addTopologyChange(e);
}




template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::removePointsWarning(const unsigned int nPoints, const std::vector<unsigned int> &indices)
{
    // Warning that these vertices will be deleted
    PointsRemoved e(indices);
    addTopologyChange(e);
}



template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::removePointsProcess(const unsigned int nPoints, std::vector<unsigned int> &indices)
{
    std::sort( indices.begin(), indices.end(), std::greater<unsigned int>() );
    PointSetTopology<DataTypes> *topology = dynamic_cast<PointSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);
    PointSetTopologyContainer * container = dynamic_cast<PointSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    int prevSizeMechObj   = topology->object->getSize();
    unsigned int prevDOFIndexArraySize = container->getDOFIndexArray().size();
    int prevPointSetIndexArraySize = container->getPointSetIndexArray().size();

    int lastIndexMech = prevSizeMechObj - 1;

    // deletting the vertices
    for (unsigned int i = 0; i < nPoints; ++i)
    {
        topology->object->replaceValue(lastIndexMech, container->getDOFIndex(indices[i]) );

        --lastIndexMech;
    }

    // resizing the state vectors
    topology->object->resize( prevSizeMechObj - this->nVertices );

    // resizing the topology container vectors
    container->getDOFIndexArray().resize(prevDOFIndexArraySize - this->nVertices);
    container->getPointSetIndexArray().resize(prevPointSetIndexArraySize - this->nVertices);
}



template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::renumberPointsProcess( const std::vector<unsigned int> &index)
{

    PointSetTopology<DataTypes> *topology = dynamic_cast<PointSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);
    topology->object->renumberValues( index );
}



/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////PointSetGeometryAlgorithms///////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataTypes>
typename DataTypes::Coord PointSetGeometryAlgorithms<DataTypes>::getPointSetCenter() const
{
    typename DataTypes::Coord center;
    // get restPosition
    PointSetTopology<DataTypes> *parent=static_cast<PointSetTopology<DataTypes> *>(m_basicTopology);
    typename DataTypes::VecCoord& p = *parent->getDOF()->getX();
    PointSetTopologyContainer *ps=static_cast<PointSetTopologyContainer *>( parent->getTopologyContainer());
    const std::vector<unsigned int> &va=ps->getDOFIndexArray();
    unsigned int i;
    for(i=0; i<ps->getNumberOfVertices(); i++)
    {
        center+=p[va[i]];
    }
    center/= (ps->getNumberOfVertices());
    return center;
}



template<class DataTypes>
void  PointSetGeometryAlgorithms<DataTypes>::getEnclosingSphere(typename DataTypes::Coord &center,
        typename DataTypes::Real &radius) const
{
    Coord dp;
    Real val;
    // get restPosition
    PointSetTopology<DataTypes> *parent=static_cast<PointSetTopology<DataTypes> *>(m_basicTopology);
    typename DataTypes::VecCoord& p = *parent->getDOF()->getX();
    PointSetTopologyContainer *ps=static_cast<PointSetTopologyContainer *>( parent->getTopologyContainer());
    const std::vector<unsigned int> &va=ps->getDOFIndexArray();
    unsigned int i;
    for(i=0; i<ps->getNumberOfVertices(); i++)
    {
        center+=p[va[i]];
    }
    center/= (ps->getNumberOfVertices());
    dp=center-p[0];
    radius=dot(dp,dp);
    for(i=1; i<ps->getNumberOfVertices(); i++)
    {
        dp=center-p[va[i]];
        val=dot(dp,dp);
        if (val<radius)
            radius=val;
    }
    radius=(Real)sqrt((double) radius);
}



template<class DataTypes>
void  PointSetGeometryAlgorithms<DataTypes>::getAABB(typename DataTypes::Real /*bb*/[6] ) const
{
}



/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////PointSetTopology/////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template<class DataTypes>
PointSetTopology<DataTypes>::PointSetTopology(MechanicalObject<DataTypes> *obj) : object(obj)
{
}


template<class DataTypes>
void PointSetTopology<DataTypes>::propagateTopologicalChanges()
{
    Sofa::Components::TopologyChangedEvent topEvent((BasicTopology *)this);
    Sofa::Components::Graph::PropagateEventAction propKey( &topEvent );
    Sofa::Components::Graph::GNode *groot=dynamic_cast<Sofa::Components::Graph::GNode *>(this->getContext());
    if (groot)
    {
        groot->execute(propKey);
        /// remove list of events
        m_topologyContainer->getChangeList().erase(m_topologyContainer->getChangeList().begin(), m_topologyContainer->getChangeList().end());
    }
}



template<class DataTypes>
void PointSetTopology<DataTypes>::init()
{
}


} // namespace Components

} // namespace Sofa

#endif
