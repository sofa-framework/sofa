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
template<class TDataTypes>
void PointSetTopologyModifier<TDataTypes>::swapPoints(const int i1,const int i2)
{
    PointSetTopology<TDataTypes> *topology = dynamic_cast<PointSetTopology<TDataTypes> *>m_basicTopology;
    if (topology != 0)
    {
        PointSetTopologyContainer * container = dynamic_cast<PointSetTopologyContainer *>topology->getTopologyContainer());
        if (container != 0)
    {
        topology->object->swapValues( container->getDOFIndex(i1), container->getDOFIndex(i2) );

            PointsIndicesSwap e( i1, i2 ); // Indices locaux ou globaux? (exemple de arretes)
            addTopologyChange(e);
        }
    }
}



template<class TDataTypes>
void PointSetTopologyModifier<TDataTypes>::addPointsProcess(const unsigned int nPoints,
        VecCoord &X  = (VecCoord &)0,
        VecCoord &X0 = (VecCoord &)0,
        VecDeriv &V  = (VecDeriv &)0,
        VecDeriv &V0 = (VecDeriv &)0,
        VecDeriv &F  = (VecDeriv &)0,
        VecDeriv &DX = (VecDeriv &)0
                                                           )
{
    PointSetTopology<TDataTypes> *topology = dynamic_cast<PointSetTopology<TDataTypes> *>m_basicTopology;
    if (topology != 0)
    {
        PointSetTopologyContainer * container = dynamic_cast<PointSetTopologyContainer *>topology->getTopologyContainer());
        if (container != 0)
    {
        unsigned int prevSizeMechObj   = topology->object->getSize();
            unsigned int prevSizeContainer = container->getDOFIndexArray().size();

            // resizing the state vectors
            topology->object->resize( prevSizeMechObj + nPoints );

            // setting the new positions
            if (X != (VecCoord)0 )
            {
                for (unsigned int i = 0; i < nPoints; ++i)
                    topology->object->getX()[prevSizeMechObj + i] = X[i];
            }
            if (X0 != (VecCoord)0 )
            {
                for (unsigned int i = 0; i < nPoints; ++i)
                    topology->object->getX0()[prevSizeMechObj + i] = X0[i];
            }
            if (V != (VecDeriv)0 )
            {
                for (unsigned int i = 0; i < nPoints; ++i)
                    topology->object->getV()[prevSizeMechObj + i] = V[i];
            }
            if (V0 != (VecDeriv)0 )
            {
                for (unsigned int i = 0; i < nPoints; ++i)
                    topology->object->getV0()[prevSizeMechObj + i] = V0[i];
            }
            if (F != (VecDeriv)0 )
            {
                for (unsigned int i = 0; i < nPoints; ++i)
                    topology->object->getF()[prevSizeMechObj + i] = F[i];
            }
            if (DX != (VecDeriv)0 )
            {
                for (unsigned int i = 0; i < nPoints; ++i)
                    topology->object->getDx()[prevSizeMechObj + i] = DX[i];
            }

            // setting the new indices
            std::vector<unsigned int> DOFIndex = container->getDOFIndexArray();
            DOFIndex.resize(prevSizeContainer + nPoints);
            for (unsigned int i = 0; i < nPoints; ++i)
            {
                DOFIndex[prevSizeContainer + i] = prevSizeMechObj + i;
            }

        }
    }
}


template<class TDataTypes>
void PointSetTopologyModifier<TDataTypes>::addPointsWarning(const unsigned int nPoints)
{
    // Warning that vertices just got created
    PointsAdded e(nPoints);
    addTopologyChange(e);
}




template<class TDataTypes>
void PointSetTopologyModifier<TDataTypes>::removePointsWarning(const unsigned int nPoints, std::vector<unsigned int> &indices)
{
    // Warning that these vertices will be deleted
    PointsRemoved e(indices);
    addTopologyChange(e);
}



template<class TDataTypes>
void PointSetTopologyModifier<TDataTypes>::removePointsProcess(const unsigned int nPoints, std::vector<unsigned int> &indices)
{
    PointSetTopology<TDataTypes> *topology = dynamic_cast<PointSetTopology<TDataTypes> *>m_basicTopology;
    if (topology != 0)
    {
        PointSetTopologyContainer * container = dynamic_cast<PointSetTopologyContainer *>topology->getTopologyContainer());
        if (container != 0)
    {

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
            topology->object->resize( prevSizeMechObj - nVertices );

            // resizing the topology container vectors
            container->getDOFIndexArray().resize(prevDOFIndexArraySize - nVertices);
            container->getPointSetIndexArray().resize(prevPointSetIndexArraySize - nVertices);

        }
    }
}



template<class TDataTypes>
void PointSetTopologyModifier<TDataTypes>::renumberPointsProcess( std::vector<unsigned int> &index)
{

    PointSetTopology<TDataTypes> *topology = dynamic_cast<PointSetTopology<TDataTypes> *>m_basicTopology;
    if (topology != 0)
    {
        topology->object->renumberValues( index );
    }
}



/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////PointSetGeometryAlgorithms///////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
template <class TDataTypes>
typename TDataTypes::Coord PointSetGeometryAlgorithms<TDataTypes>::getPointSetCenter() const
{
    typename TDataTypes::Coord center;
    // get restPosition
    PointSetTopology<TDataTypes> *parent=static_cast<PointSetTopology<TDataTypes> *>(m_basicTopology);
    typename TDataTypes::VecCoord& p = *parent->getDOF()->getX();
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



template<class TDataTypes>
void  PointSetGeometryAlgorithms<TDataTypes>::getEnclosingSphere(typename TDataTypes::Coord &center,
        typename TDataTypes::Real &radius) const
{
    Coord dp;
    Real val;
    // get restPosition
    PointSetTopology<TDataTypes> *parent=static_cast<PointSetTopology<TDataTypes> *>(m_basicTopology);
    typename TDataTypes::VecCoord& p = *parent->getDOF()->getX();
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



template<class TDataTypes>
void  PointSetGeometryAlgorithms<TDataTypes>::getAABB(typename TDataTypes::Real bb[6] ) const
{
}



/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////PointSetTopology/////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template<class TDataTypes>
PointSetTopology<TDataTypes>::PointSetTopology(MechanicalObject<TDataTypes> *obj) : object(obj)
{
}


template<class TDataTypes>
void PointSetTopology<TDataTypes>::propagateTopologicalChanges()
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



template<class TDataTypes>
void PointSetTopology<TDataTypes>::init()
{
}


} // namespace Components

} // namespace Sofa

#endif
