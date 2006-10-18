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

template<class TDataTypes>
void PointSetTopologyModifier<TDataTypes>::swapVertices(const indexType i1,const indexType i2)
{
    VertexSwap *e = new VertexSwap(i1,i2);
    addTopologyChange(e);
}
template<class TDataTypes>
void PointSetTopologyModifier<TDataTypes>::addVertices(const unsigned int nVertices)
{
    VertexAdded *e = new VertexAdded(nVertices);
    addTopologyChange(e);
}
template<class TDataTypes>
void PointSetTopologyModifier<TDataTypes>::removeVertices(const unsigned int nVertices)
{
    VertexRemoved *e = new VertexRemoved(nVertices);
    addTopologyChange(e);
}

template <class TDataTypes>
typename TDataTypes::Coord PointSetGeometryAlgorithms<TDataTypes>::getPointSetCenter() const
{
    typename TDataTypes::Coord center;
    // get restPosition
    PointSetTopology<TDataTypes> *parent=static_cast<PointSetTopology<TDataTypes> *>(topology);
    typename TDataTypes::VecCoord& p = *parent->getDOF()->getX();
    PointSetTopologyContainer *ps=static_cast<PointSetTopologyContainer *>( parent->getTopologyContainer());
    const PointSetTopologyContainer::VertexArray &va=ps->getVertexArray();
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
    PointSetTopology<TDataTypes> *parent=static_cast<PointSetTopology<TDataTypes> *>(topology);
    typename TDataTypes::VecCoord& p = *parent->getDOF()->getX();
    PointSetTopologyContainer *ps=static_cast<PointSetTopologyContainer *>( parent->getTopologyContainer());
    const PointSetTopologyContainer::VertexArray &va=ps->getVertexArray();
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


template<class TDataTypes>
PointSetTopology<TDataTypes>::PointSetTopology(MechanicalObject<TDataTypes> *obj) : object(obj)
{
}

template<class TDataTypes>
void PointSetTopology<TDataTypes>::createNewVertices() const
{
    std::list<const TopologyChange *>::const_iterator it=firstChange();
    for (; it!=lastChange(); ++it)
    {
        if ((*it)->getChangeType()==VERTEXADDED_CODE)
        {
            VertexAdded *e=(VertexAdded *) (*it);
            object->resize(object->getSize()+e->getNbAddedVertices());
        }
    }
}
template<class TDataTypes>
void PointSetTopology<TDataTypes>::removeVertices() const
{
    std::list<const TopologyChange *>::const_iterator it=firstChange();
    unsigned int i;
    for (; it!=lastChange(); ++it)
    {
        if ((*it)->getChangeType()==VERTEXREMOVED_CODE)
        {
            VertexRemoved *e=(VertexRemoved *) (*it);
            const std::vector<VertexRemoved::index_type> &a=e->getArray();
            for (i=0; i<e->getNbRemovedVertices(); ++i)
            {
                object->replaceValue(object->getSize()-1-i,a[i]);
            }
            object->resize(object->getSize()-e->getNbRemovedVertices());
        }
    }
}

template<class TDataTypes>
void PointSetTopology<TDataTypes>::propagateTopologicalChanges()
{
    Sofa::Components::TopologyChangedEvent topEvent((BasicTopology *)this);
    Sofa::Components::Graph::PropagateEventAction propKey( &topEvent );
    Sofa::Components::Graph::GNode *groot=dynamic_cast<Sofa::Components::Graph::GNode *>(this->getContext());
    if (groot)
    {
        // createNewVertices();
        groot->execute(propKey);
        //	removeVertices();
        /// remove list of events
        changeList.erase(changeList.begin(),changeList.end());
    }
}

template<class TDataTypes>
void PointSetTopology<TDataTypes>::init()
{
}


} // namespace Components

} // namespace Sofa

#endif
