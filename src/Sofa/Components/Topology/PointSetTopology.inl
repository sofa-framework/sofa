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
void PointSetTopologyModifier<TDataTypes>::swapVertices(const int i1,const int i2)
{
    PointSetTopologyContainer * container = dynamic_cast<PointSetTopologyContainer *>(m_basicTopology->getTopologyContainer());
    if (container != 0)
    {
        int tmp = container->vertexArray[i1];
        container->vertexArray[i1] = container->vertexArray[i2];
        container->vertexArray[i2] = tmp;
    }

    PointsIndicewsSwap *e = new PointsIndicesSwap(i1,i2);
    addTopologyChange(e);
}



template<class TDataTypes>
void PointSetTopologyModifier<TDataTypes>::addVertices(const unsigned int nVertices,
        VecCoord &X = (VecCoord &)0 )
{
    PointSetTopology *topology = dynamic_cast<PointSetTopology *>m_basicTopology;
    if (topology != 0)
    {
        PointSetTopologyContainer * container = dynamic_cast<PointSetTopologyContainer *>topology->getTopologyContainer());
        if (container != 0)
    {
        int prevSizeMechObj   = topology->object->getSize();
            int prevSizeContainer = container->vertexArray.size();

            // resizing the state vectors
            topology->object->resize( prevSizeMechObj + nVertices );

            // resizing the topology container vectors
            container->vertexArray.resize(prevSizeContainer + nVertices);
            container->vertexInSetArray.resize(prevSizeContainer + nVertices);

            // setting the new positions
            if (X != (VecCoord)0 )
            {
                for (int i = 0; i < nVertices; ++i)
                    topology->object->getX()[prevSizeMechObj + i] = X[i];
            }
            // TODO : same thing for other state vectors (X0, V and V0)

            // setting the new indices
            for (int i = 0; i < nVertices; ++i)
            {
                container->vertexArray[prevSizeContainer + i] = prevSizeMechObj + i;
                container->vertexInSetArray[prevSizeContainer + i] = true;
            }

            // Warning that vertices just got created
            PointsAdded *e = new PointsAdded(nVertices);
            addTopologyChange(e);

            //m_basicTopology->propagateTopologicalChanges();

        }
    }
}



template<class TDataTypes>
void PointSetTopologyModifier<TDataTypes>::removeVertices(const unsigned int nVertices, std::vector<int> &indices)
{
    PointSetTopology *topology = dynamic_cast<PointSetTopology *>m_basicTopology;
    if (topology != 0)
    {
        PointSetTopologyContainer * container = dynamic_cast<PointSetTopologyContainer *>topology->getTopologyContainer());
        if (container != 0)
    {
        // Warning that these vertices will be deleted
        PointsRemoved *e = new PointsRemoved(indices);
            addTopologyChange(e);

            //m_basicTopology->propagateTopologicalChanges();


            int prevSizeMechObj   = topology->object->getSize();
            int prevSizeContainer = container->vertexArray.size();
            int lastIndexMech = prevSizeMechObj - 1;
            int lastIndexTopo = prevSizeContainer - 1;

            // deletting the vertices
            for (int i = 0; i < nVertices; ++i)
            {
                //topology->object->getX()[ container->vertexArray()[indices[i]] ] = topology->object->getX()[lastIndexMech];
                topology->object->replaceValue(lastIndexMech, container->vertexArray()[indices[i]] );



                //container->vertexArray[indices[i]] = container->vertexArray[lastIndexTopo];
                //container->vertexInSetArray[prevSizeContainer + i] = container->vertexInSetArray[lastIndexTopo];
                --lastIndexMech;
                --lastIndexTopo;
            }

            // resizing the state vectors
            topology->object->resize( prevSizeMechObj - nVertices );

            // resizing the topology container vectors
            container->vertexArray.resize(prevSizeContainer - nVertices);
            container->vertexInSetArray.resize(prevSizeContainer - nVertices);

        }
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
    const std::vector<int> &va=ps->getVertexArray();
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
    const std::vector<int> &va=ps->getVertexArray();
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
void PointSetTopology<TDataTypes>::createNewVertices() const
{
    std::list<const TopologyChange *>::const_iterator it=firstChange();
    for (; it!=lastChange(); ++it)
    {
        if ((*it)->getChangeType()==VERTEXADDED_CODE)
        {
            PointsAdded *e=(PointsAdded *) (*it);
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
            PointsRemoved *e=(PointsRemoved *) (*it);
            const std::vector<int> &a=e->getArray();
            for (i = 0; i < e->removedVertexArray.size(); ++i)
            {
                object->replaceValue(object->getSize()-1-i,a[i]);
            }
            object->resize(object->getSize() - e->removedVertexArray.size());
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
        m_changeList.erase(m_changeList.begin(), m_changeList.end());
    }
}



template<class TDataTypes>
void PointSetTopology<TDataTypes>::init()
{
}


} // namespace Components

} // namespace Sofa

#endif
