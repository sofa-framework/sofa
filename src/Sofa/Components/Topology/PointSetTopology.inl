#ifndef SOFA_COMPONENTS_POINTSETTOPOLOGY_INL
#define SOFA_COMPONENTS_POINTSETTOPOLOGY_INL


#include "PointSetTopology.h"
#include "TopologyChangedEvent.h"
#include <Sofa/Components/Graph/PropagateEventAction.h>
#include <Sofa/Components/Graph/GNode.h>
#include <Sofa/Components/MeshTopologyLoader.h>

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
class PointSetTopologyLoader : public MeshTopologyLoader
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;

    VecCoord pointArray;

    PointSetTopologyLoader()
    {
    }
    virtual void addPoint(double px, double py, double pz)
    {
        pointArray.push_back(Coord((Real)px,(Real)py,(Real)pz));
    }
    virtual void addLine(int p1, int p2)
    {
    }
    virtual void addTriangle(int p1, int p2, int p3)
    {
    }
    virtual void addQuad(int p1, int p2, int p3, int p4)
    {
    }
    virtual void addTetra(int p1, int p2, int p3, int p4)
    {
    }
    virtual void addCube(int p1, int p2, int p3, int p4, int p5, int p6, int p7, int p8)
    {
    }
};
template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::loadPointSet(PointSetTopologyLoader<DataTypes> *loader)
{
    PointSetTopology<DataTypes> *topology = dynamic_cast<PointSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);
    PointSetTopologyContainer * container = static_cast<PointSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    /// resize the DOF stored in the mechanical object
    topology->object->resize(loader->pointArray.size());
    /// resize the point set container
    std::vector<unsigned int> DOFIndex = container->getDOFIndexArray();
    DOFIndex.resize(loader->pointArray.size());
    /// store position and vertex index in containers
    unsigned int index;
    for (index=0; index<loader->pointArray.size(); ++index)
    {
        (*topology->object->getX())[index]=loader->pointArray[index];
        DOFIndex[index] = index;
    }
}
template<class DataTypes>
bool PointSetTopologyModifier<DataTypes>::load(const char *filename)
{

    PointSetTopologyLoader<DataTypes> loader;
    if (!loader.load(filename))
        return false;
    else
    {
        loadPointSet(&loader);
        return true;
    }
}


template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::swapPoints(const int i1,const int i2)
{
    PointSetTopology<DataTypes> *topology = dynamic_cast<PointSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);
    PointSetTopologyContainer * container = static_cast<PointSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    topology->object->swapValues( container->getDOFIndex(i1), container->getDOFIndex(i2) );

    PointsIndicesSwap *e=new PointsIndicesSwap( i1, i2 ); // Indices locaux ou globaux? (exemple de arretes)
    addTopologyChange(e);
}




template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::addPointsProcess(const unsigned int nPoints,
        const std::vector< std::vector< unsigned int > >& ancestors,
        const std::vector< std::vector< double > >& baryCoefs)
{
    PointSetTopology<DataTypes> *topology = dynamic_cast<PointSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);
    PointSetTopologyContainer * container = static_cast<PointSetTopologyContainer *>(topology->getTopologyContainer());
    assert (container != 0);
    unsigned int prevSizeMechObj   = topology->object->getSize();
    unsigned int prevSizeContainer = container->getDOFIndexArray().size();

    // resizing the state vectors
    topology->object->resize( prevSizeMechObj + nPoints );

    if ( ancestors != (const std::vector< std::vector< unsigned int > >)0 )
    {
        assert( baryCoefs == (const std::vector< std::vector< double > >)0 || ancestors.size() == baryCoefs.size() );

        std::vector< std::vector< double > > coefs;
        coefs.resize(ancestors.size());

        for (unsigned int i = 0; i < ancestors.size(); ++i)
        {
            assert( baryCoefs == (const std::vector< std::vector< double > >)0 || baryCoefs[i].size() == 0 || ancestors[i].size() == baryCoefs[i].size() );
            coefs[i].resize(ancestors[i].size());


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
    assert(container->getPointSetIndexSize()==0);

}



template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::addPointsWarning(const unsigned int nPoints,
        const std::vector< std::vector< unsigned int > > &ancestors,
        const std::vector< std::vector< double       > >& coefs)
{
    // Warning that vertices just got created
    PointsAdded *e=new PointsAdded(nPoints, ancestors, coefs);
    addTopologyChange(e);
}




template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::removePointsWarning(const unsigned int nPoints, const std::vector<unsigned int> &indices)
{
    // Warning that these vertices will be deleted
    PointsRemoved *e=PointsRemoved(indices);
    addTopologyChange(e);
}



template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::removePointsProcess(const unsigned int nPoints, std::vector<unsigned int> &indices)
{
    std::sort( indices.begin(), indices.end(), std::greater<unsigned int>() );
    PointSetTopology<DataTypes> *topology = dynamic_cast<PointSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);
    PointSetTopologyContainer * container = static_cast<PointSetTopologyContainer *>(topology->getTopologyContainer());
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
    topology->object->resize( prevSizeMechObj - nPoints );

    // resizing the topology container vectors
    if (prevDOFIndexArraySize)
        container->getDOFIndexArrayForModification().resize(prevDOFIndexArraySize - nPoints);
    if (prevPointSetIndexArraySize)
        container->getPointSetIndexArrayForModification().resize(prevPointSetIndexArraySize - nPoints);
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
    m_topologyContainer= new PointSetTopologyContainer(this);
    m_topologyModifier= new PointSetTopologyModifier<DataTypes>(this);
    m_topologyAlgorithms= new PointSetTopologyAlgorithms<DataTypes>(this);
    m_geometryAlgorithms= new PointSetGeometryAlgorithms<DataTypes>(this);
}
template<class DataTypes>
PointSetTopology<DataTypes>::PointSetTopology(MechanicalObject<DataTypes> *obj,const PointSetTopology *pst) : object(obj)
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
        this->resetTopologyChangeList();
//                m_topologyContainer->getChangeList().erase(m_topologyContainer->getChangeList().begin(), m_topologyContainer->getChangeList().end());
    }
}



template<class DataTypes>
void PointSetTopology<DataTypes>::init()
{

}
template<class DataTypes>
bool PointSetTopology<DataTypes>::load(const char *filename)
{
    return ((PointSetTopologyModifier<DataTypes> *) m_topologyModifier)->load(filename);
}


} // namespace Components

} // namespace Sofa

#endif
