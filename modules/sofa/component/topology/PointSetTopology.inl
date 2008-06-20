#ifndef SOFA_COMPONENT_TOPOLOGY_POINTSETTOPOLOGY_INL
#define SOFA_COMPONENT_TOPOLOGY_POINTSETTOPOLOGY_INL


#include <sofa/component/topology/PointSetTopology.h>
#include <sofa/simulation/common/TopologyChangeVisitor.h>
#include <sofa/simulation/common/StateChangeVisitor.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/helper/io/MeshTopologyLoader.h>
#include <sofa/defaulttype/VecTypes.h>
namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::behavior;



/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////PointSetTopologyModifier/////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template<class DataTypes>
class PointSetTopologyLoader : public helper::io::MeshTopologyLoader
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
        //BUGFIX(Jeremie A.): The following does not work for 1D/2D datatypes
        //pointArray.push_back(Coord((Real)px,(Real)py,(Real)pz));
        Coord c;
        DataTypes::set(c,px,py,pz);
        pointArray.push_back(c);
    }
};

template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::loadPointSet(PointSetTopologyLoader<DataTypes> *loader)
{
    PointSetTopology<DataTypes> *topology = dynamic_cast<PointSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);
    if ((loader->pointArray.size()>0) && (topology->object->getSize()<=1))
    {
        /// resize the DOF stored in the mechanical object
        topology->object->resize(loader->pointArray.size());

        /// store position and vertex index in containers
        unsigned int index;
        for (index=0; index<loader->pointArray.size(); ++index)
        {
            (*topology->object->getX())[index]=loader->pointArray[index];
        }
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
void PointSetTopologyModifier<DataTypes>::applyTranslation (const double dx,const double dy,const double dz)
{
    PointSetTopology<DataTypes> *topology = dynamic_cast<PointSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);
    topology->object->applyTranslation(dx,dy,dz);
}
template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::applyScale (const double s)
{
    PointSetTopology<DataTypes> *topology = dynamic_cast<PointSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);
    topology->object->applyScale(s);
}
template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::swapPoints(const int i1,const int i2)
{
    PointSetTopology<DataTypes> *topology = dynamic_cast<PointSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);
    topology->object->swapValues( i1, i2 );

    PointsIndicesSwap *e=new PointsIndicesSwap( i1, i2 ); // local or global indices ? (example of edges)
    addTopologyChange(e);
}

template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::addPointsProcess(const unsigned int nPoints,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
        const sofa::helper::vector< sofa::helper::vector< double > >& /*baryCoefs*/, const bool addDOF)
{}

template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::addNewPoint(unsigned int i, const sofa::helper::vector< double >& x)
{
    PointSetTopology<DataTypes> *topology = dynamic_cast<PointSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);

    unsigned int prevSizeMechObj   = i;//container->getNumberOfVertices();//topology->object->getSize();

    // resizing the state vectors
    topology->object->resize( prevSizeMechObj + 1 );

    topology->object->computeNewPoint(prevSizeMechObj, x);
}



template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::addPointsWarning(const unsigned int nPoints,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
        const sofa::helper::vector< sofa::helper::vector< double       > >& coefs, const bool addDOF)
{

    PointSetTopology<DataTypes> *topology = dynamic_cast<PointSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);

    if(addDOF)
    {

        PointsAdded *e2=new PointsAdded(nPoints, ancestors, coefs);
        addStateChange(e2);
        topology->propagateStateChanges();

    }

    // Warning that vertices just got created
    PointsAdded *e=new PointsAdded(nPoints, ancestors, coefs);
    addTopologyChange(e);

}


template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::removePointsWarning(sofa::helper::vector<unsigned int> &indices, const bool removeDOF)
{

    PointSetTopology<DataTypes> *topology = dynamic_cast<PointSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);

    std::sort( indices.begin(), indices.end(), std::greater<unsigned int>() ); // BUG FIXED : sort indices here
    // Warning that these vertices will be deleted
    PointsRemoved *e=new PointsRemoved(indices);
    addTopologyChange(e);

    if(removeDOF)
    {
        PointsRemoved *e2=new PointsRemoved(indices);
        addStateChange(e2);
    }

}



template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::removePointsProcess( sofa::helper::vector<unsigned int> &/*indices*/, const bool removeDOF)
{

    PointSetTopology<DataTypes> *topology = dynamic_cast<PointSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);

    if(removeDOF)
    {
        topology->propagateStateChanges();
    }

}


template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::renumberPointsWarning( const sofa::helper::vector<unsigned int> &index, const sofa::helper::vector<unsigned int> &inv_index, const bool renumberDOF)
{
    // Warning that these vertices will be deleted
    PointsRenumbering *e=new PointsRenumbering(index, inv_index);
    addTopologyChange(e);

    if(renumberDOF)
    {
        PointsRenumbering *e2=new PointsRenumbering(index, inv_index);
        addStateChange(e2);
    }
}

template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::renumberPointsProcess( const sofa::helper::vector<unsigned int> &index, const sofa::helper::vector<unsigned int> &/*inv_index*/, const bool renumberDOF)
{

    PointSetTopology<DataTypes> *topology = dynamic_cast<PointSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);

    if(renumberDOF)
    {
        topology->propagateStateChanges();
    }

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
    typename DataTypes::VecCoord& p = *parent->getDOF()->getX0();
    PointSetTopologyContainer *ps=static_cast<PointSetTopologyContainer *>( parent->getTopologyContainer());

    unsigned int i;
    for(i=0; i<ps->getNumberOfVertices(); i++)
    {
        center+=p[i];
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
    typename DataTypes::VecCoord& p = *parent->getDOF()->getX0();
    PointSetTopologyContainer *ps=static_cast<PointSetTopologyContainer *>( parent->getTopologyContainer());

    unsigned int i;
    for(i=0; i<ps->getNumberOfVertices(); i++)
    {
        center+=p[i];
    }
    center/= (ps->getNumberOfVertices());
    dp=center-p[0];
    radius=dot(dp,dp);
    for(i=1; i<ps->getNumberOfVertices(); i++)
    {
        dp=center-p[i];
        val=dot(dp,dp);
        if (val<radius)
            radius=val;
    }
    radius=(Real)sqrt((double) radius);
}



template<class DataTypes>
void  PointSetGeometryAlgorithms<DataTypes>::getAABB(typename DataTypes::Real bb[6] ) const
{
    // get restPosition
    PointSetTopology<DataTypes> *parent=static_cast<PointSetTopology<DataTypes> *>(m_basicTopology);
    typename DataTypes::VecCoord& p = *parent->getDOF()->getX0();

    bb[0] = (Real) p[0][0];
    bb[1] = (Real) p[0][1];
    bb[2] = (Real) p[0][2];
    bb[3] = (Real) p[0][0];
    bb[4] = (Real) p[0][1];
    bb[5] = (Real) p[0][2];

    for(unsigned int i=1; i<p.size(); ++i)
    {
        // min
        if(bb[0] > (Real) p[i][0]) bb[0] = (Real) p[i][0];	// x
        if(bb[1] > (Real) p[i][1]) bb[1] = (Real) p[i][1];	// y
        if(bb[2] > (Real) p[i][2]) bb[2] = (Real) p[i][2];	// z

        // max
        if(bb[3] < (Real) p[i][0]) bb[3] = (Real) p[i][0];	// x
        if(bb[4] < (Real) p[i][1]) bb[4] = (Real) p[i][1];	// y
        if(bb[5] < (Real) p[i][2]) bb[5] = (Real) p[i][2];	// z
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////PointSetTopology/////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template<class DataTypes>
PointSetTopology<DataTypes>::PointSetTopology(MechanicalObject<DataTypes> *obj) :
    object(obj),
    f_m_topologyContainer(new DataPtr< PointSetTopologyContainer >(new PointSetTopologyContainer(), "Point Container")),
    revisionCounter(0)
{
    m_topologyContainer=f_m_topologyContainer->beginEdit();
    this->m_topologyContainer->setTopology(this);
    m_topologyModifier=(new PointSetTopologyModifier<DataTypes>(this));
    m_topologyAlgorithms=(new PointSetTopologyAlgorithms<DataTypes>(this));
    m_geometryAlgorithms=(new PointSetGeometryAlgorithms<DataTypes>(this));

    this->addField(this->f_m_topologyContainer, "pointcontainer");
}

template<class DataTypes>
PointSetTopology<DataTypes>::PointSetTopology(MechanicalObject<DataTypes> *obj,const PointSetTopology *) : object(obj), f_m_topologyContainer(new DataPtr< PointSetTopologyContainer >(new PointSetTopologyContainer(), "Point Container"))

{
    m_topologyContainer=f_m_topologyContainer->beginEdit();
    this->m_topologyContainer->setTopology(this);
    m_topologyModifier=(new PointSetTopologyModifier<DataTypes>(this));
    m_topologyAlgorithms=(new PointSetTopologyAlgorithms<DataTypes>(this));
    m_geometryAlgorithms=(new PointSetGeometryAlgorithms<DataTypes>(this));

    this->addField(this->f_m_topologyContainer, "pointcontainer");
}

template<class DataTypes>
void PointSetTopology<DataTypes>::propagateTopologicalChanges()
{

    sofa::simulation::TopologyChangeVisitor a;
    a.resetNbIter();
    this->getContext()->executeVisitor(&a);
    // BUGFIX (Jeremie A. 06/12/07): remove the changes we just propagated, so that we don't send then again next time
    this->resetTopologyChangeList();

    revisionCounter++;
}

template<class DataTypes>
void PointSetTopology<DataTypes>::propagateStateChanges()
{

    sofa::simulation::StateChangeVisitor a;
    this->getContext()->executeVisitor(&a);
    // BUGFIX (Jeremie A. 06/12/07): remove the changes we just propagated, so that we don't send then again next time
    this->resetStateChangeList();
}

template<class DataTypes>
void PointSetTopology<DataTypes>::init()
{
    f_m_topologyContainer->beginEdit();
}
template<class DataTypes>
bool PointSetTopology<DataTypes>::load(const char *filename)
{
    return ((PointSetTopologyModifier<DataTypes> *) m_topologyModifier)->load(filename);
}
template<class DataTypes>
void PointSetTopology<DataTypes>::applyScale(const double scale)
{
    return ((PointSetTopologyModifier<DataTypes> *) m_topologyModifier)->applyScale(scale);
}
template<class DataTypes>
void PointSetTopology<DataTypes>::applyTranslation(const double dx,const double dy,const double dz)
{
    return ((PointSetTopologyModifier<DataTypes> *) m_topologyModifier)->applyTranslation(dx,dy,dz);
}


} // namespace topology

} // namespace component

} // namespace sofa

#endif
