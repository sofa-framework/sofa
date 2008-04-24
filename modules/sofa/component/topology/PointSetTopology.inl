#ifndef SOFA_COMPONENT_TOPOLOGY_POINTSETTOPOLOGY_INL
#define SOFA_COMPONENT_TOPOLOGY_POINTSETTOPOLOGY_INL


#include <sofa/component/topology/PointSetTopology.h>
#include <sofa/simulation/tree/TopologyChangeVisitor.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/helper/io/MeshTopologyLoader.h>

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
    typedef Vector3::value_type Real_Sofa;

    VecCoord pointArray;

    PointSetTopologyLoader()
    {
    }
    virtual void addPoint(Real_Sofa px, Real_Sofa py, Real_Sofa pz)
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
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
{
    PointSetTopology<DataTypes> *topology = dynamic_cast<PointSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);
    unsigned int prevSizeMechObj   = topology->object->getSize();

    // resizing the state vectors
    topology->object->resize( prevSizeMechObj + nPoints );

    if ( ancestors != (const sofa::helper::vector< sofa::helper::vector< unsigned int > >)0 )
    {
        assert( baryCoefs == (const sofa::helper::vector< sofa::helper::vector< double > >)0 || ancestors.size() == baryCoefs.size() );

        sofa::helper::vector< sofa::helper::vector< double > > coefs;
        coefs.resize(ancestors.size());

        for (unsigned int i = 0; i < ancestors.size(); ++i)
        {
            assert( baryCoefs == (const sofa::helper::vector< sofa::helper::vector< double > >)0 || baryCoefs[i].size() == 0 || ancestors[i].size() == baryCoefs[i].size() );
            coefs[i].resize(ancestors[i].size());


            for (unsigned int j = 0; j < ancestors[i].size(); ++j)
            {
                // constructng default coefs if none were defined
                if (baryCoefs == (const sofa::helper::vector< sofa::helper::vector< double > >)0 || baryCoefs[i].size() == 0)
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

}

template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::addNewPoint( const sofa::helper::vector< double >& x)
{
    PointSetTopology<DataTypes> *topology = dynamic_cast<PointSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);
    unsigned int prevSizeMechObj   = topology->object->getSize();

    // resizing the state vectors
    topology->object->resize( prevSizeMechObj + 1 );

    topology->object->computeNewPoint(prevSizeMechObj, x);

}



template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::addPointsWarning(const unsigned int nPoints,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
        const sofa::helper::vector< sofa::helper::vector< double       > >& coefs)
{
    // Warning that vertices just got created
    PointsAdded *e=new PointsAdded(nPoints, ancestors, coefs);
    addTopologyChange(e);
}




template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::removePointsWarning(sofa::helper::vector<unsigned int> &indices)
{

    std::sort( indices.begin(), indices.end(), std::greater<unsigned int>() ); // BUG FIXED : sort indices here
    // Warning that these vertices will be deleted
    PointsRemoved *e=new PointsRemoved(indices);
    addTopologyChange(e);
}



template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::removePointsProcess( sofa::helper::vector<unsigned int> &indices, const bool removeDOF)
{
    // BUG FIXED : do not sort indices here
    //std::sort( indices.begin(), indices.end(), std::greater<unsigned int>() );

    // Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)

    PointSetTopology<DataTypes> *topology = dynamic_cast<PointSetTopology<DataTypes> *>(m_basicTopology);
    assert (topology != 0);
    int prevSizeMechObj   = topology->object->getSize();

    int lastIndexMech = prevSizeMechObj - 1;

    if(removeDOF)
    {

        // deleting the vertices
        for (unsigned int i = 0; i < indices.size(); ++i)
        {
            topology->object->replaceValue(lastIndexMech, indices[i] );

            --lastIndexMech;
        }

        // resizing the state vectors
        topology->object->resize( prevSizeMechObj - indices.size() );

    }
    else
    {
        topology->object->resize( prevSizeMechObj );
    }

}


template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::renumberPointsWarning( const sofa::helper::vector<unsigned int> &index, const sofa::helper::vector<unsigned int> &inv_index)
{
    // Warning that these vertices will be deleted
    PointsRenumbering *e=new PointsRenumbering(index, inv_index);
    addTopologyChange(e);
}

template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::renumberPointsProcess( const sofa::helper::vector<unsigned int> &index, const sofa::helper::vector<unsigned int> &/*inv_index*/, const bool renumberDOF)
{

    if(renumberDOF)
    {
        PointSetTopology<DataTypes> *topology = dynamic_cast<PointSetTopology<DataTypes> *>(m_basicTopology);
        assert (topology != 0);
        topology->object->renumberValues( index );
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
void  PointSetGeometryAlgorithms<DataTypes>::getAABB(typename DataTypes::Real /*bb*/[6] ) const
{
}


/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////PointSetTopology/////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template<class DataTypes>
PointSetTopology<DataTypes>::PointSetTopology(MechanicalObject<DataTypes> *obj) :
    object(obj), f_m_topologyContainer(new DataPtr< PointSetTopologyContainer >(new PointSetTopologyContainer(), "Point Container"))
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

    sofa::simulation::tree::TopologyChangeVisitor a;
    a.resetNbIter();
    getContext()->executeVisitor(&a);
    // BUGFIX (Jeremie A. 06/12/07): remove the changes we just propagated, so that we don't send then again next time
    this->resetTopologyChangeList();
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
