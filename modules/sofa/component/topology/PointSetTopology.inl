/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
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
////////////////////////////////////PointSetTopology/////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template<class DataTypes>
PointSetTopology<DataTypes>::PointSetTopology(MechanicalObject<DataTypes> *obj)
    : f_m_topologyContainer(new DataPtr< PointSetTopologyContainer >(new PointSetTopologyContainer(), "Point Container"))
    , object(obj)
    , revisionCounter(0)
{
    // TODO: move this to init if possible
    m_topologyContainer = f_m_topologyContainer->beginEdit();
    this->m_topologyContainer->setTopology(this);
    m_topologyModifier=(new PointSetTopologyModifier<DataTypes>(this));
    m_topologyAlgorithms=(new PointSetTopologyAlgorithms<DataTypes>(this));
    m_geometryAlgorithms=(new PointSetGeometryAlgorithms<DataTypes>(this));

    this->addField(this->f_m_topologyContainer, "pointcontainer");
}

template<class DataTypes>
void PointSetTopology<DataTypes>::init()
{
    f_m_topologyContainer->beginEdit();
}

template<class DataTypes>
void PointSetTopology<DataTypes>::propagateTopologicalChanges()
{
    sofa::simulation::TopologyChangeVisitor a;
    this->getContext()->executeVisitor(&a);

    // remove the changes we just propagated, so that we don't send then again next time
    this->resetTopologyChangeList();

    ++revisionCounter;
}

template<class DataTypes>
void PointSetTopology<DataTypes>::propagateStateChanges()
{
    sofa::simulation::StateChangeVisitor a;
    this->getContext()->executeVisitor(&a);

    // remove the changes we just propagated, so that we don't send then again next time
    this->resetStateChangeList();
}

template<class DataTypes>
bool PointSetTopology<DataTypes>::load(const char *filename)
{
    return getPointSetTopologyModifier()->load(filename);
}

template<class DataTypes>
void PointSetTopology<DataTypes>::applyScale(const double scale)
{
    return getPointSetTopologyModifier()->applyScale(scale);
}

template<class DataTypes>
void PointSetTopology<DataTypes>::applyTranslation(const double dx,const double dy,const double dz)
{
    return getPointSetTopologyModifier()->applyTranslation(dx,dy,dz);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////PointSetGeometryAlgorithms///////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template <class DataTypes>
typename DataTypes::Coord PointSetGeometryAlgorithms<DataTypes>::getPointSetCenter() const
{
    typename DataTypes::Coord center;
    // get restPosition
    PointSetTopology<DataTypes> *topology = getPointSetTopology();
    typename DataTypes::VecCoord& p = *(topology->getDOF()->getX0());

    const unsigned int numVertices = topology->getDOFNumber();
    for(unsigned int i=0; i<numVertices; ++i)
    {
        center += p[i];
    }

    center /= numVertices;
    return center;
}

template<class DataTypes>
void  PointSetGeometryAlgorithms<DataTypes>::getEnclosingSphere(typename DataTypes::Coord &center,
        typename DataTypes::Real &radius) const
{
    // get restPosition
    PointSetTopology<DataTypes> *topology = getPointSetTopology();
    typename DataTypes::VecCoord& p = *(topology->getDOF()->getX0());

    const unsigned int numVertices = topology->getDOFNumber();
    for(unsigned int i=0; i<numVertices; ++i)
    {
        center += p[i];
    }
    center /= numVertices;
    radius = (Real) 0;

    for(unsigned int i=0; i<numVertices; ++i)
    {
        const Coord dp = center-p[i];
        const Real val = dot(dp,dp);
        if(val > radius)
            radius = val;
    }
    radius = (Real)sqrt((double) radius);
}

template<class DataTypes>
void  PointSetGeometryAlgorithms<DataTypes>::getAABB(typename DataTypes::Real bb[6] ) const
{
    // get restPosition
    PointSetTopology<DataTypes> *parent = getPointSetTopology();
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

    PointSetTopologyLoader() {}

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
    PointSetTopology<DataTypes> *topology = getPointSetTopology();

    if ((loader->pointArray.size() > 0) && (topology->getDOFNumber() <= 1))
    {
        /// resize the DOF stored in the mechanical object
        topology->getDOF()->resize(loader->pointArray.size());

        /// store position and vertex index in containers
        for (unsigned int index=0; index<loader->pointArray.size(); ++index)
        {
            (*topology->getDOF()->getX())[index] = loader->pointArray[index];
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
    getPointSetTopology()->getDOF()->applyTranslation(dx,dy,dz);
}

template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::applyScale (const double s)
{
    getPointSetTopology()->getDOF()->applyScale(s);
}

template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::swapPoints(const int i1,const int i2)
{
    getPointSetTopology()->getDOF()->swapValues( i1, i2 );

    PointsIndicesSwap *e = new PointsIndicesSwap( i1, i2 ); // local or global indices ? (example of edges)
    this->addTopologyChange(e);
}

template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::addPointsProcess(const unsigned int /*nPoints*/, const bool /*addDOF*/)
{}

template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::addPointsProcess(const unsigned int /*nPoints*/,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& /*ancestors*/,
        const sofa::helper::vector< sofa::helper::vector< double > >& /*baryCoefs*/,
        const bool /*addDOF*/)
{}

template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::addNewPoint(unsigned int i, const sofa::helper::vector< double >& x)
{
    PointSetTopology<DataTypes> *topology = getPointSetTopology();

    unsigned int prevSizeMechObj = i; //container->getNumberOfVertices();//topology->getDOFNumber();

    // resizing the state vectors
    topology->getDOF()->resize( prevSizeMechObj + 1 );

    topology->getDOF()->computeNewPoint(prevSizeMechObj, x);
}

template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::addPointsWarning(const unsigned int nPoints, const bool addDOF)
{
    PointSetTopology<DataTypes> *topology = getPointSetTopology();

    if(addDOF)
    {
        PointsAdded *e2 = new PointsAdded(nPoints);
        addStateChange(e2);
        topology->propagateStateChanges();
    }

    // Warning that vertices just got created
    PointsAdded *e = new PointsAdded(nPoints);
    this->addTopologyChange(e);
}

template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::addPointsWarning(const unsigned int nPoints,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > &ancestors,
        const sofa::helper::vector< sofa::helper::vector< double       > >& coefs,
        const bool addDOF)
{
    PointSetTopology<DataTypes> *topology = getPointSetTopology();

    if(addDOF)
    {
        PointsAdded *e2 = new PointsAdded(nPoints, ancestors, coefs);
        addStateChange(e2);
        topology->propagateStateChanges();
    }

    // Warning that vertices just got created
    PointsAdded *e = new PointsAdded(nPoints, ancestors, coefs);
    this->addTopologyChange(e);
}

template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::removePointsWarning(sofa::helper::vector<unsigned int> &indices,
        const bool removeDOF)
{
    // TODO: clarify why sorting is necessary
    std::sort( indices.begin(), indices.end(), std::greater<unsigned int>() );

    // Warning that these vertices will be deleted
    PointsRemoved *e = new PointsRemoved(indices);
    this->addTopologyChange(e);

    if(removeDOF)
    {
        PointsRemoved *e2=new PointsRemoved(indices);
        addStateChange(e2);
    }
}

template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::removePointsProcess(const sofa::helper::vector<unsigned int> & /*indices*/,
        const bool removeDOF)
{
    if(removeDOF)
    {
        getPointSetTopology()->propagateStateChanges();
    }
}

template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::renumberPointsWarning( const sofa::helper::vector<unsigned int> &index,
        const sofa::helper::vector<unsigned int> &inv_index,
        const bool renumberDOF)
{
    // Warning that these vertices will be deleted
    PointsRenumbering *e = new PointsRenumbering(index, inv_index);
    this->addTopologyChange(e);

    if(renumberDOF)
    {
        PointsRenumbering *e2 = new PointsRenumbering(index, inv_index);
        addStateChange(e2);
    }
}

template<class DataTypes>
void PointSetTopologyModifier<DataTypes>::renumberPointsProcess( const sofa::helper::vector<unsigned int> &/*index*/,
        const sofa::helper::vector<unsigned int> &/*inv_index*/,
        const bool renumberDOF)
{
    if(renumberDOF)
    {
        getPointSetTopology()->propagateStateChanges();
    }
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif
