
#include <sofa/component/topology/PointSetTopology.h>
#include <sofa/component/topology/PointSetTopology.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(PointSetTopology)


template class PointSetTopologyModifier<Vec3dTypes>;
template class PointSetTopologyModifier<Vec3fTypes>;
template class PointSetTopology<Vec3dTypes>;
template class PointSetTopology<Vec3fTypes>;
template class PointSetGeometryAlgorithms<Vec3fTypes>;
template class PointSetGeometryAlgorithms<Vec3dTypes>;

PointSetTopologyContainer::PointSetTopologyContainer(core::componentmodel::topology::BaseTopology *top)
    : core::componentmodel::topology::TopologyContainer(top)
{
}

/*
PointSetTopologyContainer::PointSetTopologyContainer(core::componentmodel::topology::BaseTopology *top, const sofa::helper::vector<unsigned int>& DOFIndex)
: core::componentmodel::topology::TopologyContainer(top), m_DOFIndex(DOFIndex)
{
}


void PointSetTopologyContainer::createPointSetIndex() {
    // resizing
    m_PointSetIndex.resize( m_basicTopology->getDOFNumber() );

    // initializing
    for (unsigned int i = 0; i < m_PointSetIndex.size(); ++i)
        m_PointSetIndex[i] = -1;

    // overwriting defined DOFs indices
    for (unsigned int i = 0; i < m_DOFIndex.size(); ++i)
    {
        m_PointSetIndex[ m_DOFIndex[i] ] = i;
    }
}



const sofa::helper::vector<int> &PointSetTopologyContainer::getPointSetIndexArray() {
    if (!m_PointSetIndex.size())
        createPointSetIndex();
	return m_PointSetIndex;
}

unsigned int PointSetTopologyContainer::getPointSetIndexSize() const{
	return m_PointSetIndex.size();
}

sofa::helper::vector<int> &PointSetTopologyContainer::getPointSetIndexArrayForModification() {
    if (!m_PointSetIndex.size())
        createPointSetIndex();
	return m_PointSetIndex;
}



int PointSetTopologyContainer::getPointSetIndex(const unsigned int i) {
	if (!m_PointSetIndex.size())
        createPointSetIndex();
	return m_PointSetIndex[i];
}

*/

unsigned int PointSetTopologyContainer::getNumberOfVertices() const
{
    //return m_DOFIndex.size();
    return m_basicTopology->getDOFNumber();
}

/*

const sofa::helper::vector<unsigned int> &PointSetTopologyContainer::getDOFIndexArray() const
{
	return m_DOFIndex;
}

sofa::helper::vector<unsigned int> &PointSetTopologyContainer::getDOFIndexArrayForModification()
{
	return m_DOFIndex;
}

unsigned int PointSetTopologyContainer::getDOFIndex(const int i) const
{
	return m_DOFIndex[i];
}
*/

bool PointSetTopologyContainer::checkTopology() const
{

    //std::cout << "*** CHECK PointSetTopologyContainer ***" << std::endl;

    /*
    if (m_PointSetIndex.size()>0) {
    	 unsigned int i;
    	 for (i=0;i<m_DOFIndex.size();++i) {
    		 bool check_vertex = m_PointSetIndex[m_DOFIndex[i]]!= -1;
    		 if(!check_vertex){
    			std::cout << "*** CHECK FAILED : check_vertex, i = " << i << std::endl;
    		 }
    		 assert(check_vertex);
    	 }
    	 //std::cout << "******** DONE : check_point" << std::endl;
    }
    */

    return true;
}

int PointSetTopologyClass = core::RegisterObject("Topology consisting of a set of points")
        .add< PointSetTopology<Vec3dTypes> >()
        .add< PointSetTopology<Vec3fTypes> >()
        ;




} // namespace topology

} // namespace component

} // namespace sofa

