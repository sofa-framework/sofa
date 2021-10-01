/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once
#include <SofaBaseTopology/TopologyData.h>
#include <SofaBaseTopology/TopologyDataHandler.inl>

namespace sofa::component::topology
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Generic Topology Data Implementation   /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TopologyElementType, typename VecT>
TopologyData <TopologyElementType, VecT>::TopologyData(const typename sofa::core::topology::BaseTopologyData< VecT >::InitData& data)
    : sofa::core::topology::BaseTopologyData< VecT >(data)
    , m_topologyHandler(nullptr)
    , m_isTopologyDynamic(false)
{
}


template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::createTopologyHandler(sofa::core::topology::BaseMeshTopology* _topology)
{
    if (_topology == nullptr)
    {
        msg_error(this->getOwner()) << "Topology used to register this TopologyData: " << this->getName() << " is invalid. TopologyData won't be registered.";
        return;
    }
    this->m_topology = _topology;

    // Create TopologyHandler
    this->m_topologyHandler = new TopologyDataHandler< TopologyElementType, VecT>(this, _topology);
    this->m_topologyHandler->setNamePrefix("TopologyDataHandler( " + this->getOwner()->getName() + " )");
    this->m_topologyHandler->init();

    // Register the TopologyHandler
    m_isTopologyDynamic = this->m_topologyHandler->registerTopology(_topology);
    if (m_isTopologyDynamic)
    {
        this->linkToElementDataArray((TopologyElementType*)nullptr);
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " initialized with dynamic " << _topology->getClassName() << "Topology.";
    }
}


template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::createTopologyHandler(sofa::core::topology::BaseMeshTopology* _topology, sofa::component::topology::TopologyDataHandler< TopologyElementType, VecT>* topoHandler)
{
    if (_topology == nullptr)
    {
        msg_error(this->getOwner()) << "Topology used to register this TopologyData: " << this->getName() << " is invalid. TopologyData won't be registered.";
        return;
    }

    this->m_topology = _topology;

    // Set Topology TopologyHandler
    this->m_topologyHandler = topoHandler;
    this->m_topologyHandler->setNamePrefix("TopologyDataHandler( " + this->getOwner()->getName() + " )");
    this->m_topologyHandler->init();

    // Register the TopologyHandler
    m_isTopologyDynamic = this->m_topologyHandler->registerTopology(_topology);
    if (m_isTopologyDynamic)
    {
        this->linkToElementDataArray((TopologyElementType*)nullptr);
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " initialized with dynamic " << this->m_topology->getClassName() << "Topology.";
    }
    else
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " has no TopologyHandler. Topological changes will be disabled. Use createTopologyHandler method before registerTopologicalData to allow topological changes.";
   
}


/// Method used to link Data to point Data array, using the TopologyHandler's method
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::linkToPointDataArray()
{
    if (this->m_topologyHandler && m_isTopologyDynamic)
    {
        this->m_topologyHandler->linkToPointDataArray();
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " linkToPointDataArray ";
    }
    else
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " won't be linkToPointDataArray as toplogy is not dynamic";
}

/// Method used to link Data to edge Data array, using the TopologyHandler's method
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::linkToEdgeDataArray()
{
    if (this->m_topologyHandler && m_isTopologyDynamic)
    {
        this->m_topologyHandler->linkToEdgeDataArray();
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " linkToEdgeDataArray ";
    }
    else
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " won't be linkToEdgeDataArray as toplogy is not dynamic";
}

/// Method used to link Data to triangle Data array, using the TopologyHandler's method
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::linkToTriangleDataArray()
{
    if (this->m_topologyHandler && m_isTopologyDynamic)
    {
        this->m_topologyHandler->linkToTriangleDataArray();
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " linkToTriangleDataArray ";
    }
    else
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " won't be linkToTriangleDataArray as toplogy is not dynamic";
}

/// Method used to link Data to quad Data array, using the TopologyHandler's method
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::linkToQuadDataArray()
{
    if (this->m_topologyHandler && m_isTopologyDynamic)
    {
        this->m_topologyHandler->linkToQuadDataArray();
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " linkToQuadDataArray ";
    }
    else
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " won't be linkToQuadDataArray as toplogy is not dynamic";
}

/// Method used to link Data to tetrahedron Data array, using the TopologyHandler's method
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::linkToTetrahedronDataArray()
{
    if (this->m_topologyHandler && m_isTopologyDynamic)
    {
        this->m_topologyHandler->linkToTetrahedronDataArray();
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " linkToTetrahedronDataArray ";
    }
    else
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " won't be linkToTetrahedronDataArray as toplogy is not dynamic";
}

/// Method used to link Data to hexahedron Data array, using the TopologyHandler's method
template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::linkToHexahedronDataArray()
{
    if (this->m_topologyHandler && m_isTopologyDynamic)
    {
        this->m_topologyHandler->linkToHexahedronDataArray();
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " linkToHexahedronDataArray ";
    }
    else
        msg_info(this->getOwner()) << "TopologyData: " << this->getName() << " won't be linkToHexahedronDataArray as toplogy is not dynamic";
}


/////////////////////// Protected functions on TopologyData init ///////////////////////////////

template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::linkToElementDataArray(sofa::core::topology::BaseMeshTopology::Point*) 
{ 
    this->setDataSetArraySize(this->m_topology->getNbPoints());
    linkToPointDataArray(); 
}

template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::linkToElementDataArray(sofa::core::topology::BaseMeshTopology::Edge*) 
{ 
    this->setDataSetArraySize(this->m_topology->getNbEdges());
    linkToEdgeDataArray(); 
}

template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::linkToElementDataArray(sofa::core::topology::BaseMeshTopology::Triangle*) 
{ 
    this->setDataSetArraySize(this->m_topology->getNbTriangles());
    linkToTriangleDataArray(); 
}

template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::linkToElementDataArray(sofa::core::topology::BaseMeshTopology::Quad*) 
{ 
    this->setDataSetArraySize(this->m_topology->getNbQuads());
    linkToQuadDataArray(); 
}

template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::linkToElementDataArray(sofa::core::topology::BaseMeshTopology::Tetrahedron*) 
{ 
    this->setDataSetArraySize(this->m_topology->getNbTetrahedra());
    linkToTetrahedronDataArray(); 
}

template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::linkToElementDataArray(sofa::core::topology::BaseMeshTopology::Hexahedron*) 
{ 
    this->setDataSetArraySize(this->m_topology->getNbHexahedra());
    linkToHexahedronDataArray(); 
}


///////////////////// Protected functions on TopologyData changes /////////////////////////////

template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::swap(Index i1, Index i2)
{
    container_type& data = *(this->beginEdit());
    value_type tmp = data[i1];
    data[i1] = data[i2];
    data[i2] = tmp;
    this->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::remove(const sofa::type::vector<Index>& index)
{

    container_type& data = *(this->beginEdit());
    if (data.size() > 0)
    {
        for (std::size_t i = 0; i < index.size(); ++i)
        {
            if (this->m_topologyHandler) {
                this->m_topologyHandler->applyDestroyFunction(index[i], data[index[i]]);
            }

            if (p_onDestructionCallback)
            {
                p_onDestructionCallback(index[i], data[index[i]]);
            }

            this->swap(index[i], this->m_lastElementIndex);
            --this->m_lastElementIndex;
        }

        data.resize(data.size() - index.size());
    }
    this->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::add(const sofa::type::vector<Index>& index,
    const sofa::type::vector< TopologyElementType >& elems,
    const sofa::type::vector<sofa::type::vector<Index> >& ancestors,
    const sofa::type::vector<sofa::type::vector<double> >& coefs,
    const sofa::type::vector< AncestorElem >& ancestorElems)
{
    std::size_t nbElements = index.size();
    if (nbElements == 0) return;
    // Using default values
    container_type& data = *(this->beginEdit());
    std::size_t i0 = data.size();
    if (i0 != index[0])
    {
        msg_error(this->getOwner()) << "TopologyDataHandler SIZE MISMATCH in Data "
            << this->getName() << ": " << nbElements << " "
            << core::topology::TopologyElementInfo<TopologyElementType>::name()
            << " ADDED starting from index " << index[0]
            << " while vector size is " << i0;
        i0 = index[0];
    }
    data.resize(i0 + nbElements);
    this->m_lastElementIndex += sofa::Index(nbElements);

    const sofa::type::vector< Index > empty_vecint;
    const sofa::type::vector< double > empty_vecdouble;

    if (this->m_topologyHandler)
    {
        for (Index i = 0; i < nbElements; ++i)
        {
            value_type& t = data[i0 + i];
        
            this->m_topologyHandler->applyCreateFunction(Index(i0 + i), t, elems[i],
                    (ancestors.empty() || coefs.empty()) ? empty_vecint : ancestors[i],
                    (ancestors.empty() || coefs.empty()) ? empty_vecdouble : coefs[i],
                    (ancestorElems.empty()) ? nullptr : &ancestorElems[i]);
            
            if (p_onCreationCallback)
            {
                p_onCreationCallback(Index(i0 + i), t, elems[i],
                    (ancestors.empty() || coefs.empty()) ? empty_vecint : ancestors[i],
                    (ancestors.empty() || coefs.empty()) ? empty_vecdouble : coefs[i]);
            }
        }
    }
    this->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::move(const sofa::type::vector<Index>& indexList,
    const sofa::type::vector< sofa::type::vector< Index > >& ancestors,
    const sofa::type::vector< sofa::type::vector< double > >& coefs)
{
    container_type& data = *(this->beginEdit());

    if (this->m_topologyHandler)
    {
        for (std::size_t i = 0; i < indexList.size(); i++)
        {
            this->m_topologyHandler->applyDestroyFunction(indexList[i], data[indexList[i]]);
            this->m_topologyHandler->applyCreateFunction(indexList[i], data[indexList[i]], ancestors[i], coefs[i]);
        }
    }

    this->endEdit();
}



template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::renumber(const sofa::type::vector<Index>& index)
{
    container_type& data = *(this->beginEdit());

    container_type copy = this->getValue(); // not very efficient memory-wise, but I can see no better solution...
    for (std::size_t i = 0; i < index.size(); ++i)
        data[i] = copy[index[i]];

    this->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::addOnMovedPosition(const sofa::type::vector<Index>& indexList,
    const sofa::type::vector<TopologyElementType>& elems)
{
    container_type& data = *(this->beginEdit());

    // Recompute data
    sofa::type::vector< Index > ancestors;
    sofa::type::vector< double >  coefs;
    coefs.push_back(1.0);
    ancestors.resize(1);

    if (this->m_topologyHandler)
    {
        for (std::size_t i = 0; i < indexList.size(); i++)
        {
            ancestors[0] = indexList[i];
            this->m_topologyHandler->applyCreateFunction(indexList[i], data[indexList[i]], elems[i], ancestors, coefs);
        }
    }
    this->m_lastElementIndex += sofa::Index(indexList.size());
    this->endEdit();
}


template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::removeOnMovedPosition(const sofa::type::vector<Index>& indices)
{
    container_type& data = *(this->beginEdit());

    if (this->m_topologyHandler)
    {
        for (std::size_t i = 0; i < indices.size(); i++) {
            this->m_topologyHandler->applyDestroyFunction(indices[i], data[indices[i]]);
        }
    }

    this->m_lastElementIndex -= sofa::Index(indices.size());
    this->endEdit();

    // TODO check why this call.
    //this->remove( indices );
}


template <typename TopologyElementType, typename VecT>
void TopologyData <TopologyElementType, VecT>::addTopologyEventCallBack(core::topology::TopologyChangeType type, TopologyChangeCallback callback)
{
    if (m_topologyHandler != nullptr)
    {
        m_topologyHandler->addCallBack(type, callback);
    }
    else
    {
        msg_warning(this->getOwner()) << "No TopologyHandler has been creating to manage this TopologyData: " << this->getName() 
            << ". Callback for event: '" << parseTopologyChangeTypeToString(type) << "' won't be registered.";
    }

}


} //namespace sofa::component::topology
