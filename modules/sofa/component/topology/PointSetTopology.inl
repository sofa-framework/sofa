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

#include <sofa/component/topology/PointSetTopologyAlgorithms.inl>
#include <sofa/component/topology/PointSetGeometryAlgorithms.inl>
#include <sofa/component/topology/PointSetTopologyModifier.inl>

namespace sofa
{

namespace component
{

namespace topology
{

template<class DataTypes>
PointSetTopology<DataTypes>::PointSetTopology(MechanicalObject<DataTypes> *obj)
    : object(obj)
    , f_m_topologyContainer(new DataPtr< PointSetTopologyContainer >(NULL, "Point Container"))
    , revisionCounter(0)
{
}

template<class DataTypes>
void PointSetTopology<DataTypes>::createComponents()
{
    this->m_topologyContainer  = new PointSetTopologyContainer(this);
    this->m_topologyModifier   = new PointSetTopologyModifier<DataTypes>(this);
    this->m_topologyAlgorithms = new PointSetTopologyAlgorithms<DataTypes>(this);
    this->m_geometryAlgorithms = new PointSetGeometryAlgorithms<DataTypes>(this);
}

template<class DataTypes>
void PointSetTopology<DataTypes>::parse(sofa::core::objectmodel::BaseObjectDescription* arg)
{
    // Create the container, modifier and algorithms
    createComponents();
    // Add them in the context
    this->getContext()->addObject(this->m_topologyContainer);
    this->getContext()->addObject(this->m_topologyModifier);
    this->getContext()->addObject(this->m_topologyAlgorithms);
    this->getContext()->addObject(this->m_geometryAlgorithms);

    // parse the given parameters, also transmit them to each component
    this->addField(this->f_m_topologyContainer, "pointcontainer");
    this->f_m_topologyContainer->beginEdit();
    core::componentmodel::topology::BaseTopology::parse(arg);
    this->m_topologyContainer->parse(arg);
    this->m_topologyModifier->parse(arg);
    this->m_topologyAlgorithms->parse(arg);
    this->m_geometryAlgorithms->parse(arg);
    // set the name of each component
    this->m_topologyContainer->setName(this->getName()+std::string("Container"));
    this->m_topologyModifier->setName(this->getName()+std::string("Modifier"));
    this->m_topologyAlgorithms->setName(this->getName()+std::string("Algorithms"));
    this->m_geometryAlgorithms->setName(this->getName()+std::string("Geometry"));

    if (arg->getAttribute("filename"))
        this->load(arg->getAttribute("filename"));		// this is called at creation time, a container and modifier must exist !!!
    if (arg->getAttribute("scale")!=NULL)
    {
        this->applyScale(atof(arg->getAttribute("scale")));
    }
    if (arg->getAttribute("dx")!=NULL || arg->getAttribute("dy")!=NULL || arg->getAttribute("dz")!=NULL)
    {
        this->applyTranslation(atof(arg->getAttribute("dx","0.0")),atof(arg->getAttribute("dy","0.0")),atof(arg->getAttribute("dz","0.0")));
    }
    this->core::componentmodel::topology::BaseTopology::parse(arg);
}

template<class DataTypes>
void PointSetTopology<DataTypes>::init()
{
    core::componentmodel::topology::BaseTopology::init();
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

} // namespace topology

} // namespace component

} // namespace sofa

#endif
