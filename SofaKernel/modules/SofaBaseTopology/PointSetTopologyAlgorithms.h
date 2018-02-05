/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_POINTSETTOPOLOGYALGORITHMS_H
#define SOFA_COMPONENT_TOPOLOGY_POINTSETTOPOLOGYALGORITHMS_H
#include "config.h"

#include <sofa/core/topology/BaseTopology.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace topology
{
class PointSetTopologyContainer;

class PointSetTopologyModifier;

template < class DataTypes >
class PointSetGeometryAlgorithms;

/** A class that performs complex algorithms on a PointSet.
*
*/
template<class DataTypes>
class PointSetTopologyAlgorithms : public core::topology::TopologyAlgorithms
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PointSetTopologyAlgorithms,DataTypes), core::topology::TopologyAlgorithms);
protected:
    PointSetTopologyAlgorithms()
        : TopologyAlgorithms()
    {}

    virtual ~PointSetTopologyAlgorithms() {}
public:
    virtual void init() override;

    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        return BaseObject::canCreate(obj, context, arg);
    }

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const PointSetTopologyAlgorithms<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }
private:
    PointSetTopologyContainer*					m_container;
    PointSetTopologyModifier*					m_modifier;
    PointSetGeometryAlgorithms< DataTypes >*	m_geometryAlgorithms;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_TOPOLOGY_POINTSETTOPOLOGYALGORITHMS_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_TOPOLOGY_API PointSetTopologyAlgorithms<defaulttype::Vec3dTypes>;
extern template class SOFA_BASE_TOPOLOGY_API PointSetTopologyAlgorithms<defaulttype::Vec2dTypes>;
extern template class SOFA_BASE_TOPOLOGY_API PointSetTopologyAlgorithms<defaulttype::Vec1dTypes>;
extern template class SOFA_BASE_TOPOLOGY_API PointSetTopologyAlgorithms<defaulttype::Rigid3dTypes>;
extern template class SOFA_BASE_TOPOLOGY_API PointSetTopologyAlgorithms<defaulttype::Rigid2dTypes>;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_TOPOLOGY_API PointSetTopologyAlgorithms<defaulttype::Vec3fTypes>;
extern template class SOFA_BASE_TOPOLOGY_API PointSetTopologyAlgorithms<defaulttype::Vec2fTypes>;
extern template class SOFA_BASE_TOPOLOGY_API PointSetTopologyAlgorithms<defaulttype::Vec1fTypes>;
extern template class SOFA_BASE_TOPOLOGY_API PointSetTopologyAlgorithms<defaulttype::Rigid3fTypes>;
extern template class SOFA_BASE_TOPOLOGY_API PointSetTopologyAlgorithms<defaulttype::Rigid2fTypes>;
#endif
#endif

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_POINTSETTOPOLOGYALGORITHMS_H
