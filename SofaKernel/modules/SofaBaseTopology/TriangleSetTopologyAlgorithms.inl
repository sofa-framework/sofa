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
#ifndef SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGYALGORITHMS_INL

#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
#include <SofaBaseTopology/TriangleSetTopologyModifier.h>
#include <SofaBaseTopology/TriangleSetTopologyAlgorithms.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>

#include <SofaBaseMechanics/MechanicalObject.h>

#include <algorithm>
#include <functional>

namespace sofa
{
namespace component
{
namespace topology
{

template<class DataTypes>
void TriangleSetTopologyAlgorithms< DataTypes >::init()
{
    EdgeSetTopologyAlgorithms< DataTypes >::init();
    this->getContext()->get(m_container);
    this->getContext()->get(m_modifier);
    this->getContext()->get(m_geometryAlgorithms);
}

template<class DataTypes>
void TriangleSetTopologyAlgorithms< DataTypes >::reinit()
{
    if (!(m_listTriRemove.getValue () ).empty() && this->getContext()->getAnimate())
    {
        sofa::helper::vector< TriangleID > items = m_listTriRemove.getValue ();
        m_modifier->removeItems(items);

        m_modifier->propagateTopologicalChanges();
        items.clear();
    }

    if (!(m_listTriAdd.getValue () ).empty() && this->getContext()->getAnimate())
    {
        size_t nbrBefore = m_container->getNbTriangles();

        m_modifier->addTrianglesProcess(m_listTriAdd.getValue ());

        sofa::helper::vector< TriangleID > new_triangles_id;

        for (size_t i = 0; i < (m_listTriAdd.getValue ()).size(); i++)
            new_triangles_id.push_back((TriangleID)(m_container->getNbTriangles()-(m_listTriAdd.getValue ()).size()+i));

        if (nbrBefore != m_container->getNbTriangles()) // Triangles have been added
        {
            m_modifier->addTrianglesWarning(m_listTriAdd.getValue().size(), m_listTriAdd.getValue(), new_triangles_id);
            m_modifier->propagateTopologicalChanges();
        }
        else
        {
            msg_info() << "Nothing added ";
        }

    }

}






} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_TriangleSetTOPOLOGY_INL
