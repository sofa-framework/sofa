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
#include <sofa/component/topology/container/constant/config.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa::component::topology::container::constant
{

class SOFA_COMPONENT_TOPOLOGY_CONTAINER_CONSTANT_API MeshTopologyContainer :
    public sofa::core::topology::BaseMeshTopology
{
public:
    SOFA_CLASS(MeshTopologyContainer, core::topology::BaseMeshTopology);

    template<class ElementType> //e.g. sofa::geometry::Edge
    using TopologyElement = sofa::topology::Element<ElementType>;

    template<class ElementType> //e.g. sofa::geometry::Edge
    using SeqElement = sofa::type::vector<TopologyElement<ElementType>>;

    template<class ElementType> //e.g. sofa::geometry::Edge
    using DataSeqElement = sofa::Data<SeqElement<ElementType>>;

    template<class ElementType> //e.g. sofa::geometry::Edge
    using DataSeqElementPtr = std::unique_ptr<DataSeqElement<ElementType>>;

    using AllElements = std::tuple<
        DataSeqElementPtr<sofa::geometry::Edge>,
        DataSeqElementPtr<sofa::geometry::Triangle>,
        DataSeqElementPtr<sofa::geometry::Quad>,
        DataSeqElementPtr<sofa::geometry::Tetrahedron>,
        DataSeqElementPtr<sofa::geometry::Hexahedron>,
        DataSeqElementPtr<sofa::geometry::Prism>,
        DataSeqElementPtr<sofa::geometry::Pyramid>,
        DataSeqElementPtr<sofa::geometry::QuadraticEdge>,
        DataSeqElementPtr<sofa::geometry::QuadraticTriangle>,
        DataSeqElementPtr<sofa::geometry::QuadraticQuad>,
        DataSeqElementPtr<sofa::geometry::QuadraticTetrahedron>,
        DataSeqElementPtr<sofa::geometry::QuadraticHexahedron>,
        DataSeqElementPtr<sofa::geometry::QuadraticPrism>,
        DataSeqElementPtr<sofa::geometry::QuadraticPyramid>
    >;

    AllElements m_container;

    const SeqEdges& getEdges() override;
    const SeqTriangles& getTriangles() override;
    const SeqQuads& getQuads() override;
    const SeqTetrahedra& getTetrahedra() override;
    const SeqHexahedra& getHexahedra() override;
    const SeqPrisms& getPrisms() override;
    const SeqPyramids& getPyramids() override;

    template<class ElementType>
    const DataSeqElementPtr<ElementType>& get() const
    {
        return std::get<DataSeqElementPtr<ElementType>>(m_container);
    }

    template<class ElementType>
    DataSeqElementPtr<ElementType>& get()
    {
        return std::get<DataSeqElementPtr<ElementType>>(m_container);
    }

protected:

    const void* getElementsRaw(const sofa::geometry::ElementType& elementType) const noexcept override;

    MeshTopologyContainer();

    // A proxy for legacy data members
    template<class ElementType>
    struct SeqElementProxy
    {
        MeshTopologyContainer* m_meshTopologyContainer { nullptr };
        SeqElementProxy(MeshTopologyContainer* meshTopologyContainer) : m_meshTopologyContainer(meshTopologyContainer) {}

        DataSeqElement<ElementType>& toData()
        {
            return *m_meshTopologyContainer->get<ElementType>();
        }

        const DataSeqElement<ElementType>& toData() const
        {
            return *m_meshTopologyContainer->get<ElementType>();
        }

        operator const DataSeqElement<ElementType>&() const
        {
            return toData();
        }

        operator DataSeqElement<ElementType>&()
        {
            return toData();
        }

        SeqElement<ElementType>* beginEdit()
        {
            return toData().beginEdit();
        }

        void endEdit()
        {
            toData().endEdit();
        }

        const SeqElement<ElementType>& getValue() const
        {
            return toData().getValue();
        }

        void setValue(const DataSeqElement<ElementType>& value)
        {
            toData().setValue(value);
        }

        core::BaseData* getParent() const
        {
            return toData().getParent();
        }

        bool setParent(core::BaseData* parent, const std::string& path = std::string())
        {
            return toData().setParent(parent, path);
        }

        void delInput(core::objectmodel::DDGNode* n)
        {
            toData().delInput(n);
        }

        SeqElement<ElementType>* beginWriteOnly()
        {
            return toData().beginWriteOnly();
        }
    };
};

}

namespace sofa::helper
{
template<class ElementType>
auto getWriteOnlyAccessor(sofa::component::topology::container::constant::MeshTopologyContainer::SeqElementProxy<ElementType>& m_container)
{
    return getWriteOnlyAccessor(m_container.toData());
}

template<class ElementType>
auto getWriteAccessor(sofa::component::topology::container::constant::MeshTopologyContainer::SeqElementProxy<ElementType>& m_container)
{
    return getWriteAccessor(m_container.toData());
}

template<class ElementType>
auto getReadAccessor(const sofa::component::topology::container::constant::MeshTopologyContainer::SeqElementProxy<ElementType>& m_container)
{
    return getReadAccessor(m_container.toData());
}

}
