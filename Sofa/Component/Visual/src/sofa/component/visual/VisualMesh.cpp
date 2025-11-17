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
#include <sofa/component/visual/VisualMesh.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa::component::visual
{

void registerVisualMesh(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Render a mesh")
        .add<VisualMesh>()
    );
}

VisualMesh::VisualMesh()
    : d_position(initData(&d_position, "position", "The position of the vertices of mesh"))
    , d_elementSpace(initData(&d_elementSpace, 0.333_sreal, "elementSpace",
                              "The space between element (scalar between 0 and 1)"))
    , l_topology(initLink("topology", "Link to a topology containing elements"))
{}

void VisualMesh::init()
{
    VisualModel::init();

    if (!this->isComponentStateInvalid())
    {
        this->validateTopology();
    }

    if (!this->isComponentStateInvalid())
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    }
}

void VisualMesh::drawTetrahedra(helper::visual::DrawTool* drawTool)
{
    if (!l_topology)
        return;

    const auto& elements = l_topology->getTetrahedra();
    const auto& facets = l_topology->getTriangles();

    static constexpr std::size_t NumberTrianglesInTetrahedron = 4;
    static constexpr std::size_t NumberVerticesInTriangle = 3;
    m_renderedPoints.resize(elements.size() * NumberTrianglesInTetrahedron * NumberVerticesInTriangle);
    m_renderedColors.resize(elements.size() * NumberTrianglesInTetrahedron * NumberVerticesInTriangle);

    const auto elementSpace = d_elementSpace.getValue();
    const auto& positionAccessor = sofa::helper::getReadAccessor(d_position);

    auto pointsIt = m_renderedPoints.begin();
    auto colorIt = m_renderedColors.begin();

    for (sofa::Size i = 0; i < elements.size(); ++i)
    {
        const auto& element = elements[i];
        const auto& facetsInElement = l_topology->getTrianglesInTetrahedron(i);
        assert(facetsInElement.size() == NumberTrianglesInTetrahedron);

        const sofa::type::Vec3 center = elementCenter(positionAccessor.ref(), element);

        static constexpr std::array colors{
            sofa::type::RGBAColor::blue(),
            sofa::type::RGBAColor::black(),
            sofa::type::RGBAColor(0.0f, 0.5f, 1.0f, 1.0f),
            sofa::type::RGBAColor::cyan(),
        };

        setPointsAndColors(facetsInElement, facets, positionAccessor.ref(), center, elementSpace, pointsIt, colorIt, colors);
    }
    drawTool->drawTriangles(m_renderedPoints, m_renderedColors);
}

void VisualMesh::drawHexahedra(helper::visual::DrawTool* drawTool)
{
    if (!l_topology)
        return;

    const auto& elements = l_topology->getHexahedra();
    const auto& facets = l_topology->getQuads();

    static constexpr std::size_t NumberQuadsInHexahedron = 6;
    static constexpr std::size_t NumberVerticesInQuad = 4;
    m_renderedPoints.resize(elements.size() * NumberQuadsInHexahedron * NumberVerticesInQuad);
    m_renderedColors.resize(elements.size() * NumberQuadsInHexahedron * NumberVerticesInQuad);

    const auto elementSpace = d_elementSpace.getValue();
    const auto& positionAccessor = sofa::helper::getReadAccessor(d_position);

    auto pointsIt = m_renderedPoints.begin();
    auto colorIt = m_renderedColors.begin();

    for (sofa::Size i = 0; i < elements.size(); ++i)
    {
        const auto& element = elements[i];
        const auto& facetsInElement = l_topology->getQuadsInHexahedron(i);
        assert(facetsInElement.size() == NumberQuadsInHexahedron);

        const sofa::type::Vec3 center = elementCenter(positionAccessor.ref(), element);

        static constexpr std::array colors {
            sofa::type::RGBAColor(0.7f,0.7f,0.1f,1.f),
            sofa::type::RGBAColor(0.7f,0.0f,0.0f,1.f),
            sofa::type::RGBAColor(0.0f,0.7f,0.0f,1.f),
            sofa::type::RGBAColor(0.0f,0.0f,0.7f,1.f),
            sofa::type::RGBAColor(0.1f,0.7f,0.7f,1.f),
            sofa::type::RGBAColor(0.7f,0.1f,0.7f,1.f)
        };

        setPointsAndColors(facetsInElement, facets, positionAccessor.ref(), center, elementSpace, pointsIt, colorIt, colors);
    }
    drawTool->drawQuads(m_renderedPoints, m_renderedColors);
}

void VisualMesh::doDrawVisual(const core::visual::VisualParams* vparams)
{
    auto* drawTool = vparams->drawTool();

    vparams->drawTool()->disableLighting();

    drawTetrahedra(drawTool);
    drawHexahedra(drawTool);
}

void VisualMesh::validateTopology()
{
    if (l_topology.empty())
    {
        msg_info() << "Link to Topology container should be set to ensure right behavior. First "
                      "Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    if (l_topology == nullptr)
    {
        msg_error() << "No topology component found at path: " << this->l_topology.getLinkedPath()
                    << ", nor in current context: " << this->getContext()->name
                    << ". Object must have a BaseMeshTopology. "
                    << "The list of available BaseMeshTopology components is: "
                    << sofa::core::ObjectFactory::getInstance()
                           ->listClassesDerivedFrom<sofa::core::topology::BaseMeshTopology>();
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
}

}  // namespace sofa::component::visual
