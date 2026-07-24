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
    , d_elementSpace(initData(&d_elementSpace, 0.15_sreal, "elementSpace",
                              "The space between element (scalar between 0 and 1)"))
    , d_lighting(initData(&d_lighting, true, "lighting", "If true, light is simulated on the mesh. Otherwise, no lighting effect."))
    , d_vertexValues(initData(&d_vertexValues, "vertexValues", "Optional list of values associated to the vertices of the mesh. If provided, the values are converted to colors."))
    , d_colorMap(initData(&d_colorMap, *sofa::helper::ColorMap::getDefault(), "colorMap", "Color map used to convert vertex values to colors."))
    , l_topology(initLink("topology", "Link to a topology containing elements"))
{
}

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

void VisualMesh::computeBBox(const core::ExecParams* exec_params, bool onlyVisible)
{
    if (!d_enable.getValue())
        return;

    if (onlyVisible && !sofa::core::visual::VisualParams::defaultInstance()->displayFlags().getShowVisualModels())
        return;

    type::BoundingBox bbox;
    for (const auto& i : sofa::helper::getReadAccessor(d_position))
    {
        bbox.include(i);
    }
    this->f_bbox.setValue(bbox);
}

void VisualMesh::doDrawVisual(const core::visual::VisualParams* vparams)
{
    auto* drawTool = vparams->drawTool();

    if (d_lighting.getValue())
        vparams->drawTool()->enableLighting();
    else
        vparams->drawTool()->disableLighting();

    const auto positionAccessor = sofa::helper::getReadAccessor(d_position);

    if (d_vertexValues.isSet())
    {
        const auto vertexValuesAccessor = sofa::helper::getReadAccessor(d_vertexValues);
        if (vertexValuesAccessor.size() >= positionAccessor.size())
        {
            const auto& colorMap = d_colorMap.getValue();
            const auto [minIt, maxIt] = std::minmax_element(vertexValuesAccessor.begin(), vertexValuesAccessor.end());
            auto colorEvaluator = colorMap.getEvaluator<SReal>(*minIt, *maxIt);

            sofa::type::vector<sofa::type::RGBAColor> vertexColors;
            for (const auto& v : vertexValuesAccessor)
            {
                vertexColors.push_back(colorEvaluator(v));
            }

            m_drawColoredMesh.setElementSpace(d_elementSpace.getValue());
            m_drawColoredMesh.draw(drawTool, positionAccessor.ref(), vertexColors, l_topology.get());
            return;
        }
    }

    m_drawMesh.setElementSpace(d_elementSpace.getValue());
    m_drawMesh.draw(drawTool, positionAccessor.ref(), l_topology.get());
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
