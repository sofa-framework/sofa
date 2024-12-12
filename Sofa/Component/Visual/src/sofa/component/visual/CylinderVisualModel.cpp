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
#include <sofa/component/visual/CylinderVisualModel.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::visual
{

void registerCylinderVisualModel(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Visualize a set of cylinders.")
        .add< CylinderVisualModel >());
}

CylinderVisualModel::CylinderVisualModel()
    : radius(initData(&radius, 1.0f, "radius", "Radius of the cylinder.")),
      color(initData(&color, sofa::type::RGBAColor::white(), "color", "Color of the cylinders."))
    , d_edges(initData(&d_edges,"edges","List of edge indices"))
{
}

CylinderVisualModel::~CylinderVisualModel() = default;

void CylinderVisualModel::init()
{
    VisualModel::init();

    reinit();
}

void CylinderVisualModel::doDrawVisual(const core::visual::VisualParams* vparams)
{
    const VecCoord& pos = this->read( core::vec_id::read_access::position )->getValue();

    auto* drawTool = vparams->drawTool();

    drawTool->setLightingEnabled(true);

    const float _radius = radius.getValue();
    const sofa::type::RGBAColor& col = color.getValue();

    const SeqEdges& edges = d_edges.getValue();

    for(const auto& edge : edges)
    {
        const Coord& p1 = pos[edge[0]];
        const Coord& p2 = pos[edge[1]];

        drawTool->drawCylinder(p1, p2, _radius, col);
    }
}

void CylinderVisualModel::exportOBJ(std::string name, std::ostream* out, std::ostream* /*mtl*/, Index& vindex, Index& /*nindex*/, Index& /*tindex*/, int& /*count*/)
{
    const VecCoord& x = this->read( core::vec_id::read_access::position )->getValue();
    const SeqEdges& edges = d_edges.getValue();

    const int nbv = int(x.size());

    *out << "g "<<name<<"\n";

    for( int i=0 ; i<nbv; i++ )
        *out << "v "<< std::fixed << x[i][0]<<' '<< std::fixed <<x[i][1]<<' '<< std::fixed <<x[i][2]<<'\n';

    for( size_t i = 0 ; i < edges.size() ; i++ )
        *out << "f " << edges[i][0]+vindex+1 << " " << edges[i][1]+vindex+1 << '\n';

    *out << std::endl;

    vindex += nbv;
}

} // namespace sofa::component::visual
