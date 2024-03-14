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
#include <sofa/gl/component/rendering3d/OglCylinderModel.h>
#include <sofa/gl/component/rendering3d/config.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/TopologyData.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa::gl::component::rendering3d
{

int OglCylinderModelClass = core::RegisterObject("A simple visualization for set of cylinder.")
        .add< OglCylinderModel >()
        ;

using namespace sofa::defaulttype;
using namespace sofa::core::topology;

OglCylinderModel::OglCylinderModel()
    : radius(initData(&radius, 1.0f, "radius", "Radius of the cylinder.")),
      color(initData(&color, sofa::type::RGBAColor(1.0,1.0,1.0,1.0), "color", "Color of the cylinders."))
    , d_edges(initData(&d_edges,"edges","List of edge indices"))
      // , pointData(initData(&pointData, "pointData", "scalar field modulating point colors"))
{
}

OglCylinderModel::~OglCylinderModel()
{
}

void OglCylinderModel::init()
{
    VisualModel::init();

    reinit();

    updateVisual();
}

void OglCylinderModel::reinit()
{
}

void OglCylinderModel::doDrawVisual(const core::visual::VisualParams* vparams)
{
    const VecCoord& pos = this->read( core::ConstVecCoordId::position() )->getValue();

    vparams->drawTool()->setLightingEnabled(true);
    const float _radius = radius.getValue();

    const sofa::type::RGBAColor col( r, g, b, a );

    const SeqEdges& edges = d_edges.getValue();

    for(auto edge : edges)
    {
        const Coord& p1 = pos[edge[0]];
        const Coord& p2 = pos[edge[1]];

        vparams->drawTool()->drawCylinder(p1,p2,_radius,col);
    }
}


void OglCylinderModel::setColor(float r, float g, float b, float a)
{
    this->r = r;
    this->g = g;
    this->b = b;
    this->a = a;
}

static int hexval(char c)
{
    if (c>='0' && c<='9') return c-'0';
    else if (c>='a' && c<='f') return (c-'a')+10;
    else if (c>='A' && c<='F') return (c-'A')+10;
    else return 0;
}

void OglCylinderModel::setColor(std::string color)
{
    if (color.empty()) return;
    float r = 1.0f;
    float g = 1.0f;
    float b = 1.0f;
    float a = 1.0f;
    if (color[0]>='0' && color[0]<='9')
    {
        sscanf(color.c_str(),"%f %f %f %f", &r, &g, &b, &a);
    }
    else if (color[0]=='#' && color.length()>=7)
    {
        r = (hexval(color[1])*16+hexval(color[2]))/255.0f;
        g = (hexval(color[3])*16+hexval(color[4]))/255.0f;
        b = (hexval(color[5])*16+hexval(color[6]))/255.0f;
        if (color.length()>=9)
            a = (hexval(color[7])*16+hexval(color[8]))/255.0f;
    }
    else if (color[0]=='#' && color.length()>=4)
    {
        r = (hexval(color[1])*17)/255.0f;
        g = (hexval(color[2])*17)/255.0f;
        b = (hexval(color[3])*17)/255.0f;
        if (color.length()>=5)
            a = (hexval(color[4])*17)/255.0f;
    }
    else if (color == "white")    { r = 1.0f; g = 1.0f; b = 1.0f; }
    else if (color == "black")    { r = 0.0f; g = 0.0f; b = 0.0f; }
    else if (color == "red")      { r = 1.0f; g = 0.0f; b = 0.0f; }
    else if (color == "green")    { r = 0.0f; g = 1.0f; b = 0.0f; }
    else if (color == "blue")     { r = 0.0f; g = 0.0f; b = 1.0f; }
    else if (color == "cyan")     { r = 0.0f; g = 1.0f; b = 1.0f; }
    else if (color == "magenta")  { r = 1.0f; g = 0.0f; b = 1.0f; }
    else if (color == "yellow")   { r = 1.0f; g = 1.0f; b = 0.0f; }
    else if (color == "gray")     { r = 0.5f; g = 0.5f; b = 0.5f; }
    else
    {
        msg_error() << "Unknown color " << color;
        return;
    }
    setColor(r,g,b,a);
}

void OglCylinderModel::exportOBJ(std::string name, std::ostream* out, std::ostream* /*mtl*/, Index& vindex, Index& /*nindex*/, Index& /*tindex*/, int& /*count*/)
{
    const VecCoord& x = this->read( core::ConstVecCoordId::position() )->getValue();
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

} // namespace sofa::gl::component::rendering3d
