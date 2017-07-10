/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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

#include <sofa/core/ObjectFactory.h>
#include <SofaOpenglVisual/OglCylinderModel.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

SOFA_DECL_CLASS(OglCylinderModel)

int OglCylinderModelClass = core::RegisterObject("A simple visualization for set of cylinder.")
        .add< OglCylinderModel >()
        ;

using namespace sofa::defaulttype;
using namespace sofa::core::topology;

OglCylinderModel::OglCylinderModel()
    : radius(initData(&radius, 1.0f, "radius", "Radius of the cylinder."))
    , d_color(initData(&d_color, defaulttype::RGBAColor(1.0,1.0,1.0,1.0), "color", "Color of the cylinders."))
    , d_depthTest(initData(&d_depthTest, true, "depthTest", "perform depth test"))
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

void OglCylinderModel::drawVisual(const core::visual::VisualParams* vparams)
{
    if(!vparams->displayFlags().getShowVisualModels()) return;

    const VecCoord& pos = this->read( core::ConstVecCoordId::position() )->getValue();

    const bool& depthTest = d_depthTest.getValue();
    if( !depthTest )
    {
        glPushAttrib(GL_ENABLE_BIT);
        glDisable(GL_DEPTH_TEST);
    }

    vparams->drawTool()->setLightingEnabled(true);
    Real _radius = radius.getValue();

    const defaulttype::RGBAColor& col = d_color.getValue();

    const SeqEdges& edges = d_edges.getValue();

    for( SeqEdges::const_iterator it=edges.begin(), itend=edges.end() ; it !=itend ; ++it )
    {
        const Coord& p1 = pos[(*it)[0]];
        const Coord& p2 = pos[(*it)[1]];

        vparams->drawTool()->drawCylinder(p1,p2,_radius,col);
    }

    if( !depthTest )
        glPopAttrib();
}


void OglCylinderModel::exportOBJ(std::string name, std::ostream* out, std::ostream* /*mtl*/, int& vindex, int& /*nindex*/, int& /*tindex*/, int& /*count*/)
{
    const VecCoord& x = this->read( core::ConstVecCoordId::position() )->getValue();
    const SeqEdges& edges = d_edges.getValue();

    int nbv = x.size();

    *out << "g "<<name<<"\n";

    for( int i=0 ; i<nbv; i++ )
        *out << "v "<< std::fixed << x[i][0]<<' '<< std::fixed <<x[i][1]<<' '<< std::fixed <<x[i][2]<<'\n';

    for( size_t i = 0 ; i < edges.size() ; i++ )
        *out << "f " << edges[i][0]+vindex+1 << " " << edges[i][1]+vindex+1 << '\n';

    *out << sendl;

    vindex += nbv;
}



} // namespace visualmodel

} // namespace component

} // namespace sofa

