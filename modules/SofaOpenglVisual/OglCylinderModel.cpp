/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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

#include <map>
#include <sofa/helper/gl/template.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/topology/TopologyChange.h>

#include <sofa/core/loader/VoxelLoader.h>

#include <SofaOpenglVisual/OglCylinderModel.h>
#include <sofa/core/visual/VisualParams.h>

#include <SofaBaseTopology/TopologyData.inl>

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
    : radius(initData(&radius, 1.0f, "radius", "Radius of the cylinder.")),
      color(initData(&color, std::string("white"), "color", "Color of the cylinders.")),
      _topology(NULL),
      _mstate(NULL)
	  // , pointData(initData(&pointData, "pointData", "scalar field modulating point colors"))
{
}

OglCylinderModel::~OglCylinderModel()
{
}

void OglCylinderModel::init()
{
    getContext()->get(_topology);
    if(_topology)
        _mstate = dynamic_cast<core::behavior::BaseMechanicalState*>(_topology->getContext()->getMechanicalState());
    else
        getContext()->get(_mstate);

    VisualModel::init();

    reinit();

    updateVisual();
}

void OglCylinderModel::reinit()
{
    setColor(color.getValue());
}

void OglCylinderModel::draw(const core::visual::VisualParams* vparams)
{
    if(!vparams->displayFlags().getShowVisualModels()) return;

    // glPushAttrib(GL_ENABLE_BIT);

    vparams->drawTool()->setLightingEnabled(true);
	Real _radius = radius.getValue();

	Vec<4,float> col( r, g, b, a );

	if(_topology){
		for(BaseMeshTopology::SeqEdges::const_iterator it = _topology->getEdges().begin(); it != _topology->getEdges().end(); ++it){
			int pid1 = (*it)[0];
			int pid2 = (*it)[1];
			Coord p1((Real)_mstate->getPX(pid1),(Real)_mstate->getPY(pid1),(Real)_mstate->getPZ(pid1));
			Coord p2((Real)_mstate->getPX(pid2),(Real)_mstate->getPY(pid2),(Real)_mstate->getPZ(pid2));
			vparams->drawTool()->drawCylinder(p1,p2,_radius,col);
		}
	}

    // glPopAttrib();
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
        serr << "Unknown color "<<color<<sendl;
        return;
    }
    setColor(r,g,b,a);
}


} // namespace visualmodel

} // namespace component

} // namespace sofa

