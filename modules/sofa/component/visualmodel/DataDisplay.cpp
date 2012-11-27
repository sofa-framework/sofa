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
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/component/visualmodel/DataDisplay.h>


namespace sofa
{

namespace component
{

namespace visualmodel
{

using namespace sofa::defaulttype;
using sofa::component::visualmodel::ColorMap;

SOFA_DECL_CLASS(DataDisplay)

int DataDisplayClass = core::RegisterObject("Rendering of meshes colored by data")
        .add< DataDisplay >()
        ;

void DataDisplay::init()
{
    topology = this->getContext()->getMeshTopology();
    if (!topology) {
        sout << "No topology information, drawing just points." << sendl;
    }

    this->getContext()->get(colorMap);
    if (!colorMap) {
        serr << "No ColorMap found, using default." << sendl;
        colorMap = ColorMap::getDefault();
    }
}

void DataDisplay::drawVisual(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowVisualModels()) return;

    const VecCoord& x = this->read(sofa::core::ConstVecCoordId::position())->getValue();
    const VecPointData &ptData = f_pointData.getValue();

    bool bDrawPointData = false;

    // Safety checks
    if (ptData.size() > 0) {
        if (ptData.size() != x.size()) {
            serr << "Size of pointData doesn't mach number of nodes" << sendl;
        } else {
            bDrawPointData = true;
        }
    }

    Real ptMin=0.0, ptMax=0.0;
    if (bDrawPointData) {
        VecPointData::const_iterator i = ptData.begin();
        ptMin = *i;
        ptMax = *i;
        while (++i != ptData.end()) {
            if (ptMin > *i) ptMin = *i;
            if (ptMax < *i) ptMax = *i;
        }
    }

    glDisable(GL_LIGHTING);

    if (!topology && bDrawPointData) {
        ColorMap::evaluator<Real> eval = colorMap->getEvaluator(ptMin, ptMax);
        // Just the points
        glPointSize(10);
        glBegin(GL_POINTS);
        for (unsigned int i=0; i<x.size(); ++i)
        {
            Vec4f color = eval(ptData[i]);
            glColor4f(color[0], color[1], color[2], color[3]);
            //glColor4f(1.0, 1.0, 1.0, 1.0);
            glVertex3f(x[i][0], x[i][1], x[i][2]);
        }
        glEnd();

    } else if (bDrawPointData) {
        ColorMap::evaluator<Real> eval = colorMap->getEvaluator(ptMin, ptMax);
        // Triangles
        glBegin(GL_TRIANGLES);
        for (int i=0; i<topology->getNbTriangles(); ++i)
        {
            const Triangle &t = topology->getTriangle(i);
            for (int j=0; j<3; j++) {
                Vec4f color = eval(ptData[t[j]]);
                glColor4f(color[0], color[1], color[2], color[3]);
                //glColor4f(1.0, 1.0, 1.0, 1.0);
                glVertex3f(
                    x[ t[j] ][0],
                    x[ t[j] ][1],
                    x[ t[j] ][2]);
            }
        }
        glEnd();
        // Quads
        glBegin(GL_QUADS);
        for (int i=0; i<topology->getNbQuads(); ++i)
        {
            const Quad &q = topology->getQuad(i);
            for (int j=0; j<4; j++) {
                Vec4f color = eval(ptData[q[j]]);
                glColor4f(color[0], color[1], color[2], color[3]);
                //glColor4f(1.0, 1.0, 1.0, 1.0);
                glVertex3f(
                    x[ q[j] ][0],
                    x[ q[j] ][1],
                    x[ q[j] ][2]);
            }
        }
        glEnd();
    }
}

} // namespace visualmodel

} // namespace component

} // namespace sofa
