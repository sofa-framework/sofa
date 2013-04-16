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

#include <cmath>

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/component/visualmodel/DataDisplay.h>

#include <sofa/component/topology/TriangleSetTopologyContainer.h>


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
    const VecCellData &clData = f_cellData.getValue();

    bool bDrawPointData = false;
    bool bDrawCellData = false;

    // For now support only triangular topology
    topology::TriangleSetTopologyContainer* tt = dynamic_cast<topology::TriangleSetTopologyContainer*>(topology);

    // Safety checks
    // TODO: can this go to updateVisual()?
    if (ptData.size() > 0) {
        if (ptData.size() != x.size()) {
            serr << "Size of pointData does not mach number of nodes" << sendl;
        } else {
            bDrawPointData = true;
        }
    } else if (clData.size() > 0) {
        if (!topology || !tt) {
            serr << "Triangular topology is necessary for drawing cell data" << sendl;
        } else if ((int)clData.size() != tt->getNbTriangles()) {
            serr << "Size of cellData does not match number of triangles" << sendl;
        } else {
            bDrawCellData = true;
        }
    }

    // Range for points
    Real min=0.0, max=0.0;
    if (bDrawPointData) {
        VecPointData::const_iterator i = ptData.begin();
        min = *i;
        max = *i;
        while (++i != ptData.end()) {
            if (min > *i) min = *i;
            if (max < *i) max = *i;
        }
    }

    // Range for cells
    if (bDrawCellData) {
        VecCellData::const_iterator i = clData.begin();
        min = *i;
        max = *i;
        while (++i != clData.end()) {
            if (min > *i) min = *i;
            if (max < *i) max = *i;
        }
    }
    if (max > oldMax) oldMax = max;
    if (min < oldMin) oldMin = min;

    if (f_maximalRange.getValue()) {
        max = oldMax;
        min = oldMin;
    }

    vparams->drawTool()->setLightingEnabled(false);

    if (bDrawCellData) {
        // Triangles
        ColorMap::evaluator<Real> eval = colorMap->getEvaluator(min, max);
        int nbTriangles = tt->getNbTriangles();
        glBegin(GL_TRIANGLES);
        for (int i=0; i<nbTriangles; i++)
        {
            Vec4f color = std::isnan(clData[i])
                ? f_colorNaN.getValue()
                : eval(clData[i]);
            Triangle tri = tt->getTriangle(i);
            vparams->drawTool()->drawTriangle(
                x[ tri[0] ], x[ tri[1] ], x[ tri[2] ],
                Vector3(0,0,0), color);
        }
        glEnd();
    }

    if ((bDrawCellData || !topology) && bDrawPointData) {
        ColorMap::evaluator<Real> eval = colorMap->getEvaluator(min, max);
        // Just the points
        glPointSize(10);
        glBegin(GL_POINTS);
        for (unsigned int i=0; i<x.size(); ++i)
        {
            Vec4f color = std::isnan(ptData[i])
                ? f_colorNaN.getValue()
                : eval(ptData[i]);
            vparams->drawTool()->drawPoint(x[i], color);
        }
        glEnd();

    } else if (bDrawPointData) {
        ColorMap::evaluator<Real> eval = colorMap->getEvaluator(min, max);
        // Triangles
        glBegin(GL_TRIANGLES);
        for (int i=0; i<topology->getNbTriangles(); ++i)
        {
            const Triangle &t = topology->getTriangle(i);
            Vec4f color[3];
            for (int j=0; j<3; j++) {
                color[j] = std::isnan(ptData[t[j]])
                    ? f_colorNaN.getValue()
                    : eval(ptData[t[j]]);
            }
            vparams->drawTool()->drawTriangle(
                x[ t[0] ], x[ t[1] ], x[ t[2] ],
                Vector3(0,0,0), color[0], color[1], color[2]);
        }
        glEnd();
        // Quads
        glBegin(GL_QUADS);
        for (int i=0; i<topology->getNbQuads(); ++i)
        {
            const Quad &q = topology->getQuad(i);
            for (int j=0; j<4; j++) {
                Vec4f color = std::isnan(ptData[q[j]])
                    ? f_colorNaN.getValue()
                    : eval(ptData[q[j]]);
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
