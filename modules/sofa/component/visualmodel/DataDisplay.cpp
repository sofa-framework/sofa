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
    // Prepare texture for legend
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_1D, texture);
    //glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    //glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    topology = this->getContext()->getMeshTopology();
    if (!topology) {
        sout << "No topology information, drawing just points." << sendl;
    }

    this->getContext()->get(colorMap);
    if (!colorMap) {
        serr << "No ColorMap found, using default." << sendl;
        colorMap = ColorMap::getDefault();
    }

    reinit();
}

void DataDisplay::updateVisual()
{
    // TODO: We don't have to do this every time if we can detect when ColorMap
    // was changed.
    prepareLegend();
}

void DataDisplay::prepareLegend()
{
    int width = colorMap->getNbColors();
    unsigned char *data = new unsigned char[ width * 3 ];

    for (int i=0; i<width; i++) {
        ColorMap::Color c = colorMap->getColor(i);
        data[i*3+0] = c[0]*255;
        data[i*3+1] = c[1]*255;
        data[i*3+2] = c[2]*255;
    }

    glBindTexture(GL_TEXTURE_1D, texture);

    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB, width, 0, GL_RGB, GL_UNSIGNED_BYTE,
        data);

    delete data;
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
        // For now just triangles ...
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
    }

    //
    // Draw legend
    //
    // TODO: show the min/max
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT,viewport);
    const int vWidth = viewport[2];
    const int vHeight = viewport[3];

    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_1D);
    glDisable(GL_LIGHTING);
    glDisable(GL_BLEND);

    // Setup orthogonal projection
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, vWidth, vHeight, 0.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glBindTexture(GL_TEXTURE_1D, texture);

    //glBlendFunc(GL_ONE, GL_ONE);
    glColor3f(1.0f, 1.0f, 1.0f);

    glBegin(GL_QUADS);

    glTexCoord1f(1.0);
    glVertex3f(10, 10, 0.0);

    glTexCoord1f(1.0);
    glVertex3f(20, 10, 0.0);

    glTexCoord1f(0.0);
    glVertex3f( 20, 110, 0.0);

    glTexCoord1f(0.0);
    glVertex3f(10, 110, 0.0);

    glEnd();

    // Restore projection matrix
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    // Restore model view matrix
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    // Restore state
    glPopAttrib();
}

} // namespace visualmodel

} // namespace component

} // namespace sofa
