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

#if defined(WIN32) && (_MSC_VER < 1800) // for all version anterior to Visual Studio 2013
# include <float.h>
# define isnan(x)  (_isnan(x))
#else
# include <cmath>
# define isnan(x) (std::isnan(x))
#endif

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

#include <SofaOpenglVisual/DataDisplay.h>

#include <SofaBaseTopology/TriangleSetTopologyContainer.h>


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


void DataDisplay::updateVisual()
{
    computeNormals();
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
            Vec4f color = isnan(clData[i])
                ? f_colorNaN.getValue()
                : eval(clData[i]);
            Triangle t = tt->getTriangle(i);
            vparams->drawTool()->drawTriangle(
                x[ t[0] ], x[ t[1] ], x[ t[2] ],
                m_normals[ t[0] ], m_normals[ t[1] ], m_normals[ t[2] ],
                color, color, color);
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
            Vec4f color = isnan(ptData[i])
                ? f_colorNaN.getValue()
                : eval(ptData[i]);
            vparams->drawTool()->drawPoint(x[i], color);
        }
        glEnd();

    } else if (bDrawPointData) {
        ColorMap::evaluator<Real> eval = colorMap->getEvaluator(min, max);

        glPushAttrib ( GL_LIGHTING_BIT );
        glEnable ( GL_LIGHTING );
        glEnable( GL_COLOR_MATERIAL );

        // Triangles
        glBegin(GL_TRIANGLES);
        for (int i=0; i<topology->getNbTriangles(); ++i)
        {
            const Triangle &t = topology->getTriangle(i);
            Vec4f color[3];
            for (int j=0; j<3; j++) {
                color[j] = isnan(ptData[t[j]])
                    ? f_colorNaN.getValue()
                    : eval(ptData[t[j]]);
            }

            vparams->drawTool()->drawTriangle(
                x[ t[0] ], x[ t[1] ], x[ t[2] ],
                m_normals[ t[0] ], m_normals[ t[1] ], m_normals[ t[2] ],
                color[0], color[1], color[2]);
        }
        glEnd();

        // Quads
        glBegin(GL_QUADS);
        for (int i=0; i<topology->getNbQuads(); ++i)
        {
            const Quad &q = topology->getQuad(i);
            Vec4f color[4];
            for (int j=0; j<4; j++)
            {
                color[j] = isnan(ptData[q[j]])
                ? f_colorNaN.getValue()
                : eval(ptData[q[j]]);
            }

            vparams->drawTool()->drawQuad(
                x[ q[0] ], x[ q[1] ], x[ q[2] ], x[ q[3] ],
                m_normals[ q[0] ], m_normals[ q[1] ], m_normals[ q[2] ], m_normals[ q[3] ],
                color[0], color[1], color[2], color[3]);

        }
        glEnd();

        glPopAttrib();
    }

}

void DataDisplay::computeNormals()
{
    if( !topology ) return;
    const VecCoord& x = this->read(sofa::core::ConstVecCoordId::position())->getValue();

    m_normals.resize(x.size(),Vec3f(0,0,0));

    for (int i=0; i<topology->getNbTriangles(); ++i)
    {
        const Triangle &t = topology->getTriangle(i);

        Coord edge0 = (x[t[1]]-x[t[0]]); edge0.normalize();
        Coord edge1 = (x[t[2]]-x[t[0]]); edge1.normalize();
        Real triangleSurface = edge0*edge1*0.5;
        Vec3f triangleNormal = cross( edge0, edge1 ) * triangleSurface;

        for( int i=0 ; i<3 ; ++i )
        {
            m_normals[t[i]] += triangleNormal;
        }
    }

    for (int i=0; i<topology->getNbQuads(); ++i)
    {
        const Quad &q = topology->getQuad(i);

        for( int i=0 ; i<4 ; ++i )
        {
            Coord edge0 = (x[q[(i+1)%3]]-x[q[i]]); edge0.normalize();
            Coord edge1 = (x[q[(i+2)%3]]-x[q[i]]); edge1.normalize();
            Real triangleSurface = edge0*edge1*0.5;
            Vec3f quadNormal = cross( edge0, edge1 ) * triangleSurface;

            m_normals[q[i]] += quadNormal;
        }
    }

    // normalization
    for (size_t i=0; i<x.size(); ++i)
    {
        m_normals[i].normalize();
    }



}

} // namespace visualmodel

} // namespace component

} // namespace sofa
