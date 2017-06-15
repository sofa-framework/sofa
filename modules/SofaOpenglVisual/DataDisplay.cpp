/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
using sofa::component::visualmodel::OglColorMap;

SOFA_DECL_CLASS(DataDisplay)

int DataDisplayClass = core::RegisterObject("Rendering of meshes colored by data")
        .add< DataDisplay >()
        ;

void DataDisplay::init()
{
    topology = this->getContext()->getMeshTopology();
    if (!topology)
        sout << "No topology information, drawing just points." << sendl;
    else
        sout << "using topology "<< topology->getPathName() << sendl;

    this->getContext()->get(colorMap);
    if (!colorMap) {
        serr << "No ColorMap found, using default." << sendl;
        colorMap = OglColorMap::getDefault();
    }
}


void DataDisplay::updateVisual()
{
    computeNormals();
}

void DataDisplay::drawVisual(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowVisualModels()) return;

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    const VecCoord& x = this->read(sofa::core::ConstVecCoordId::position())->getValue();
    const VecPointData &ptData = f_pointData.getValue();
    const VecCellData &triData = f_triangleData.getValue();
    const VecCellData &quadData = f_quadData.getValue();
    const VecPointData &pointTriData = f_pointTriangleData.getValue();
    const VecPointData &pointQuadData = f_pointQuadData.getValue();

    bool bDrawPointData = false;
    bool bDrawCellData = false;


    // Safety checks
    // TODO: can this go to updateVisual()?
    if (ptData.size() > 0) {
        if (ptData.size() != x.size()) {
            serr << "Size of pointData does not mach number of nodes" << sendl;
        } else {
            bDrawPointData = true;
        }
    } else if (triData.size() > 0 || quadData.size()>0 ) {
        if (!topology ) {
            serr << "Topology is necessary for drawing cell data" << sendl;
        } else if ((int)triData.size() != topology->getNbTriangles()) {
            serr << "Size of triangleData does not match number of triangles" << sendl;
        } else if ((int)quadData.size() != topology->getNbQuads()) {
            serr << "Size of quadData does not match number of quads" << sendl;
        } else {
            bDrawCellData = true;
        }
    } else if (pointTriData.size()>0 || pointQuadData.size()>0) {
        if (!topology ) {
            serr << "Topology is necessary for drawing cell data" << sendl;
        } else if ((int)pointTriData.size() != topology->getNbTriangles()*3) {
            serr << "Size of pointTriData does not match number of triangles" << sendl;
        } else if ((int)pointQuadData.size() != topology->getNbQuads()*4) {
            serr << "Size of pointQuadData does not match number of quads" << sendl;
        } else {
            bDrawCellData = true;
        }
    }

    // Range for points
    float& min = *d_currentMin.beginWriteOnly();
    float& max = *d_currentMax.beginWriteOnly();
    min = max = 0;
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
        if(!triData.empty()) {
        VecCellData::const_iterator i = triData.begin();
        min = *i;
        max = *i;
        while (++i != triData.end()) {
            if (min > *i) min = *i;
            if (max < *i) max = *i;
        }
        }
        if(!quadData.empty()) {
        VecCellData::const_iterator i = quadData.begin();
        min = *i;
        max = *i;
        while (++i != quadData.end()) {
            if (min > *i) min = *i;
            if (max < *i) max = *i;
        }
        }
        if(!pointTriData.empty()) {
        VecPointData::const_iterator i = pointTriData.begin();
        min = *i;
        max = *i;
        while (++i != pointTriData.end()) {
            if (min > *i) min = *i;
            if (max < *i) max = *i;
        }
        }
        if(!pointQuadData.empty()) {
        VecPointData::const_iterator i = pointQuadData.begin();
        min = *i;
        max = *i;
        while (++i != pointQuadData.end()) {
            if (min > *i) min = *i;
            if (max < *i) max = *i;
        }
        }
    }

    if(m_normals.size() != x.size())
        computeNormals();

    if( d_userRange.getValue()[0] < d_userRange.getValue()[1] )
    {
        if( f_maximalRange.getValue() )
        {
            if( max > d_userRange.getValue()[1] ) max=d_userRange.getValue()[1];
            if( min < d_userRange.getValue()[0] ) min=d_userRange.getValue()[0];
        }
        else
        {
            max=d_userRange.getValue()[1];
            min=d_userRange.getValue()[0];
        }
    }


    if (max > oldMax) oldMax = max;
    if (min < oldMin) oldMin = min;

    if (f_maximalRange.getValue()) {
        max = oldMax;
        min = oldMin;
    }
    d_currentMin.endEdit();
    d_currentMax.endEdit();

    glPushAttrib ( GL_LIGHTING_BIT );

    static const Vec4f emptyColor = Vec4f();
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, emptyColor.ptr());
    glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, emptyColor.ptr());

    if( d_shininess.getValue()>=0 )
    {
        static const Vec4f specular = Vec4f(.5,.5,.5,1);
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular.ptr());
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, d_shininess.getValue());
    }
    else
    {
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, emptyColor.ptr());
    }

    if (bDrawCellData) {

        glDisable( GL_LIGHTING );
        helper::ColorMap::evaluator<Real> eval = colorMap->getEvaluator(min, max);

        if( !triData.empty() )
        {
            // Triangles
            int nbTriangles = topology->getNbTriangles();
            glBegin(GL_TRIANGLES);
            for (int i=0; i<nbTriangles; i++)
            {
                Vec4f color = isnan(triData[i])
                    ? f_colorNaN.getValue()
                    : defaulttype::RGBAColor::fromVec4(eval(triData[i]));
                const Triangle& t = topology->getTriangle(i);
                vparams->drawTool()->drawTriangle(
                    x[ t[0] ], x[ t[1] ], x[ t[2] ],
                    m_normals[ t[0] ], m_normals[ t[1] ], m_normals[ t[2] ],
                    color, color, color);
            }
            glEnd();
        }
        else if( !pointTriData.empty() )
        {
            glEnable( GL_LIGHTING );
            // Triangles
            int nbTriangles = topology->getNbTriangles();
            glBegin(GL_TRIANGLES);
            for (int i=0; i<nbTriangles; i++)
            {
                Vec4f color0 = isnan(pointTriData[i*3])
                    ? f_colorNaN.getValue()
                    : defaulttype::RGBAColor::fromVec4(eval(pointTriData[i*3]));
                Vec4f color1 = isnan(pointTriData[i*3+1])
                        ? f_colorNaN.getValue()
                        : defaulttype::RGBAColor::fromVec4(eval(pointTriData[i*3+1]));
                Vec4f color2 = isnan(pointTriData[i*3+2])
                    ? f_colorNaN.getValue()
                    : defaulttype::RGBAColor::fromVec4(eval(pointTriData[i*3+2]));
                const Triangle& t = topology->getTriangle(i);

                glNormal3fv(m_normals[t[0]].ptr());
                glMaterialfv(GL_FRONT,GL_DIFFUSE,color0.ptr());
                helper::gl::glVertexNv<3>(x[t[0]].ptr());

                glNormal3fv(m_normals[t[1]].ptr());
                glMaterialfv(GL_FRONT,GL_DIFFUSE,color1.ptr());
                helper::gl::glVertexNv<3>(x[t[1]].ptr());

                glNormal3fv(m_normals[t[2]].ptr());
                glMaterialfv(GL_FRONT,GL_DIFFUSE,color2.ptr());
                helper::gl::glVertexNv<3>(x[t[2]].ptr());

            }
            glEnd();
        }

        if( !quadData.empty() )
        {
            glDisable( GL_LIGHTING );
            int nbQuads = topology->getNbQuads();
            glBegin(GL_QUADS);
            for (int i=0; i<nbQuads; i++)
            {
                Vec4f color = isnan(quadData[i])
                    ? f_colorNaN.getValue()
                    : defaulttype::RGBAColor::fromVec4(eval(quadData[i]));
                const Quad& t = topology->getQuad(i);
                vparams->drawTool()->drawQuad(
                    x[ t[0] ], x[ t[1] ], x[ t[2] ], x[ t[3] ],
                    m_normals[ t[0] ], m_normals[ t[1] ], m_normals[ t[2] ], m_normals[ t[3] ],
                    color, color, color, color);
            }
            glEnd();
        }
        else if( !pointQuadData.empty() )
        {
            glEnable( GL_LIGHTING );
            int nbQuads = topology->getNbQuads();
            glBegin(GL_QUADS);
            for (int i=0; i<nbQuads; i++)
            {
                Vec4f color0 = isnan(pointQuadData[i*4])
                    ? f_colorNaN.getValue()
                    : defaulttype::RGBAColor::fromVec4(eval(pointQuadData[i*4]));
                Vec4f color1 = isnan(pointQuadData[i*4+1])
                        ? f_colorNaN.getValue()
                        : defaulttype::RGBAColor::fromVec4(eval(pointQuadData[i*4+1]));
                Vec4f color2 = isnan(pointQuadData[i*4+2])
                    ? f_colorNaN.getValue()
                    : defaulttype::RGBAColor::fromVec4(eval(pointQuadData[i*4+2]));
                Vec4f color3 = isnan(pointQuadData[i*4+3])
                    ? f_colorNaN.getValue()
                    : defaulttype::RGBAColor::fromVec4(eval(pointQuadData[i*4+3]));
                const Quad& q = topology->getQuad(i);

                glNormal3fv(m_normals[q[0]].ptr());
                glMaterialfv(GL_FRONT,GL_DIFFUSE,color0.ptr());
                helper::gl::glVertexNv<3>(x[q[0]].ptr());

                glNormal3fv(m_normals[q[1]].ptr());
                glMaterialfv(GL_FRONT,GL_DIFFUSE,color1.ptr());
                helper::gl::glVertexNv<3>(x[q[1]].ptr());

                glNormal3fv(m_normals[q[2]].ptr());
                glMaterialfv(GL_FRONT,GL_DIFFUSE,color2.ptr());
                helper::gl::glVertexNv<3>(x[q[2]].ptr());

                glNormal3fv(m_normals[q[3]].ptr());
                glMaterialfv(GL_FRONT,GL_DIFFUSE,color3.ptr());
                helper::gl::glVertexNv<3>(x[q[3]].ptr());

            }
            glEnd();
        }
    }

    if ((bDrawCellData || !topology) && bDrawPointData) {
        helper::ColorMap::evaluator<Real> eval = colorMap->getEvaluator(min, max);
        // Just the points
        glPointSize(10);
        glBegin(GL_POINTS);
        for (unsigned int i=0; i<x.size(); ++i)
        {
            Vec4f color = isnan(ptData[i])
                ? f_colorNaN.getValue()
                : defaulttype::RGBAColor::fromVec4(eval(ptData[i]));
            vparams->drawTool()->drawPoint(x[i], color);
        }
        glEnd();

    } else if (bDrawPointData) {
        helper::ColorMap::evaluator<Real> eval = colorMap->getEvaluator(min, max);

        glEnable ( GL_LIGHTING );

        // Triangles
        glBegin(GL_TRIANGLES);
        for (int i=0; i<topology->getNbTriangles(); ++i)
        {
            const Triangle &t = topology->getTriangle(i);
            Vec4f color[3];
            for (int j=0; j<3; j++) {
                color[j] = isnan(ptData[t[j]])
                    ? f_colorNaN.getValue()
                    : defaulttype::RGBAColor::fromVec4(eval(ptData[t[j]]));
            }

            glNormal3fv(m_normals[t[0]].ptr());
            glMaterialfv(GL_FRONT,GL_DIFFUSE,color[0].ptr());
            helper::gl::glVertexNv<3>(x[t[0]].ptr());

            glNormal3fv(m_normals[t[1]].ptr());
            glMaterialfv(GL_FRONT,GL_DIFFUSE,color[1].ptr());
            helper::gl::glVertexNv<3>(x[t[1]].ptr());

            glNormal3fv(m_normals[t[2]].ptr());
            glMaterialfv(GL_FRONT,GL_DIFFUSE,color[2].ptr());
            helper::gl::glVertexNv<3>(x[t[2]].ptr());

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
                : defaulttype::RGBAColor::fromVec4(eval(ptData[q[j]]));
            }

            glNormal3fv(m_normals[q[0]].ptr());
            glMaterialfv(GL_FRONT,GL_DIFFUSE,color[0].ptr());
            helper::gl::glVertexNv<3>(x[q[0]].ptr());

            glNormal3fv(m_normals[q[1]].ptr());
            glMaterialfv(GL_FRONT,GL_DIFFUSE,color[1].ptr());
            helper::gl::glVertexNv<3>(x[q[1]].ptr());

            glNormal3fv(m_normals[q[2]].ptr());
            glMaterialfv(GL_FRONT,GL_DIFFUSE,color[2].ptr());
            helper::gl::glVertexNv<3>(x[q[2]].ptr());

            glNormal3fv(m_normals[q[3]].ptr());
            glMaterialfv(GL_FRONT,GL_DIFFUSE,color[3].ptr());
            helper::gl::glVertexNv<3>(x[q[3]].ptr());

        }
        glEnd();
    }

    glPopAttrib();

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void DataDisplay::computeNormals()
{
    if( !topology ) return;
    const VecCoord& x = this->read(sofa::core::ConstVecCoordId::position())->getValue();

    m_normals.resize(x.size(),Vec3f(0,0,0));

    for (int i=0; i<topology->getNbTriangles(); ++i)
    {
        const Triangle &t = topology->getTriangle(i);

        const Coord& v1 = x[t[0]];
        const Coord& v2 = x[t[1]];
        const Coord& v3 = x[t[2]];
        Coord n = cross(v2-v1, v3-v1);

        m_normals[t[0]] += n;
        m_normals[t[1]] += n;
        m_normals[t[2]] += n;
    }

    for (int i=0; i<topology->getNbQuads(); ++i)
    {
        const Quad &q = topology->getQuad(i);

        const Coord & v1 = x[q[0]];
        const Coord & v2 = x[q[1]];
        const Coord & v3 = x[q[2]];
        const Coord & v4 = x[q[3]];
        Coord n1 = cross(v2-v1, v4-v1);
        Coord n2 = cross(v3-v2, v1-v2);
        Coord n3 = cross(v4-v3, v2-v3);
        Coord n4 = cross(v1-v4, v3-v4);

        m_normals[q[0]] += n1;
        m_normals[q[1]] += n2;
        m_normals[q[2]] += n3;
        m_normals[q[3]] += n4;
    }

    // normalization
    for (size_t i=0; i<x.size(); ++i)
        m_normals[i].normalize();
}

} // namespace visualmodel

} // namespace component

} // namespace sofa
