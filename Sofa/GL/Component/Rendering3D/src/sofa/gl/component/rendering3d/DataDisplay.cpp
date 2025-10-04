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

#include <cmath>

#include <sofa/defaulttype/VecTypes.h>

#include <sofa/gl/component/rendering3d/DataDisplay.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::gl::component::rendering3d
{

using namespace sofa::type;
using namespace sofa::defaulttype;
using sofa::gl::component::rendering2d::OglColorMap;

void registerDataDisplay(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Color rendering of data associated with a mesh.")
        .add< DataDisplay >());
}

DataDisplay::DataDisplay()
    : f_maximalRange(initData(&f_maximalRange, true, "maximalRange", "Keep the maximal range through all timesteps"))
    , f_pointData(initData(&f_pointData, "pointData", "Data associated with nodes"))
    , f_triangleData(initData(&f_triangleData, "triangleData", "Data associated with triangles"))
    , f_quadData(initData(&f_quadData, "quadData", "Data associated with quads"))
    , f_pointTriangleData(initData(&f_pointTriangleData, "pointTriangleData", "Data associated with nodes per triangle"))
    , f_pointQuadData(initData(&f_pointQuadData, "pointQuadData", "Data associated with nodes per quad"))
    , f_colorNaN(initData(&f_colorNaN, sofa::type::RGBAColor(0.0f,0.0f,0.0f,1.0f), "colorNaN", "Color used for NaN values (default=[0.0,0.0,0.0,1.0])"))
    , d_userRange(initData(&d_userRange, type::Vec2f(1,-1), "userRange", "Clamp to this values (if max>min)"))
    , d_currentMin(initData(&d_currentMin, Real(0.0), "currentMin", "Current min range"))
    , d_currentMax(initData(&d_currentMax, Real(0.0), "currentMax", "Current max range"))
    , d_shininess(initData(&d_shininess, -1.f, "shininess", "Shininess for rendering point-based data [0,128].  <0 means no specularity"))
    , d_transparency(initData(&d_transparency, Real(1.0), "transparency", "transparency draw objects with transparency, the value varies between 0. and 1. "
                                                                          "Where 1. means no transparency and 0 full transparency"))
    , l_colorMap(initLink("colorMap", "link to the color map"))
    , m_topology(nullptr)
    , l_topology(initLink("topology", "link to the topology container"))
    , m_oldMin(std::numeric_limits<Real>::max())
    , m_oldMax(std::numeric_limits<Real>::lowest())
{
    this->addAlias(&f_triangleData,"cellData"); // backward compatibility
    d_currentMin.setReadOnly(true);
    d_currentMax.setReadOnly(true);
}

void DataDisplay::init()
{
    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    m_topology = l_topology.get();
    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";
    
    if (!m_topology)
        msg_info() << "No topology information, drawing just points.";

    if (!l_colorMap)
    {
        l_colorMap.set(getContext()->get<gl::component::rendering2d::OglColorMap>());

        if (!l_colorMap)
        {
            if (const auto colorMap = sofa::core::objectmodel::New<gl::component::rendering2d::OglColorMap>())
            {
                getContext()->addObject(colorMap);
                colorMap->setName( this->getContext()->getNameHelper().resolveName(colorMap->getClassName(), {}));
                colorMap->f_printLog.setValue(this->f_printLog.getValue());
                l_colorMap.set(colorMap);

                msg_warning() << "A " << l_colorMap->getClassName() << " is required by " << this->getClassName()
                    << " but has not been found: a default " << l_colorMap->getClassName()
                    << " is automatically added in the scene for you. To remove this warning, add a " << l_colorMap->getClassName() << " in the scene.";
            }
            else
            {
                msg_fatal() << "A " << l_colorMap->getClassName() << " is required by " << this->getClassName()
                    << " but has not been found: a default " << l_colorMap->getClassName()
                    << " could not be automatically added in the scene. To remove this error, add a " << l_colorMap->getClassName() << " in the scene.";
            }
        }
    }
}


void DataDisplay::doUpdateVisual(const core::visual::VisualParams*)
{
    computeNormals();
}

void DataDisplay::doDrawVisual(const core::visual::VisualParams* vparams)
{
    const VecCoord& x = this->read(sofa::core::vec_id::read_access::position)->getValue();
    const VecPointData &ptData = f_pointData.getValue();
    const VecCellData &triData = f_triangleData.getValue();
    const VecCellData &quadData = f_quadData.getValue();
    const VecPointData &pointTriData = f_pointTriangleData.getValue();
    const VecPointData &pointQuadData = f_pointQuadData.getValue();
    typedef sofa::type::RGBAColor RGBAColor;
    const float& transparency = d_transparency.getValue();

    auto* drawTool = vparams->drawTool();

    drawTool->enableLighting();
    drawTool->enableBlending();

    bool bDrawPointData = false;
    bool bDrawCellData = false;


    // Safety checks
    // TODO: can this go to updateVisual()?
    if (ptData.size() > 0) {
        if (ptData.size() != x.size()) {
            msg_error() << "Size of pointData does not mach number of nodes";
        } else {
            bDrawPointData = true;
        }
    } else if (triData.size() > 0 || quadData.size()>0 ) {
        if (!m_topology ) {
            msg_error() << "Topology is necessary for drawing cell data";
        } else if (triData.size() != m_topology->getNbTriangles()) {
            msg_error() << "Size of triangleData does not match number of triangles";
        } else if (quadData.size() != m_topology->getNbQuads()) {
            msg_error() << "Size of quadData does not match number of quads";
        } else {
            bDrawCellData = true;
        }
    } else if (pointTriData.size()>0 || pointQuadData.size()>0) {
        if (!m_topology ) {
            msg_error() << "Topology is necessary for drawing cell data";
        } else if (pointTriData.size() != m_topology->getNbTriangles()*3) {
            msg_error() << "Size of pointTriData does not match number of triangles";
        } else if (pointQuadData.size() != m_topology->getNbQuads()*4) {
            msg_error() << "Size of pointQuadData does not match number of quads";
        } else {
            bDrawCellData = true;
        }
    }

    // Range for points
    Real min = std::numeric_limits<Real>::max();
    Real max = std::numeric_limits<Real>::lowest();
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


    if (max > m_oldMax) m_oldMax = max;
    if (min < m_oldMin) m_oldMin = min;

    if (f_maximalRange.getValue()) {
        max = m_oldMax;
        min = m_oldMin;
    }
    d_currentMin.setValue(min);
    d_currentMax.setValue(max);

    glPushAttrib ( GL_LIGHTING_BIT );

    static const Vec4f emptyColor = Vec4f();
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, emptyColor.ptr());
    glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, emptyColor.ptr());

    if( d_shininess.getValue()>=0 )
    {
        static const Vec4f specular = Vec4f(.5f,.5f,.5f,1.f);
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular.ptr());
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, d_shininess.getValue());
    }
    else
    {
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, emptyColor.ptr());
    }

    if (bDrawCellData) {

        drawTool->disableLighting();
        auto eval = l_colorMap->getEvaluator(min, max);

        if( !triData.empty() )
        {
            // Triangles
            size_t nbTriangles = m_topology->getNbTriangles();
            for (unsigned int i=0; i<nbTriangles; i++)
            {
                RGBAColor color = std::isnan(triData[i])
                    ? f_colorNaN.getValue()
                    : eval(triData[i]);
                color[3] = transparency;
                const Triangle& t = m_topology->getTriangle(i);
                drawTool->drawTriangle(
                    x[ t[0] ], x[ t[1] ], x[ t[2] ],
                    m_normals[ t[0] ], m_normals[ t[1] ], m_normals[ t[2] ],
                    color, color, color);
            }
        }
        else if( !pointTriData.empty() )
        {
            std::vector<RGBAColor> colors;

            size_t nbTriangles = m_topology->getNbTriangles();
            glBegin(GL_TRIANGLES);
            for (unsigned int i=0; i<nbTriangles; i++)
            {
                RGBAColor color0 = std::isnan(pointTriData[i*3])
                    ? f_colorNaN.getValue()
                    : eval(pointTriData[i*3]);
                color0[3] = transparency;
                RGBAColor color1 = std::isnan(pointTriData[i*3+1])
                        ? f_colorNaN.getValue()
                        : eval(pointTriData[i*3+1]);
                color1[3] = transparency;
                RGBAColor color2 = std::isnan(pointTriData[i*3+2])
                    ? f_colorNaN.getValue()
                    : eval(pointTriData[i*3+2]);
                color2[3] = transparency;
                const Triangle& t = m_topology->getTriangle(i);

                glNormalT(m_normals[t[0]]);
                glMaterialfv(GL_FRONT,GL_DIFFUSE,color0.data());
                sofa::gl::glVertexNv<3>(x[t[0]].ptr());

                glNormalT(m_normals[t[1]]);
                glMaterialfv(GL_FRONT,GL_DIFFUSE,color1.data());
                sofa::gl::glVertexNv<3>(x[t[1]].ptr());

                glNormalT(m_normals[t[2]]);
                glMaterialfv(GL_FRONT,GL_DIFFUSE,color2.data());
                sofa::gl::glVertexNv<3>(x[t[2]].ptr());

            }
            glEnd();
        }

        if( !quadData.empty() )
        {
            drawTool->disableLighting();

            size_t nbQuads = m_topology->getNbQuads();
            for (unsigned int i=0; i<nbQuads; i++)
            {
                RGBAColor color = std::isnan(quadData[i])
                    ? f_colorNaN.getValue()
                    : eval(quadData[i]);
                color[3] = transparency;
                const Quad& t = m_topology->getQuad(i);
                drawTool->drawQuad(
                    x[ t[0] ], x[ t[1] ], x[ t[2] ], x[ t[3] ],
                    m_normals[ t[0] ], m_normals[ t[1] ], m_normals[ t[2] ], m_normals[ t[3] ],
                    color, color, color, color);
            }
        }
        else if( !pointQuadData.empty() )
        {
            drawTool->enableLighting();
            size_t nbQuads = m_topology->getNbQuads();
            glBegin(GL_QUADS);
            for (unsigned int i=0; i<nbQuads; i++)
            {
                RGBAColor color0 = std::isnan(pointQuadData[i*4])
                    ? f_colorNaN.getValue()
                    : eval(pointQuadData[i*4]);
                RGBAColor color1 = std::isnan(pointQuadData[i*4+1])
                        ? f_colorNaN.getValue()
                        : eval(pointQuadData[i*4+1]);
                color1[3] = transparency;
                RGBAColor color2 = std::isnan(pointQuadData[i*4+2])
                    ? f_colorNaN.getValue()
                    : eval(pointQuadData[i*4+2]);
                color2[3] = transparency;
                RGBAColor color3 = std::isnan(pointQuadData[i*4+3])
                    ? f_colorNaN.getValue()
                    : eval(pointQuadData[i*4+3]);
                color1[3] = transparency;
                const Quad& q = m_topology->getQuad(i);

                glNormalT(m_normals[q[0]]);
                glMaterialfv(GL_FRONT,GL_DIFFUSE,color0.data());
                sofa::gl::glVertexNv<3>(x[q[0]].ptr());

                glNormalT(m_normals[q[1]]);
                glMaterialfv(GL_FRONT,GL_DIFFUSE,color1.data());
                sofa::gl::glVertexNv<3>(x[q[1]].ptr());

                glNormalT(m_normals[q[2]]);
                glMaterialfv(GL_FRONT,GL_DIFFUSE,color2.data());
                sofa::gl::glVertexNv<3>(x[q[2]].ptr());

                glNormalT(m_normals[q[3]]);
                glMaterialfv(GL_FRONT,GL_DIFFUSE,color3.data());
                sofa::gl::glVertexNv<3>(x[q[3]].ptr());

            }
            glEnd();
        }
    }

    if ((bDrawCellData || !m_topology) && bDrawPointData) {
        helper::ColorMap::evaluator<Real> eval = l_colorMap->getEvaluator(min, max);
        // Just the points
        glPointSize(10);
        for (unsigned int i=0; i<x.size(); ++i)
        {
            RGBAColor color = std::isnan(ptData[i])
                ? f_colorNaN.getValue()
                : eval(ptData[i]);
            color[3] = transparency;
            drawTool->drawPoint(x[i], color);
        }

    } else if (bDrawPointData) {
        helper::ColorMap::evaluator<Real> eval = l_colorMap->getEvaluator(min, max);


        // Triangles
        glBegin(GL_TRIANGLES);
        for (sofa::core::topology::Topology::TriangleID i=0; i<m_topology->getNbTriangles(); ++i)
        {
            const Triangle &t = m_topology->getTriangle(i);
            RGBAColor color[3];
            for (int j=0; j<3; j++) {
                color[j] = std::isnan(ptData[t[j]])
                        ? f_colorNaN.getValue()
                        : eval(ptData[t[j]]);
                color[j][3] = transparency;
            }
            glNormalT(m_normals[t[0]]);
            glMaterialfv(GL_FRONT,GL_DIFFUSE,color[0].data());
            sofa::gl::glVertexNv<3>(x[t[0]].ptr());

            glNormalT(m_normals[t[1]]);
            glMaterialfv(GL_FRONT,GL_DIFFUSE,color[1].data());
            sofa::gl::glVertexNv<3>(x[t[1]].ptr());

            glNormalT(m_normals[t[2]]);
            glMaterialfv(GL_FRONT,GL_DIFFUSE,color[2].data());
            sofa::gl::glVertexNv<3>(x[t[2]].ptr());

        }
        glEnd();

        // Quads
        glBegin(GL_QUADS);
        for (sofa::core::topology::Topology::QuadID i=0; i<m_topology->getNbQuads(); ++i)
        {
            const Quad &q = m_topology->getQuad(i);
            RGBAColor color[4];
            for (int j=0; j<4; j++)
            {
                color[j] = std::isnan(ptData[q[j]])
                ? f_colorNaN.getValue()
                : eval(ptData[q[j]]);
                color[j][3] = transparency;
            }

            glNormalT(m_normals[q[0]]);
            glMaterialfv(GL_FRONT,GL_DIFFUSE,color[0].data());
            sofa::gl::glVertexNv<3>(x[q[0]].ptr());

            glNormalT(m_normals[q[1]]);
            glMaterialfv(GL_FRONT,GL_DIFFUSE,color[1].data());
            sofa::gl::glVertexNv<3>(x[q[1]].ptr());

            glNormalT(m_normals[q[2]]);
            glMaterialfv(GL_FRONT,GL_DIFFUSE,color[2].data());
            sofa::gl::glVertexNv<3>(x[q[2]].ptr());

            glNormalT(m_normals[q[3]]);
            glMaterialfv(GL_FRONT,GL_DIFFUSE,color[3].data());
            sofa::gl::glVertexNv<3>(x[q[3]].ptr());

        }
        glEnd();
    }

    glPopAttrib();
}

void DataDisplay::computeNormals()
{
    if( !m_topology ) return;
    const VecCoord& x = this->read(sofa::core::vec_id::read_access::position)->getValue();

    m_normals.resize(x.size(),Vec3(0,0,0));

    for (sofa::core::topology::Topology::TriangleID i=0; i<m_topology->getNbTriangles(); ++i)
    {
        const Triangle &t = m_topology->getTriangle(i);
        const Coord n = sofa::geometry::Triangle::normal(x[t[0]], x[t[1]], x[t[2]]);

        m_normals[t[0]] += n;
        m_normals[t[1]] += n;
        m_normals[t[2]] += n;
    }

    for (sofa::core::topology::Topology::QuadID i=0; i<m_topology->getNbQuads(); ++i)
    {
        const Quad &q = m_topology->getQuad(i);

        const Coord & v1 = x[q[0]];
        const Coord & v2 = x[q[1]];
        const Coord & v3 = x[q[2]];
        const Coord & v4 = x[q[3]];
        m_normals[q[0]] += sofa::geometry::Triangle::normal(v1, v2, v4);
        m_normals[q[1]] += sofa::geometry::Triangle::normal(v2, v3, v1);
        m_normals[q[2]] += sofa::geometry::Triangle::normal(v3, v4, v2);
        m_normals[q[3]] += sofa::geometry::Triangle::normal(v4, v1, v3);
    }

    // normalization
    for (auto& n : m_normals)
        n.normalize();
}

} // namespace sofa::gl::component::rendering3d
