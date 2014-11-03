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
#ifndef SOFA_COMPONENT_TOPOLOGY_QUADSETGEOMETRYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_QUADSETGEOMETRYALGORITHMS_INL

#include <SofaBaseTopology/QuadSetGeometryAlgorithms.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseTopology/CommonAlgorithms.h>

namespace sofa
{

namespace component
{

namespace topology
{

template< class DataTypes>
void QuadSetGeometryAlgorithms< DataTypes >::computeQuadAABB(const QuadID i, Coord& minCoord, Coord& maxCoord) const
{
    const Quad &t = this->m_topology->getQuad(i);
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());

    for(unsigned int i=0; i<3; ++i)
    {
        minCoord[i] = std::min(std::min(p[t[0]][i], p[t[3]][i]), std::min(p[t[1]][i], p[t[2]][i]));
        maxCoord[i] = std::max(std::max(p[t[0]][i], p[t[3]][i]), std::max(p[t[1]][i], p[t[2]][i]));
    }
}

template<class DataTypes>
typename DataTypes::Coord QuadSetGeometryAlgorithms<DataTypes>::computeQuadCenter(const QuadID i) const
{
    const Quad &t = this->m_topology->getQuad(i);
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());

    return (p[t[0]] + p[t[1]] + p[t[2]] + p[t[3]]) * (Real) 0.25;
}

template< class DataTypes>
void QuadSetGeometryAlgorithms< DataTypes >::getQuadVertexCoordinates(const QuadID i, Coord pnt[4]) const
{
    const Quad &t = this->m_topology->getQuad(i);
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());

    for(unsigned int i=0; i<4; ++i)
    {
        pnt[i] = p[t[i]];
    }
}

template< class DataTypes>
void QuadSetGeometryAlgorithms< DataTypes >::getRestQuadVertexCoordinates(const QuadID i, Coord pnt[4]) const
{
    const Quad &t = this->m_topology->getQuad(i);
    const typename DataTypes::VecCoord& p = (this->object->read(core::ConstVecCoordId::restPosition())->getValue());

    for(unsigned int i=0; i<4; ++i)
    {
        pnt[i] = p[t[i]];
    }
}

template< class DataTypes>
typename DataTypes::Real QuadSetGeometryAlgorithms< DataTypes >::computeQuadArea( const QuadID i) const
{
    const Quad &t = this->m_topology->getQuad(i);
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());
    Real area = (Real)((areaProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]])
            + areaProduct(p[t[3]]-p[t[2]],p[t[0]]-p[t[2]])) * (Real) 0.5);
    return area;
}

template< class DataTypes>
typename DataTypes::Real QuadSetGeometryAlgorithms< DataTypes >::computeRestQuadArea( const QuadID i) const
{
    const Quad &t = this->m_topology->getQuad(i);
    const typename DataTypes::VecCoord& p = (this->object->read(core::ConstVecCoordId::restPosition())->getValue());
    Real area = (Real)((areaProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]])
            + areaProduct(p[t[3]]-p[t[2]],p[t[0]]-p[t[2]])) * (Real) 0.5);
    return area;
}

template<class DataTypes>
void QuadSetGeometryAlgorithms<DataTypes>::computeQuadArea( BasicArrayInterface<Real> &ai) const
{
    //const sofa::helper::vector<Quad> &ta=this->m_topology->getQuads();
    int nb_quads = this->m_topology->getNbQuads();
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());

    for(int i=0; i<nb_quads; ++i)
    {
        // ta.size()
        const Quad &t = this->m_topology->getQuad(i);  //ta[i];
        ai[i] = (Real)((areaProduct(p[t[1]]-p[t[0]],p[t[2]]-p[t[0]])
                + areaProduct(p[t[3]]-p[t[2]],p[t[0]]-p[t[2]])) * (Real) 0.5);
    }
}

// Computes the normal vector of a quad indexed by ind_q (not normed)
template<class DataTypes>
sofa::defaulttype::Vec<3,double> QuadSetGeometryAlgorithms< DataTypes >::computeQuadNormal(const QuadID ind_q) const
{
    // HYP :  The quad indexed by ind_q is planar

    const Quad &q = this->m_topology->getQuad(ind_q);
    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::ConstVecCoordId::position())->getValue());

    const typename DataTypes::Coord& c0=vect_c[q[0]];
    const typename DataTypes::Coord& c1=vect_c[q[1]];
    const typename DataTypes::Coord& c2=vect_c[q[2]];
    //const typename DataTypes::Coord& c3=vect_c[q[3]];

    sofa::defaulttype::Vec<3,Real> p0;
    p0[0] = (Real) (c0[0]); p0[1] = (Real) (c0[1]); p0[2] = (Real) (c0[2]);
    sofa::defaulttype::Vec<3,Real> p1;
    p1[0] = (Real) (c1[0]); p1[1] = (Real) (c1[1]); p1[2] = (Real) (c1[2]);
    sofa::defaulttype::Vec<3,Real> p2;
    p2[0] = (Real) (c2[0]); p2[1] = (Real) (c2[1]); p2[2] = (Real) (c2[2]);
    //Vec<3,Real> p3;
    //p3[0] = (Real) (c3[0]); p3[1] = (Real) (c3[1]); p3[2] = (Real) (c3[2]);

    sofa::defaulttype::Vec<3,Real> normal_q=(p1-p0).cross( p2-p0);

    return ((sofa::defaulttype::Vec<3,double>) normal_q);
}


// Test if a quad indexed by ind_quad (and incident to the vertex indexed by ind_p) is included or not in the plane defined by (ind_p, plane_vect)
template<class DataTypes>
bool QuadSetGeometryAlgorithms< DataTypes >::isQuadInPlane(const QuadID ind_q,
        const unsigned int ind_p,
        const sofa::defaulttype::Vec<3,Real>&plane_vect) const
{
    const Quad &q = this->m_topology->getQuad(ind_q);

    // HYP : ind_p==q[0] or ind_q==t[1] or ind_q==t[2] or ind_q==q[3]

    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::ConstVecCoordId::position())->getValue());

    unsigned int ind_1;
    unsigned int ind_2;
    unsigned int ind_3;

    if(ind_p==q[0])
    {
        ind_1=q[1];
        ind_2=q[2];
        ind_3=q[3];
    }
    else if(ind_p==q[1])
    {
        ind_1=q[2];
        ind_2=q[3];
        ind_3=q[0];
    }
    else if(ind_p==q[2])
    {
        ind_1=q[3];
        ind_2=q[0];
        ind_3=q[1];
    }
    else
    {
        // ind_p==q[3]
        ind_1=q[0];
        ind_2=q[1];
        ind_3=q[2];
    }

    const typename DataTypes::Coord& c0 = vect_c[ind_p];
    const typename DataTypes::Coord& c1 = vect_c[ind_1];
    const typename DataTypes::Coord& c2 = vect_c[ind_2];
    const typename DataTypes::Coord& c3 = vect_c[ind_3];

    sofa::defaulttype::Vec<3,Real> p0;
    p0[0] = (Real) (c0[0]); p0[1] = (Real) (c0[1]); p0[2] = (Real) (c0[2]);
    sofa::defaulttype::Vec<3,Real> p1;
    p1[0] = (Real) (c1[0]); p1[1] = (Real) (c1[1]); p1[2] = (Real) (c1[2]);
    sofa::defaulttype::Vec<3,Real> p2;
    p2[0] = (Real) (c2[0]); p2[1] = (Real) (c2[1]); p2[2] = (Real) (c2[2]);
    sofa::defaulttype::Vec<3,Real> p3;
    p3[0] = (Real) (c3[0]); p3[1] = (Real) (c3[1]); p3[2] = (Real) (c3[2]);

    return((p1-p0)*( plane_vect)>=0.0 && (p2-p0)*( plane_vect)>=0.0 && (p3-p0)*( plane_vect)>=0.0);
}

template<class DataTypes>
bool QuadSetGeometryAlgorithms< DataTypes >::isPointInQuad(const QuadID ind_q, const sofa::defaulttype::Vec<3,Real>& p) const
{
    const double ZERO = 1e-6;
    const Quad &q = this->m_topology->getQuad(ind_q);
    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::ConstVecCoordId::position())->getValue());

    sofa::defaulttype::Vec<3,Real> ptest = p;
    sofa::defaulttype::Vec<3,Real> p0(vect_c[q[0]][0], vect_c[q[0]][1], vect_c[q[0]][2]);
    sofa::defaulttype::Vec<3,Real> p1(vect_c[q[1]][0], vect_c[q[1]][1], vect_c[q[1]][2]);
    sofa::defaulttype::Vec<3,Real> p2(vect_c[q[2]][0], vect_c[q[2]][1], vect_c[q[2]][2]);
    sofa::defaulttype::Vec<3,Real> p3(vect_c[q[3]][0], vect_c[q[3]][1], vect_c[q[3]][2]);

    sofa::defaulttype::Vec<3,Real> v_normal = (p2-p0).cross(p1-p0);
    Real norm_v_normal = v_normal*(v_normal);
    if(norm_v_normal > ZERO)
    {
        if(fabs((ptest-p0)*(v_normal)) < ZERO) // p is in the plane defined by the triangle (p0,p1,p2)
        {

            sofa::defaulttype::Vec<3,Real> n_01 = (p1-p0).cross(v_normal);
            sofa::defaulttype::Vec<3,Real> n_12 = (p2-p1).cross(v_normal);
            sofa::defaulttype::Vec<3,Real> n_20 = (p0-p2).cross(v_normal);

            if(((ptest-p0)*(n_01) > -ZERO) && ((ptest-p1)*(n_12) > -ZERO) && ((ptest-p2)*(n_20) > -ZERO))
                return true;
        }
    }

    v_normal = (p3-p0).cross(p2-p0);
    norm_v_normal = v_normal*(v_normal);
    if(norm_v_normal > ZERO)
    {
        if(fabs((ptest-p0)*(v_normal)) < ZERO) // p is in the plane defined by the triangle (p0,p3,p2)
        {

            sofa::defaulttype::Vec<3,Real> n_01 = (p2-p0).cross(v_normal);
            sofa::defaulttype::Vec<3,Real> n_12 = (p3-p2).cross(v_normal);
            sofa::defaulttype::Vec<3,Real> n_20 = (p0-p3).cross(v_normal);

            if(((ptest-p0)*(n_01) > -ZERO) && ((ptest-p2)*(n_12) > -ZERO) && ((ptest-p3)*(n_20) > -ZERO))
                return true;
        }

    }
    return false;
}

/// Write the current mesh into a msh file
template <typename DataTypes>
void QuadSetGeometryAlgorithms<DataTypes>::writeMSHfile(const char *filename) const
{
    std::ofstream myfile;
    myfile.open (filename);

    const typename DataTypes::VecCoord& vect_c =(this->object->read(core::ConstVecCoordId::position())->getValue());

    const unsigned int numVertices = vect_c.size();

    myfile << "$NOD\n";
    myfile << numVertices <<"\n";

    for(unsigned int i=0; i<numVertices; ++i)
    {
        double x = (double) vect_c[i][0];
        double y = (double) vect_c[i][1];
        double z = (double) vect_c[i][2];

        myfile << i+1 << " " << x << " " << y << " " << z <<"\n";
    }

    myfile << "$ENDNOD\n";
    myfile << "$ELM\n";

    const sofa::helper::vector<Quad>& qa = this->m_topology->getQuads();

    myfile << qa.size() <<"\n";

    for(unsigned int i=0; i<qa.size(); ++i)
    {
        myfile << i+1 << " 3 1 1 4 " << qa[i][0]+1 << " " << qa[i][1]+1 << " " << qa[i][2]+1 << " " << qa[i][3]+1 << "\n";
    }

    myfile << "$ENDELM\n";

    myfile.close();
}

template<class Coord>
bool is_point_in_quad(const Coord& p,
        const Coord& a, const Coord& b,
        const Coord& c, const Coord& d)
{
    const double ZERO = 1e-6;

    Coord ptest = p;
    Coord p0 = a;
    Coord p1 = b;
    Coord p2 = c;
    Coord p3 = d;

    Coord v_normal = (p2-p0).cross(p1-p0);

    double norm_v_normal = v_normal*(v_normal);
    if(norm_v_normal > ZERO)
    {
        if(fabs((ptest-p0)*(v_normal)) < ZERO) // p is in the plane defined by the triangle (p0,p1,p2)
        {

            Coord n_01 = (p1-p0).cross(v_normal);
            Coord n_12 = (p2-p1).cross(v_normal);
            Coord n_20 = (p0-p2).cross(v_normal);

            if(((ptest-p0)*(n_01) > -ZERO) && ((ptest-p1)*(n_12) > -ZERO) && ((ptest-p2)*(n_20) > -ZERO))
                return true;
        }
    }

    v_normal = (p3-p0).cross(p2-p0);
    norm_v_normal = v_normal*(v_normal);
    if(norm_v_normal > ZERO)
    {
        if(fabs((ptest-p0)*(v_normal)) < ZERO) // p is in the plane defined by the triangle (p0,p2,p3)
        {

            Coord n_01 = (p2-p0).cross(v_normal);
            Coord n_12 = (p3-p2).cross(v_normal);
            Coord n_20 = (p0-p3).cross(v_normal);

            if(((ptest-p0)*(n_01) > -ZERO) && ((ptest-p2)*(n_12) > -ZERO) && ((ptest-p3)*(n_20) > -ZERO))
                return true;
        }

    }

    return false;
}


template<class DataTypes>
void QuadSetGeometryAlgorithms<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    EdgeSetGeometryAlgorithms<DataTypes>::draw(vparams);

    // Draw Quads indices
    if (showQuadIndices.getValue())
    {
        sofa::defaulttype::Mat<4,4, GLfloat> modelviewM;

        const VecCoord& coords =(this->object->read(core::ConstVecCoordId::position())->getValue());
        const sofa::defaulttype::Vec3f& color = _drawColor.getValue();
        glColor3f(color[0]-0.2f, color[1]-0.2f, color[2]-0.2f);
        glDisable(GL_LIGHTING);
        float scale = this->getIndicesScale();

        //for quads:
        scale = scale/2;

        const sofa::helper::vector<Quad>& quadArray = this->m_topology->getQuads();

        for (unsigned int i =0; i<quadArray.size(); i++)
        {

            Quad the_quad = quadArray[i];
            Coord vertex1 = coords[ the_quad[0] ];
            Coord vertex2 = coords[ the_quad[1] ];
            Coord vertex3 = coords[ the_quad[2] ];
            Coord vertex4 = coords[ the_quad[3] ];
            sofa::defaulttype::Vec3f center; center = (DataTypes::getCPos(vertex1)+DataTypes::getCPos(vertex2)+DataTypes::getCPos(vertex3)+DataTypes::getCPos(vertex4))/4;

            std::ostringstream oss;
            oss << i;
            std::string tmp = oss.str();
            const char* s = tmp.c_str();
            glPushMatrix();

            glTranslatef(center[0], center[1], center[2]);
            glScalef(scale,scale,scale);

            // Makes text always face the viewer by removing the scene rotation
            // get the current modelview matrix
            glGetFloatv(GL_MODELVIEW_MATRIX , modelviewM.ptr() );
            modelviewM.transpose();

            sofa::defaulttype::Vec3f temp = modelviewM.transform(center);

            //glLoadMatrixf(modelview);
            glLoadIdentity();

            glTranslatef(temp[0], temp[1], temp[2]);
            glScalef(scale,scale,scale);

            while(*s)
            {
                glutStrokeCharacter(GLUT_STROKE_ROMAN, *s);
                s++;
            }

            glPopMatrix();

        }
    }


    // Draw Quads
    if (_draw.getValue())
    {
        const sofa::helper::vector<Quad>& quadArray = this->m_topology->getQuads();

        if (!quadArray.empty()) // Draw Quad surfaces
        {
            const VecCoord& coords =(this->object->read(core::ConstVecCoordId::position())->getValue());

            glDisable(GL_LIGHTING);
            const sofa::defaulttype::Vec3f& color = _drawColor.getValue();
            glColor3f(color[0], color[1], color[2]);
            glBegin(GL_QUADS);
            for (unsigned int i = 0; i<quadArray.size(); i++)
            {
                const Quad& q = quadArray[i];

                for (unsigned int j = 0; j<4; j++)
                {
                    sofa::defaulttype::Vec3f coordP; coordP = DataTypes::getCPos(coords[q[j]]);
                    glVertex3f(coordP[0], coordP[1], coordP[2]);
                }
            }
            glEnd();

            glColor3f(color[0]-0.2f, color[1]-0.2f, color[2]-0.2f);
            glBegin(GL_LINES);
            const sofa::helper::vector<Edge> &edgeArray = this->m_topology->getEdges();

            if (!edgeArray.empty()) //Draw quad edges for better display
            {
                for (unsigned int i = 0; i<edgeArray.size(); i++)
                {
                    const Edge& e = edgeArray[i];
                    sofa::defaulttype::Vec3f coordP1; coordP1 = DataTypes::getCPos(coords[e[0]]);
                    sofa::defaulttype::Vec3f coordP2; coordP2 = DataTypes::getCPos(coords[e[1]]);
                    glVertex3f(coordP1[0], coordP1[1], coordP1[2]);
                    glVertex3f(coordP2[0], coordP2[1], coordP2[2]);
                }
            }
            else
            {
                for (unsigned int i = 0; i<quadArray.size(); i++)
                {
                    const Quad& q = quadArray[i];
                    sofa::helper::vector <sofa::defaulttype::Vec3f> quadCoord;

                    for (unsigned int j = 0; j<4; j++)
                    {
                        sofa::defaulttype::Vec3f p; p = DataTypes::getCPos(coords[q[j]]);
                        quadCoord.push_back(p);
                    }

                    for (unsigned int j = 0; j<4; j++)
                    {
                        glVertex3f(quadCoord[j][0], quadCoord[j][1], quadCoord[j][2]);
                        glVertex3f(quadCoord[(j+1)%4][0], quadCoord[(j+1)%4][1], quadCoord[(j+1)%4][2]);
                    }
                }
            }
            glEnd();
        }
    }
#endif /* SOFA_NO_OPENGL */
}


} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_QUADSETGEOMETRYALGORITHMS_INL
