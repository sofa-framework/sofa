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
#ifndef SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETGEOMETRYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETGEOMETRYALGORITHMS_INL

#include <SofaBaseTopology/HexahedronSetGeometryAlgorithms.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseTopology/CommonAlgorithms.h>

namespace sofa
{

namespace component
{

namespace topology
{

template< class DataTypes>
void HexahedronSetGeometryAlgorithms< DataTypes >::computeHexahedronAABB(const HexaID h, Coord& minCoord, Coord& maxCoord) const
{
    const Hexahedron &t = this->m_topology->getHexahedron(h);
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());

    for(unsigned int i=0; i<3; ++i)
    {
        minCoord[i] = std::min( std::min(std::min(p[t[0]][i], p[t[1]][i]), std::min(p[t[2]][i], p[t[3]][i])),
                std::min(std::min(p[t[4]][i], p[t[5]][i]), std::min(p[t[6]][i], p[t[7]][i])));

        maxCoord[i] = std::max( std::max(std::max(p[t[0]][i], p[t[1]][i]), std::max(p[t[2]][i], p[t[3]][i])),
                std::max(std::max(p[t[4]][i], p[t[5]][i]), std::max(p[t[6]][i], p[t[7]][i])));
    }
}

template< class DataTypes>
void HexahedronSetGeometryAlgorithms< DataTypes >::computeHexahedronRestAABB(const HexaID h, Coord& minCoord, Coord& maxCoord) const
{
    const Hexahedron &t = this->m_topology->getHexahedron(h);
    const typename DataTypes::VecCoord& p = (this->object->read(core::ConstVecCoordId::restPosition())->getValue());

    for(unsigned int i=0; i<3; ++i)
    {
        minCoord[i] = std::min( std::min(std::min(p[t[0]][i], p[t[1]][i]), std::min(p[t[2]][i], p[t[3]][i])),
                std::min(std::min(p[t[4]][i], p[t[5]][i]), std::min(p[t[6]][i], p[t[7]][i])));

        maxCoord[i] = std::max( std::max(std::max(p[t[0]][i], p[t[1]][i]), std::max(p[t[2]][i], p[t[3]][i])),
                std::max(std::max(p[t[4]][i], p[t[5]][i]), std::max(p[t[6]][i], p[t[7]][i])));
    }
}

template<class DataTypes>
typename DataTypes::Coord HexahedronSetGeometryAlgorithms<DataTypes>::computeHexahedronCenter(const HexaID h) const
{
    const Hexahedron &t = this->m_topology->getHexahedron(h);
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());

    return (p[t[0]] + p[t[1]] + p[t[2]] + p[t[3]] + p[t[4]] + p[t[5]] + p[t[6]] + p[t[7]]) * (Real) 0.125;
}

template<class DataTypes>
typename DataTypes::Coord HexahedronSetGeometryAlgorithms<DataTypes>::computeHexahedronRestCenter(const HexaID h) const
{
    const Hexahedron &t = this->m_topology->getHexahedron(h);
    const typename DataTypes::VecCoord& p = (this->object->read(core::ConstVecCoordId::restPosition())->getValue());

    return (p[t[0]] + p[t[1]] + p[t[2]] + p[t[3]] + p[t[4]] + p[t[5]] + p[t[6]] + p[t[7]]) * (Real) 0.125;
}

template< class DataTypes>
void HexahedronSetGeometryAlgorithms< DataTypes >::getHexahedronVertexCoordinates(const HexaID h, Coord pnt[8]) const
{
    const Hexahedron &t = this->m_topology->getHexahedron(h);
    const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());

    for(unsigned int i=0; i<8; ++i)
    {
        pnt[i] = p[t[i]];
    }
}

template< class DataTypes>
void HexahedronSetGeometryAlgorithms< DataTypes >::getRestHexahedronVertexCoordinates(const HexaID h, Coord pnt[8]) const
{
    const Hexahedron &t = this->m_topology->getHexahedron(h);
    const typename DataTypes::VecCoord& p = (this->object->read(core::ConstVecCoordId::restPosition())->getValue());

    for(unsigned int i=0; i<8; ++i)
    {
        pnt[i] = p[t[i]];
    }
}

template<class DataTypes>
typename DataTypes::Coord HexahedronSetGeometryAlgorithms<DataTypes>::getRestPointPositionInHexahedron(const HexaID h,
        const Real baryC[3]) const
{
    Coord	p[8];
    getRestHexahedronVertexCoordinates(h, p);

    const Real &fx = baryC[0];
    const Real &fy = baryC[1];
    const Real &fz = baryC[2];

    const Coord pos = p[0] * ((1-fx) * (1-fy) * (1-fz))
            + p[1] * ((  fx) * (1-fy) * (1-fz))
            + p[3] * ((1-fx) * (  fy) * (1-fz))
            + p[2] * ((  fx) * (  fy) * (1-fz))
            + p[4] * ((1-fx) * (1-fy) * (  fz))
            + p[5] * ((  fx) * (1-fy) * (  fz))
            + p[7] * ((1-fx) * (  fy) * (  fz))
            + p[6] * ((  fx) * (  fy) * (  fz));

    return pos;
}

template<class DataTypes>
typename DataTypes::Coord HexahedronSetGeometryAlgorithms<DataTypes>::getRestPointPositionInHexahedron(const HexaID h,
        const sofa::defaulttype::Vector3& baryC) const
{
    Coord	p[8];
    getRestHexahedronVertexCoordinates(h, p);

    const Real fx = (Real) baryC[0];
    const Real fy = (Real) baryC[1];
    const Real fz = (Real) baryC[2];

    const Coord pos = p[0] * ((1-fx) * (1-fy) * (1-fz))
            + p[1] * ((  fx) * (1-fy) * (1-fz))
            + p[3] * ((1-fx) * (  fy) * (1-fz))
            + p[2] * ((  fx) * (  fy) * (1-fz))
            + p[4] * ((1-fx) * (1-fy) * (  fz))
            + p[5] * ((  fx) * (1-fy) * (  fz))
            + p[7] * ((1-fx) * (  fy) * (  fz))
            + p[6] * ((  fx) * (  fy) * (  fz));

    return pos;
}

template<class DataTypes>
typename DataTypes::Coord HexahedronSetGeometryAlgorithms<DataTypes>::getPointPositionInHexahedron(const HexaID h,
        const Real baryC[3]) const
{
    Coord	p[8];
    getHexahedronVertexCoordinates(h, p);

    const Real &fx = baryC[0];
    const Real &fy = baryC[1];
    const Real &fz = baryC[2];

    const Coord pos = p[0] * ((1-fx) * (1-fy) * (1-fz))
            + p[1] * ((  fx) * (1-fy) * (1-fz))
            + p[3] * ((1-fx) * (  fy) * (1-fz))
            + p[2] * ((  fx) * (  fy) * (1-fz))
            + p[4] * ((1-fx) * (1-fy) * (  fz))
            + p[5] * ((  fx) * (1-fy) * (  fz))
            + p[7] * ((1-fx) * (  fy) * (  fz))
            + p[6] * ((  fx) * (  fy) * (  fz));

    return pos;
}

template<class DataTypes>
typename DataTypes::Coord HexahedronSetGeometryAlgorithms<DataTypes>::getPointPositionInHexahedron(const HexaID h,
        const sofa::defaulttype::Vector3& baryC) const
{
    Coord	p[8];
    getHexahedronVertexCoordinates(h, p);

    const Real fx = (Real) baryC[0];
    const Real fy = (Real) baryC[1];
    const Real fz = (Real) baryC[2];

    const Coord pos = p[0] * ((1-fx) * (1-fy) * (1-fz))
            + p[1] * ((  fx) * (1-fy) * (1-fz))
            + p[3] * ((1-fx) * (  fy) * (1-fz))
            + p[2] * ((  fx) * (  fy) * (1-fz))
            + p[4] * ((1-fx) * (1-fy) * (  fz))
            + p[5] * ((  fx) * (1-fy) * (  fz))
            + p[7] * ((1-fx) * (  fy) * (  fz))
            + p[6] * ((  fx) * (  fy) * (  fz));

    return pos;
}

template<class DataTypes>
sofa::defaulttype::Vector3 HexahedronSetGeometryAlgorithms<DataTypes>::computeHexahedronRestBarycentricCoeficients(const HexaID h,
        const Coord& pos) const
{
    Coord	p[8];
    getRestHexahedronVertexCoordinates(h, p);

    const unsigned int max_spatial_dimensions = std::min((unsigned int)3,(unsigned int)DataTypes::spatial_dimensions);

    sofa::defaulttype::Vector3 origin, p1, p3, p4, pnt;
    for( unsigned int w=0 ; w<max_spatial_dimensions ; ++w )
    {
        origin[w] = p[0][w];
        p1[w] = p[1][w];
        p3[w] = p[3][w];
        p4[w] = p[4][w];
        pnt[w] = pos[w];
    }

    sofa::defaulttype::Mat3x3d		m, mt, base;
    m[0] = p1-origin;
    m[1] = p3-origin;
    m[2] = p4-origin;
    mt.transpose(m);
    base.invert(mt);

    return base * (pnt - origin);
}

template<class DataTypes>
sofa::defaulttype::Vector3 HexahedronSetGeometryAlgorithms<DataTypes>::computeHexahedronBarycentricCoeficients(const HexaID h,
        const Coord& pos) const
{
    // Warning: this is only correct if the hexahedron is not deformed
    // as only 3 perpendicular edges are considered as a base
    // other edges are assumed to be parallel to the respective base edge (and have the same length)

    Coord	p[8];
    getHexahedronVertexCoordinates(h, p);

    const unsigned int max_spatial_dimensions = std::min((unsigned int)3,(unsigned int)DataTypes::spatial_dimensions);

    sofa::defaulttype::Vector3 origin, p1, p3, p4, pnt;
    for( unsigned int w=0 ; w<max_spatial_dimensions ; ++w )
    {
        origin[w] = p[0][w];
        p1[w] = p[1][w];
        p3[w] = p[3][w];
        p4[w] = p[4][w];
        pnt[w] = pos[w];
    }

    sofa::defaulttype::Mat3x3d		m, mt, base;
    m[0] = p1-origin;
    m[1] = p3-origin;
    m[2] = p4-origin;
    mt.transpose(m);
    base.invert(mt);

    return base * (pnt - origin);
}

template< class DataTypes>
typename DataTypes::Real HexahedronSetGeometryAlgorithms< DataTypes >::computeElementDistanceMeasure(const HexaID h, const Coord pos) const
{
    typedef typename DataTypes::Real Real;

    const sofa::defaulttype::Vector3 v = computeHexahedronBarycentricCoeficients(h, pos);

    Real d = (Real) std::max(std::max(-v[0], -v[1]), std::max(std::max(-v[2], v[0]-1), std::max(v[1]-1, v[2]-1)));

    if(d>0)
        d = (pos - computeHexahedronCenter(h)).norm2();

    return d;
}

template< class DataTypes>
typename DataTypes::Real HexahedronSetGeometryAlgorithms< DataTypes >::computeElementRestDistanceMeasure(const HexaID h, const Coord pos) const
{
    typedef typename DataTypes::Real Real;

    const sofa::defaulttype::Vector3 v = computeHexahedronRestBarycentricCoeficients(h, pos);

    Real d = (Real) std::max(std::max(-v[0], -v[1]), std::max(std::max(-v[2], v[0]-1), std::max(v[1]-1, v[2]-1)));

    if(d>0)
        d = (pos - computeHexahedronRestCenter(h)).norm2();

    return d;
}

template< class DataTypes>
int HexahedronSetGeometryAlgorithms< DataTypes >::findNearestElement(const Coord& pos, sofa::defaulttype::Vector3& baryC, Real& distance) const
{
    int index=-1;
    distance = 1e10;

    for(int c=0; c<this->m_topology->getNbHexahedra(); ++c)
    {
        const Real d = computeElementDistanceMeasure(c, pos);

        if(d<distance)
        {
            distance = d;
            index = c;
        }
    }

    if(index != -1)
        baryC = computeHexahedronBarycentricCoeficients(index, pos);

    return index;
}

template< class DataTypes>
void HexahedronSetGeometryAlgorithms< DataTypes >::findNearestElements(const VecCoord& pos,
        helper::vector<int>& elem,
        helper::vector<defaulttype::Vector3>& baryC,
        helper::vector<Real>& dist) const
{
    for(unsigned int i=0; i<pos.size(); ++i)
    {
        elem[i] = findNearestElement(pos[i], baryC[i], dist[i]);
    }
}

template< class DataTypes>
int HexahedronSetGeometryAlgorithms< DataTypes >::findNearestElementInRestPos(const Coord& pos, sofa::defaulttype::Vector3& baryC, Real& distance) const
{
    int index=-1;
    distance = 1e10;

    for(int c=0; c<this->m_topology->getNbHexahedra(); ++c)
    {
        const Real d = computeElementRestDistanceMeasure(c, pos);

        if(d<distance)
        {
            distance = d;
            index = c;
        }
    }

    if(index != -1)
        baryC = computeHexahedronRestBarycentricCoeficients(index, pos);

    return index;
}

template< class DataTypes>
void HexahedronSetGeometryAlgorithms< DataTypes >::findNearestElementsInRestPos( const VecCoord& pos, helper::vector<int>& elem, helper::vector<defaulttype::Vector3>& baryC, helper::vector<Real>& dist) const
{
    for(unsigned int i=0; i<pos.size(); ++i)
    {
        elem[i] = findNearestElementInRestPos(pos[i], baryC[i], dist[i]);
    }
}

template< class DataTypes>
typename DataTypes::Real HexahedronSetGeometryAlgorithms< DataTypes >::computeHexahedronVolume( const HexaID /*h*/) const
{
    //const Hexahedron &t = this->m_topology->getHexahedron(h);
    //const VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());
    Real volume=(Real)(0.0); /// @todo : implementation of computeHexahedronVolume
    return volume;
}

template< class DataTypes>
typename DataTypes::Real HexahedronSetGeometryAlgorithms< DataTypes >::computeRestHexahedronVolume( const HexaID /*h*/) const
{
    //const Hexahedron &t = this->m_topology->getHexahedron(h);
    //const VecCoord& p = (this->object->read(core::ConstVecCoordId::restPosition())->getValue());
    Real volume=(Real)(0.0); /// @todo : implementation of computeRestHexahedronVolume
    return volume;
}

template<class DataTypes>
void HexahedronSetGeometryAlgorithms<DataTypes>::computeHexahedronVolume( BasicArrayInterface<Real> &ai) const
{
    //const sofa::helper::vector<Hexahedron> &ta=this->m_topology->getHexahedra();
    //const typename DataTypes::VecCoord& p =(this->object->read(core::ConstVecCoordId::position())->getValue());
    for(int i=0; i<this->m_topology->getNbHexahedra(); ++i)
    {
        //const Hexahedron &t=this->m_topology->getHexahedron(i); //ta[i];
        ai[i]=(Real)(0.0); /// @todo : implementation of computeHexahedronVolume
    }
}

/// Write the current mesh into a msh file
template <typename DataTypes>
void HexahedronSetGeometryAlgorithms<DataTypes>::writeMSHfile(const char *filename) const
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

    const sofa::helper::vector<Hexahedron> hea = this->m_topology->getHexahedra();

    myfile << hea.size() <<"\n";

    for(unsigned int i=0; i<hea.size(); ++i)
    {
        myfile << i+1 << " 5 1 1 8 " << hea[i][4]+1 << " " << hea[i][5]+1 << " "
                << hea[i][1]+1 << " " << hea[i][0]+1 << " "
                << hea[i][7]+1 << " " << hea[i][6]+1 << " "
                << hea[i][2]+1 << " " << hea[i][3]+1 << "\n";
    }

    myfile << "$ENDELM\n";

    myfile.close();
}

template<class DataTypes>
void HexahedronSetGeometryAlgorithms<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    QuadSetGeometryAlgorithms<DataTypes>::draw(vparams);

    // Draw Hexa indices
    if (showHexaIndices.getValue())
    {
        sofa::defaulttype::Mat<4,4, GLfloat> modelviewM;

        const VecCoord& coords =(this->object->read(core::ConstVecCoordId::position())->getValue());
        const sofa::defaulttype::Vec3f& color = _drawColor.getValue();
        glColor3f(color[0], color[1], color[2]);
        glDisable(GL_LIGHTING);
        float scale = this->getIndicesScale();

        //for hexa:
        scale = scale/2;

        const sofa::helper::vector<Hexahedron> &hexaArray = this->m_topology->getHexahedra();

        for (unsigned int i =0; i<hexaArray.size(); i++)
        {

            Hexahedron the_hexa = hexaArray[i];
            sofa::defaulttype::Vec3f center;

            for (unsigned int j = 0; j<8; j++)
            {
                sofa::defaulttype::Vec3f vertex; vertex = DataTypes::getCPos(coords[ the_hexa[j] ]);
                center += vertex;
            }

            center = center/8;

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


    //Draw hexahedra
    if (_draw.getValue())
    {
        const sofa::helper::vector<Hexahedron> &hexaArray = this->m_topology->getHexahedra();

        if (!hexaArray.empty())
        {
            glDisable(GL_LIGHTING);
            const sofa::defaulttype::Vec3f& color = _drawColor.getValue();
            glColor3f(color[0], color[1], color[2]);
            glBegin(GL_LINES);
            const VecCoord& coords =(this->object->read(core::ConstVecCoordId::position())->getValue());

            for (unsigned int i = 0; i<hexaArray.size(); i++)
            {
                const Hexahedron& H = hexaArray[i];
                sofa::helper::vector <sofa::defaulttype::Vec3f> hexaCoord;

                for (unsigned int j = 0; j<8; j++)
                {
                    sofa::defaulttype::Vec3f p; p = DataTypes::getCPos(coords[H[j]]);
                    hexaCoord.push_back(p);
                }

                for (unsigned int j = 0; j<4; j++)
                {
                    glVertex3f(hexaCoord[j][0], hexaCoord[j][1], hexaCoord[j][2]);
                    glVertex3f(hexaCoord[(j+1)%4][0], hexaCoord[(j+1)%4][1], hexaCoord[(j+1)%4][2]);

                    glVertex3f(hexaCoord[j+4][0], hexaCoord[j+4][1], hexaCoord[j+4][2]);
                    glVertex3f(hexaCoord[(j+1)%4 +4][0], hexaCoord[(j+1)%4 +4][1], hexaCoord[(j+1)%4 +4][2]);

                    glVertex3f(hexaCoord[j][0], hexaCoord[j][1], hexaCoord[j][2]);
                    glVertex3f(hexaCoord[j+4][0], hexaCoord[j+4][1], hexaCoord[j+4][2]);
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

#endif // SOFA_COMPONENTS_HexahedronSetTOPOLOGY_INL
