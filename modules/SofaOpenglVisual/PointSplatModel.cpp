/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <SofaOpenglVisual/PointSplatModel.h>
#include <sofa/core/visual/VisualParams.h>

#include <SofaBaseTopology/TopologyData.inl>

namespace sofa
{

namespace component
{

namespace visualmodel
{

SOFA_DECL_CLASS(PointSplatModel)

int PointSplatModelClass = core::RegisterObject("A simple visualization for a cloud of points.")
        .add< PointSplatModel >()
        .addAlias("PointSplat")
        ;

using namespace sofa::defaulttype;
using namespace sofa::core::topology;

PointSplatModel::PointSplatModel() //const std::string &name, std::string filename, std::string loader, std::string textureName)
    : radius(initData(&radius, 1.0f, "radius", "Radius of the spheres.")),
      textureSize(initData(&textureSize, 32, "textureSize", "Size of the billboard texture.")),
      alpha(initData(&alpha, 1.0f, "alpha", "Opacity of the billboards. 1.0 is 100% opaque.")),
      color(initData(&color, std::string("white"), "color", "Billboard color.")),
      _topology(NULL),
      _mstate(NULL),
      texture_data(NULL),
      pointData(initData(&pointData, "pointData", "scalar field modulating point colors"))
{
}

PointSplatModel::~PointSplatModel()
{
    if(texture_data != NULL)
        delete [] texture_data;
}

void PointSplatModel::init()
{
    getContext()->get(_topology);
    if(_topology)
        _mstate = _topology->getContext()->getMechanicalState();
    else
        getContext()->get(_mstate);

    VisualModel::init();

    core::loader::VoxelLoader *loader;
    getContext()->get(loader);
    if(loader && _mstate)
    {
        unsigned int nbPoints = _mstate->getSize();

        const helper::vector<unsigned int> idxInRegularGrid = loader->getHexaIndicesInGrid();


        if(idxInRegularGrid.size() == nbPoints)
        {
            const unsigned char *imageData = loader->getData();

            helper::vector<unsigned char>* pData = pointData.beginEdit();
            for(unsigned int i=0; i<nbPoints; ++i)
                (*pData).push_back(imageData[idxInRegularGrid[i]]);
            pointData.endEdit();
        }
    }

    reinit();

    updateVisual();
}

void PointSplatModel::reinit()
{
    if(texture_data != NULL)
        delete [] texture_data;

    unsigned int texture_size = textureSize.getValue();
    unsigned int half_texture_size = texture_size >> 1;
    texture_data = new unsigned char [texture_size * texture_size];

    for(unsigned int i=0; i<half_texture_size; ++i)
        for(unsigned int j=0; j<half_texture_size; ++j)
        {
            const float x = i / (float) half_texture_size;
            const float y = j / (float) half_texture_size;
            const float dist = sqrt(x*x + y*y);
            const float value = cos(3.141592f/2.0f * dist);
            const unsigned char texValue = (value < 0.0f) ? 0 : (unsigned char) (255.0f * alpha.getValue() * value);

            texture_data[half_texture_size + i + (half_texture_size + j) * texture_size] = texValue;
            texture_data[half_texture_size + i + (half_texture_size - 1 - j) * texture_size] = texValue;
            texture_data[half_texture_size - 1 - i + (half_texture_size + j) * texture_size] = texValue;
            texture_data[half_texture_size - 1 - i + (half_texture_size - 1 - j) * texture_size] = texValue;
        }

    setColor(color.getValue());
}

void PointSplatModel::drawTransparent(const core::visual::VisualParams* vparams)
{
    if(!vparams->displayFlags().getShowVisualModels()) return;

    const helper::vector<unsigned char>& pData = pointData.getValue();

    glPushAttrib(GL_ENABLE_BIT);

    glDisable(GL_LIGHTING);

//	glClearColor (0, 0, 0, 0);
    glEnable (GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

    glEnable (GL_TEXTURE_2D);
    glTexEnvf (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
//	glDepthMask(GL_FALSE);
//	glDisable(GL_DEPTH_TEST);

    GLuint	texture;
    unsigned int texture_size = (unsigned) textureSize.getValue();

    glGenTextures (1, &texture);
    glBindTexture (GL_TEXTURE_2D, texture);
    glTexImage2D (GL_TEXTURE_2D, 0, GL_ALPHA, texture_size, texture_size, 0,
            GL_ALPHA, GL_UNSIGNED_BYTE, texture_data);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);


    VecCoord	vertices;

    if (_mstate)
    {
        vertices.resize(_mstate->getSize());

        for (unsigned int i=0; i<vertices.size(); i++)
        {
            vertices[i][0] = (Real)_mstate->getPX(i);
            vertices[i][1] = (Real)_mstate->getPY(i);
            vertices[i][2] = (Real)_mstate->getPZ(i);
        }
    }


    glPushMatrix ();

    float mat[16];
    glGetFloatv( GL_MODELVIEW_MATRIX, mat );

    Coord vRight( mat[0], mat[4], mat[8] );
    Coord vUp( mat[1], mat[5], mat[9] );
    Coord vView(vRight.cross(vUp));

    std::multimap< Real, unsigned int, std::less<Real> >	mapPnt;

    for(unsigned int i=0; i<vertices.size(); ++i)
    {
        const Coord&	vertex = vertices[i];

        Real dist(vertex * vView);

        mapPnt.insert(std::pair<Real, unsigned int>(dist, i));
    }

    for(std::multimap<Real, unsigned int>::const_iterator it = mapPnt.begin();
        it != mapPnt.end(); ++it)
    {
        int i = (*it).second;

        const Coord& center = vertices[i];

        const float qsize = radius.getValue();

        // Now, build a quad around the center point based on the vRight
        // and vUp vectors. This will guarantee that the quad will be
        // orthogonal to the view.
        Coord vPoint0(center + ((-vRight - vUp) * qsize));
        Coord vPoint1(center + (( vRight - vUp) * qsize));
        Coord vPoint2(center + (( vRight + vUp) * qsize));
        Coord vPoint3(center + ((-vRight + vUp) * qsize));

        // TODO: modulate color by data
        float m = 1.0f;

        if(!pData.empty())
        {
            m = 0.5f + 2.0f * pData[i] / 255.0f;
        }

        glColor4f (m*r, m*g, m*b, a);

        glBegin( GL_QUADS );
        glTexCoord2f(0.0f, 0.0f);
        glVertex3f((float) vPoint0[0], (float) vPoint0[1], (float) vPoint0[2]);
        glTexCoord2f(1.0f, 0.0f);
        glVertex3f((float) vPoint1[0], (float) vPoint1[1], (float) vPoint1[2]);
        glTexCoord2f(1.0f, 1.0f);
        glVertex3f((float) vPoint2[0], (float) vPoint2[1], (float) vPoint2[2]);
        glTexCoord2f(0.0f, 1.0f);
        glVertex3f((float) vPoint3[0], (float) vPoint3[1], (float) vPoint3[2]);
        glEnd();
    }

    // restore the previously stored modelview matrix
    glPopMatrix ();

    // glDepthMask(GL_TRUE);
    // glEnable(GL_DEPTH_TEST);

    glPopAttrib();
}

//void PointSplatModel::handleTopologyChange()
//{
//	std::list<const TopologyChange *>::const_iterator itBegin=_topology->beginChange();
//	std::list<const TopologyChange *>::const_iterator itEnd=_topology->endChange();
//
//	while( itBegin != itEnd )
//	{
//		core::topology::TopologyChangeType changeType = (*itBegin)->getChangeType();
//
//		switch( changeType )
//		{
//			case core::topology::POINTSREMOVED:
//			{
//
//				break;
//			}
//
//			default:
//				break;
//		}; // switch( changeType )
//
//		++itBegin;
//	} // while( changeIt != last; )
//
//}

void PointSplatModel::setColor(float r, float g, float b, float a)
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

void PointSplatModel::setColor(std::string color)
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

