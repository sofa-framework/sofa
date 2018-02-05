/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

#include <sofa/defaulttype/RGBAColor.h>

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
      color(initData(&color, defaulttype::RGBAColor(1.0,1.0,1.0,1.0), "color", "Billboard color.(default=[1.0,1.0,1.0,1.0])")),
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
    {
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
    }
}

void PointSplatModel::drawTransparent(const core::visual::VisualParams* vparams)
{
    if(!vparams->displayFlags().getShowVisualModels()) return;

    const helper::vector<unsigned char>& pData = pointData.getValue();

    glPushAttrib(GL_ENABLE_BIT);

    glDisable(GL_LIGHTING);
    glEnable (GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

    glEnable (GL_TEXTURE_2D);
    glTexEnvf (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

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

        float m = 1.0f;

        if(!pData.empty())
        {
            m = 0.5f + 2.0f * pData[i] / 255.0f;
        }

        const defaulttype::RGBAColor& mc = color.getValue() ;
        glColor4f (m*mc.r(), m*mc.g(), m*mc.b(), mc.a());

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
    glPopAttrib();
}

} // namespace visualmodel

} // namespace component

} // namespace sofa

