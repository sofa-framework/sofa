/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/component/visualmodel/OglModel.h>
#include <sofa/helper/gl/RAII.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sstream>

namespace sofa
{

namespace component
{

namespace visualmodel
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(OglModel)

int OglModelClass = core::RegisterObject("Generic visual model for OpenGL display")
        .add< OglModel >()
        ;


OglModel::OglModel()
    : tex(NULL)
{
}

OglModel::~OglModel()
{
    if (tex!=NULL) delete tex;
}

void OglModel::internalDraw()
{
    /*if (getContext()->getMainTopology())
    {
        int nbp = getContext()->get<core::componentmodel::behavior::BaseMechanicalState>()->getSize();
        int nbv = vertices.size();
        if (nbp != nbv)
        {
            std::cerr << "nbp mismatch: "<<nbp<<" "<<nbv<<std::endl;
        }
    }*/
    //std::cerr<<" OglModel::draw()"<<std::endl;
    if (!getContext()->getShowVisualModels()) return;
    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glEnable(GL_LIGHTING);

    //Enable<GL_BLEND> blending;
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
    glColor3f(1.0 , 1.0, 1.0);
    Vec4f ambient = material.getValue().useAmbient?material.getValue().ambient:Vec4f();
    Vec4f diffuse = material.getValue().useDiffuse?material.getValue().diffuse:Vec4f();
    Vec4f specular = material.getValue().useSpecular?material.getValue().specular:Vec4f();
    Vec4f emissive = material.getValue().useEmissive?material.getValue().emissive:Vec4f();
    float shininess = material.getValue().useShininess?material.getValue().shininess:45;

    if (isTransparent())
    {
        emissive[3] = 0; //diffuse[3];
        ambient[3] = 0; //diffuse[3];
        //diffuse[3] = 0;
        specular[3] = 0;
    }
    glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT, ambient.ptr());
    glMaterialfv (GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse.ptr());
    glMaterialfv (GL_FRONT_AND_BACK, GL_SPECULAR, specular.ptr());
    glMaterialfv (GL_FRONT_AND_BACK, GL_EMISSION, emissive.ptr());
    glMaterialf (GL_FRONT_AND_BACK, GL_SHININESS, shininess);

    glVertexPointer (3, GL_FLOAT, 0, vertices.getData());
    glNormalPointer (GL_FLOAT, 0, vnormals.getData());
    glEnableClientState(GL_NORMAL_ARRAY);
    if (tex)
    {
        glEnable(GL_TEXTURE_2D);
        tex->bind();
        glTexCoordPointer(2, GL_FLOAT, 0, vtexcoords.getData());
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    }

    if (isTransparent())
    {
        glEnable(GL_BLEND);
        glDepthMask(GL_FALSE);
        glBlendFunc(GL_ZERO, GL_ONE_MINUS_SRC_ALPHA);

        for (unsigned int i=0; i<xforms.size(); i++)
        {
            float matrix[16];
            xforms[i].writeOpenGlMatrix(matrix);
            glPushMatrix();
            glMultMatrixf(matrix);

            if (!triangles.empty())
                glDrawElements(GL_TRIANGLES, triangles.size() * 3, GL_UNSIGNED_INT, triangles.getData());
            if (!quads.empty())
                glDrawElements(GL_QUADS, quads.size() * 4, GL_UNSIGNED_INT, quads.getData());

            glPopMatrix();
        }
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    }

    for (unsigned int i=0; i<xforms.size(); i++)
    {
        float matrix[16];
        xforms[i].writeOpenGlMatrix(matrix);
        glPushMatrix();
        glMultMatrixf(matrix);

        if (!triangles.empty())
            glDrawElements(GL_TRIANGLES, triangles.size() * 3, GL_UNSIGNED_INT, triangles.getData());
        if (!quads.empty())
            glDrawElements(GL_QUADS, quads.size() * 4, GL_UNSIGNED_INT, quads.getData());

        glPopMatrix();
    }
    if (tex)
    {
        tex->unbind();
        glDisableClientState(GL_TEXTURE_COORD_ARRAY);
        glDisable(GL_TEXTURE_2D);
    }
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisable(GL_LIGHTING);
    if (isTransparent())
    {
        glDisable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        //glBlendFunc(GL_ONE, GL_ZERO);
        glDepthMask(GL_TRUE);
    }

    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    if (getContext()->getShowNormals())
    {
        glColor3f (1.0, 1.0, 1.0);
        for (unsigned int i=0; i<xforms.size(); i++)
        {
            float matrix[16];
            xforms[i].writeOpenGlMatrix(matrix);
            glPushMatrix();
            glMultMatrixf(matrix);

            for (unsigned int i = 0; i < vertices.size(); i++)
            {
                glBegin(GL_LINES);
                glVertex3fv (vertices[i].ptr());
                Coord p = vertices[i] + vnormals[i];
                glVertex3fv (p.ptr());
                glEnd();
            }

            glPopMatrix();
        }
    }

}

bool OglModel::loadTexture(const std::string& filename)
{
    helper::io::Image *img = helper::io::Image::Create(filename);
    if (!img)
        return false;
    tex = new helper::gl::Texture(img);
    return true;
}

void OglModel::initTextures()
{
    if (tex)
    {
        tex->init();
    }
}

} // namespace visualmodel

} // namespace component

} // namespace sofa

