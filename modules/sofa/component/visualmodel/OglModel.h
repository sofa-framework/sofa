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
#ifndef SOFA_COMPONENT_VISUALMODEL_OGLMODEL_H
#define SOFA_COMPONENT_VISUALMODEL_OGLMODEL_H

#include <string>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/Texture.h>
#include <sofa/core/VisualModel.h>
#include <sofa/core/componentmodel/behavior/MappedModel.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/helper/io/Mesh.h>

#ifdef _WIN32
#include <windows.h>
#endif // _WIN32



namespace sofa
{

namespace component
{

namespace visualmodel
{

using namespace sofa::defaulttype;

struct Material
{
    std::string	name;		/* name of material */
    GLfloat	diffuse[4];	/* diffuse component */
    GLfloat	ambient[4];	/* ambient component */
    GLfloat	specular[4];	/* specular component */
    GLfloat	emissive[4];	/* emmissive component */
    GLfloat	shininess;	/* specular exponent */
    bool useDiffuse;
    bool useSpecular;
    bool useAmbient;
    bool useEmissive;
    bool useShininess;

    Material& operator= (const helper::io::Mesh::Material &matLoaded);
    void setColor(float r, float g, float b, float a);

    Material();
};

/// Resizable custom vector class.
template<class T>
class ResizableExtVector : public ExtVector<T>
{
public:
    ~ResizableExtVector()
    {
        if (this->data!=NULL) delete[] this->data;
    }
    T* getData() { return this->data; }
    const T* getData() const { return this->data; }
    virtual void resize(typename ExtVector<T>::size_type size)
    {
        if (size > this->maxsize)
        {
            T* oldData = this->data;
            this->maxsize = (size > 2*this->maxsize ? size : 2*this->maxsize);
            this->data = new T[this->maxsize];
            for (typename ExtVector<T>::size_type i = 0 ; i < this->cursize ; ++i)
                this->data[i] = oldData[i];
            if (oldData!=NULL) delete[] oldData;
        }
        this->cursize = size;
    }
    void push_back(const T& v)
    {
        int i = this->size();
        resize(i+1);
        (*this)[i] = v;
    }
};

class OglModel : public core::VisualModel, public core::componentmodel::behavior::MappedModel< ExtVectorTypes< Vec<3,GLfloat>, Vec<3,GLfloat> > >
{
private:
    typedef Vec<2, GLfloat> TexCoord;
    typedef helper::fixed_array<int, 3> Triangle;
    typedef helper::fixed_array<int, 4> Quad;

    ResizableExtVector<Coord>* inputVertices;

    bool modified; ///< True if input vertices modified since last rendering
    bool useTopology; ///< True if list of facets should be taken from the attached topology
    bool useNormals; ///< True if normals should be read from file
    bool castShadow; ///< True if object cast shadows

    ResizableExtVector<Coord> vertices;
    ResizableExtVector<Coord> vnormals;
    ResizableExtVector<TexCoord> vtexcoords;

    ResizableExtVector<Triangle> triangles;
    ResizableExtVector<Quad> quads;

    /// If vertices have multiple normals/texcoords, then we need to separate them
    /// This vector store which input position is used for each vertice
    /// If it is empty then each vertex correspond to one position
    ResizableExtVector<int> vertPosIdx;

    /// Similarly this vector store which input normal is used for each vertice
    /// If it is empty then each vertex correspond to one normal
    ResizableExtVector<int> vertNormIdx;

    Material material;

    helper::gl::Texture *tex;

    void internalDraw();

public:

    OglModel();

    ~OglModel();

    void parse(core::objectmodel::BaseObjectDescription* arg);

    bool isTransparent();

    void draw();
    void drawTransparent();
    void drawShadow();

    bool load(const std::string& filename, const std::string& loader, const std::string& textureName);

    void applyTranslation(double dx, double dy, double dz);
    void applyRotation(Quat q);
    void applyScale(double s);
    void applyUVScale(double su, double sv);
    void computeNormals();

    void flipFaces();

    void setColor(float r, float g, float b, float a);
    void setColor(std::string color);

    void setUseNormals(bool val) { useNormals = val;  }
    bool getUseNormals() const   { return useNormals; }

    void setCastShadow(bool val) { castShadow = val;  }
    bool getCastShadow() const   { return castShadow; }

    void update();

    void init();

    void initTextures();

    bool addBBox(double* minBBox, double* maxBBox);

    const VecCoord* getX()  const; // { return &x;   }
    const VecDeriv* getV()  const { return NULL; }
    /*
    const VecDeriv* getF()  const { return NULL; }
    const VecDeriv* getDx() const { return NULL; }
    */

    VecCoord* getX(); //  { return &x;   }
    VecDeriv* getV()  { return NULL; }
    /*
    VecDeriv* getF()  { return NULL; }
    VecDeriv* getDx() { return NULL; }
    */

    /// Append this mesh to an OBJ format stream.
    /// The number of vertices position, normal, and texture coordinates already written is given as parameters
    /// This method should update them
    virtual void exportOBJ(std::string name, std::ostream* out, std::ostream* mtl, int& vindex, int& nindex, int& tindex);
};

typedef Vec<3,GLfloat> GLVec3f;
typedef ExtVectorTypes<GLVec3f,GLVec3f> GLExtVec3fTypes;

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif
