#ifndef SOFA_COMPONENTS_GL_OGLMODEL_H
#define SOFA_COMPONENTS_GL_OGLMODEL_H

#include <string>

#ifdef _WIN32
#include <windows.h>
#endif // _WIN32

#include <GL/gl.h>

#include "Texture.h"
#include "Sofa/Abstract/VisualModel.h"
//#include "Sofa/Components/VisualLoader.h"
//#include "Common/InterfaceVertex.h"
//#include "Common/InterfaceTransformation.h"
#include "Sofa/Core/MappedModel.h"
#include "Sofa/Components/Common/Vec.h"
#include "Sofa/Components/Common/Vec3Types.h"
#include "Sofa/Components/Common/Mesh.h"

namespace Sofa
{

namespace Components
{

namespace GL
{

using namespace Common;

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

    Material& operator= (const Mesh::Material &matLoaded);
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
        if (data!=NULL) delete[] data;
    }
    T* getData() { return data; }
    const T* getData() const { return data; }
    virtual void resize(size_type size)
    {
        if (size > maxsize)
        {
            T* oldData = data;
            maxsize = (size > 2*maxsize ? size : 2*maxsize);
            data = new T[maxsize];
            for (size_type i = 0 ; i < cursize ; ++i)
                data[i] = oldData[i];
            if (oldData!=NULL) delete[] oldData;
        }
        cursize = size;
    }
    void push_back(const T& v)
    {
        int i = size();
        resize(i+1);
        (*this)[i] = v;
    }
};

class OglModel : public Abstract::VisualModel, public Core::MappedModel< ExtVectorTypes< Vec<3,GLfloat>, Vec<3,GLfloat> > >
{
private:
    typedef Vec<2, GLfloat> TexCoord;
    typedef fixed_array<int, 3> Triangle;
    typedef fixed_array<int, 4> Quad;

    ResizableExtVector<Coord>* inputVertices;

    bool modified; ///< True if input vertices modified since last rendering
    bool useTopology; ///< True if list of facets should be taken from the attached topology

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

    Texture *tex;

    //double matTransOpenGL[16];

public:

    OglModel(const std::string &name="", std::string filename="", std::string loader="", std::string textureName="");

    ~OglModel();

    void draw();

    void init(const std::string &name, std::string filename, std::string loader, std::string textureName);
    void applyTranslation(double dx, double dy, double dz);
    void applyScale(double s);
    void computeNormals();

    void setColor(float r, float g, float b, float a);
    void setColor(std::string color);

    void update();

    void init();

    void initTextures();

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
};

} // namespace GL

} // namespace Components

} // namespace Sofa

#endif
