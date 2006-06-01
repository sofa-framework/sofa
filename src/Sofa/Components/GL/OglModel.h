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

class OglModel : public Abstract::VisualModel, public Core::MappedModel< ExtVectorTypes< Vec<3,GLfloat>, Vec<3,GLfloat> > >
{
private:
    GLfloat *vertices; /* vertices */
    GLfloat *normals; /* normals on a vertices */
    GLfloat *texCoords; /* texture coordinates : only 2 texCoord for a vertex */
    GLint *facets; /* vertices index */
    GLint *normalsIndices; /* normals index */
    GLint *texCoordIndices; /* texCoords index */
    Material material;
    int nbVertices;
    int nbFacets;
    Texture *tex;
    GLfloat *oldVertices;
    double matTransOpenGL[16];

public:

    OglModel(const std::string &name, std::string filename, std::string loader, std::string textureName);

    ~OglModel();

    void draw();
    /*
    	void applyTrans(const Vector3 &trans);
    	void applyScale(const Vector3 &scale);
    	void applyRotation(const Vector3& rotationAxis, double rotationAngle);
    */
    void init(const std::string &name, std::string filename, std::string loader, std::string textureName);
    void applyTranslation(double dx, double dy, double dz);
    void applyScale(double s);
    void computeNormal(const Vector3& s1, const Vector3& s2, const Vector3& s3, int indVertex);
    void computeNormals();

    void setColor(float r, float g, float b, float a);
    void setColor(std::string color);

    void update();

    /*
    	// For mapping interface
    	Vector3 getVertexPosition(int n) const;
    	bool setVertexDisplacement(int n, const Vector3& p);
    	Vector3 getVertexDisplacement(int) const;
    	bool setVertexPosition(int n, const Vector3& p);

    	void setMatrixTransform (const Matrix& m);
    */

    void initTextures();

    int getNumberVertices() const
    {
        return nbVertices;
    }

    VecCoord x;

    const VecCoord* getX()  const { return &x;   }
    const VecDeriv* getV()  const { return NULL; }
    /*
    const VecDeriv* getF()  const { return NULL; }
    const VecDeriv* getDx() const { return NULL; }
    */

    VecCoord* getX()  { return &x;   }
    VecDeriv* getV()  { return NULL; }
    /*
    VecDeriv* getF()  { return NULL; }
    VecDeriv* getDx() { return NULL; }
    */

    void init() { }

    void beginIteration(double /*dt*/) { }

    void endIteration(double /*dt*/) { }

    void propagateX() { computeNormals(); }

    void propagateV() { }

    void setObject(Abstract::BehaviorModel* /*obj*/) { }

    void setTopology(Core::Topology* /*topo*/) { }
};

} // namespace GL

} // namespace Components

} // namespace Sofa

#endif
