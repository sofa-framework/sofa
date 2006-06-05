#include "OglModel.h"
#include "RAII.h"
#include "../Common/Quat.h"
#include "../Common/ObjectFactory.h"

namespace Sofa
{

namespace Components
{

namespace GL
{

using namespace Common;

void create(OglModel*& obj, ObjectDescription* arg)
{
    const char* filename = arg->getAttribute("filename");
    const char* loader = arg->getAttribute("loader","");
    const char* texturename = arg->getAttribute("texturename","");
    if (!filename)
    {
        std::cerr << arg->getType() << " requires a filename attribute\n";
        obj = NULL;
    }
    else
    {
        obj = new OglModel(arg->getName(), filename, loader, texturename);
        const char* color = arg->getAttribute("color");
        if (color) obj->setColor(color);
        if (arg->getAttribute("scale")!=NULL)
            obj->applyScale(atof(arg->getAttribute("scale","1.0")));
        if (arg->getAttribute("dx")!=NULL || arg->getAttribute("dy")!=NULL || arg->getAttribute("dz")!=NULL)
            obj->applyTranslation(atof(arg->getAttribute("dx","0.0")),atof(arg->getAttribute("dy","0.0")),atof(arg->getAttribute("dz","0.0")));
    }
}

SOFA_DECL_CLASS(OglModel)

Creator< ObjectFactory, OglModel > OglModelClass("OglModel");

Material& Material::operator=(const Mesh::Material &matLoaded)
{
    for (int i = 0; i < 4; i++)
    {
        ambient[i] = (GLfloat) matLoaded.ambient[i];
        diffuse[i] = (GLfloat) matLoaded.diffuse[i];
        specular[i] = (GLfloat) matLoaded.specular[i];
        emissive[i] = 0.0;
    }
    emissive[3] = 1.0;
    shininess = (GLfloat) matLoaded.shininess;
    name = matLoaded.name;
    useDiffuse = matLoaded.useDiffuse;
    useSpecular = matLoaded.useSpecular;
    useAmbient = matLoaded.useAmbient;
    useEmissive = false;
    useShininess = matLoaded.useShininess;
    return *this;
}

Material::Material()
{
    for (int i = 0; i < 3; i++)
    {
        ambient[i] = 1.0;
        diffuse[i] = 0.5;
        specular[i] = 1.0;
        emissive[i] = 1.0;
    }
    ambient[3] = 1.0;
    diffuse[3] = 1.0;
    specular[3] = 1.0;
    emissive[3] = 1.0;

    shininess = 45;
    name = "Default";
    useAmbient = true;
    useDiffuse = true;
    useSpecular = false;
    useEmissive = false;
    useShininess = false;
}

OglModel::OglModel (const std::string &name, std::string filename, std::string loader, std::string textureName)
    : vertices(NULL), normals(NULL), texCoords(NULL), facets(NULL), normalsIndices(NULL), texCoordIndices(NULL)
    , nbVertices(0), nbFacets(0), tex(NULL), oldVertices(NULL)
{
    init (name, filename, loader, textureName);
};


OglModel::~OglModel()
{
    if (vertices) delete vertices;
    if (normals) delete normals;
    if (texCoords) delete texCoords;
    if (facets) delete facets;
    if (normalsIndices) delete normalsIndices;
    if (texCoordIndices) delete texCoordIndices;
    if (tex) delete tex;
    if (oldVertices) delete oldVertices;
}

void OglModel::draw()
{
    if (!getContext()->getShowVisualModels()) return;
    Enable<GL_TEXTURE_2D> texture;
    Enable<GL_LIGHTING> light;
    //Enable<GL_BLEND> blending;
    //glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
    //glColor3f(1.0 , 1.0, 1.0);
    if (material.useAmbient)
        glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT, material.ambient);
    if (material.useDiffuse)
        glMaterialfv (GL_FRONT_AND_BACK, GL_DIFFUSE, material.diffuse);
    if (material.useSpecular)
        glMaterialfv (GL_FRONT_AND_BACK, GL_SPECULAR, material.specular);
    if (material.useEmissive)
        glMaterialfv (GL_FRONT_AND_BACK, GL_EMISSION, material.emissive);
    if (material.useShininess)
        glMaterialf (GL_FRONT_AND_BACK, GL_SHININESS, material.shininess);

    /* 	glVertexPointer (3, GL_FLOAT, 0, vertices);
    	glNormalPointer (GL_FLOAT, 0, normals);
     	if ((tex != NULL) && (texCoords != NULL))
    	{
    		tex->bind();
    		glTexCoordPointer(2, GL_FLOAT, 0, texCoords);
    	}

    	glDrawElements(GL_TRIANGLES, nbFacets * 3, GL_UNSIGNED_INT, facets);
    	*/
    /*
    if (!deformable)
    {
    	glPushMatrix();
    	glMultMatrixd(matTransOpenGL);
    }
    */

    if ((tex != NULL) && (texCoords != NULL))
        tex->bind();

    for (int i = 0; i < nbFacets; i++)
    {
        glBegin(GL_TRIANGLES);
        for (int j = 0; j < 3; j++)
        {
            if ((tex != NULL) && (texCoords != NULL))
                glTexCoord2d(texCoords[texCoordIndices[i * 3 + j] * 2], texCoords[(texCoordIndices[i * 3 + j] * 2) + 1]);
            glNormal3d(normals[(normalsIndices[i * 3 + j]) * 3], normals[(normalsIndices[i * 3 + j] * 3) + 1], normals[(normalsIndices[i * 3 + j] * 3) + 2]);
            glVertex3d(vertices[(facets[i * 3 + j]) * 3], vertices[(facets[i * 3 + j] * 3) + 1], vertices[(facets[i * 3 + j] * 3) + 2]);
        }
        glEnd();
    }
    /*
    if (!deformable)
    {
    	glPopMatrix();
    }
    */
    if ((tex != NULL) && (texCoords != NULL))
        tex->unbind();

#ifdef DEBUGNORMAL
    glColor3f (1.0, 1.0, 1.0);
    for (int i = 0; i < nbVertices; i++)
    {
        glBegin(GL_LINES);
        glVertex3f (vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2]);
        glVertex3f (vertices[i * 3] + normals[i * 3], vertices[i * 3 + 1] + normals[i * 3 + 1], vertices[i * 3 + 2] + normals[i * 3 + 2] );
        glEnd();
    }
#endif
}

void OglModel::init(const std::string &/*name*/, std::string filename, std::string loader, std::string textureName)
{
    texCoords = NULL;

    memset(matTransOpenGL, 0, sizeof(matTransOpenGL));
    matTransOpenGL[15]=1.0;

    if (textureName != "")
    {
        Image *img = Image::Create(textureName);
        if (img)
        {
            tex = new Texture(img);
        }
    }

    Mesh *objLoader;
    if (loader.empty())
        objLoader = Mesh::Create(filename);
    else
        objLoader = Mesh::Factory::CreateObject(loader, filename);

    //OBJLoader *objLoader = static_cast<OBJLoader*> (ldr);
    if (objLoader) // finish to create the oglModel
    {
        std::vector< std::vector< std::vector<int> > > &facetsImport = objLoader->getFacets();
        std::vector<Vector3> &verticesImport = objLoader->getVertices();
        std::vector<Vector3> &normalsImport = objLoader->getNormals();
        std::vector<Vector3> &texCoordsImport = objLoader->getTexCoords();

        Mesh::Material &materialImport = objLoader->getMaterial();

        if (materialImport.activated)
            material = materialImport;

        std::cout << "Vertices Import size : " << verticesImport.size() << std::endl;
        nbVertices = verticesImport.size();
        if (nbVertices != 0)
        {
            vertices = new GLfloat[nbVertices * 3];
            oldVertices = new GLfloat[nbVertices * 3];
            for (int i = 0; i < nbVertices; i++)
            {

                for (int j = 0; j < 3; j++)
                {
                    vertices[i * 3 + j] = (GLfloat) verticesImport[i][j];
                    oldVertices[i * 3 + j] = vertices[i * 3 + j];
                }
                //xVisual.push_back(new Vector3(verticesImport[i]));
            }
            if (normalsImport.size() != 0)
            {
                normals = new GLfloat[normalsImport.size() * 3];
                for (int i = 0; i < (int) normalsImport.size(); i++)
                {
                    normals[i * 3] = (GLfloat) normalsImport[i][0];
                    normals[(i * 3) + 1] = (GLfloat) normalsImport[i][1];
                    normals[(i * 3) + 2] = (GLfloat) normalsImport[i][2];
                }
            }
            if ((texCoordsImport.size() != 0) && (textureName != ""))
            {
                texCoords = new GLfloat [texCoordsImport.size() * 2];
                for (int i = 0; i < (int) texCoordsImport.size(); i++)
                {
                    texCoords[i * 2] = (GLfloat) texCoordsImport[i][0];
                    texCoords[(i * 2) + 1] = (GLfloat) texCoordsImport[i][1];
                }
            }

            nbFacets = facetsImport.size();
            facets = new GLint[facetsImport.size() * 3];
            normalsIndices = new GLint[facetsImport.size() * 3];
            texCoordIndices = new GLint[facetsImport.size() * 3];
            for (int i = 0; i < (int) facetsImport.size(); i++)
            {
                std::vector<std::vector <int> > vertNormTexIndex = facetsImport[i];
                std::vector<int> vertices = vertNormTexIndex[0];
                std::vector<int> norms = vertNormTexIndex[1];
                std::vector<int> texs = vertNormTexIndex[2];
                for (int j = 0; j < 3; j++)
                {
                    facets[(i * 3) + j] = (GLint) vertices[j];
                    normalsIndices[(i * 3) + j] = (GLint) norms[j];
                    texCoordIndices[(i * 3) + j] = (GLint) texs[j];
                }
            }
        }
    }

    if (objLoader->getNormals().size() == 0) // compute the normals
    {
        normals = new GLfloat[nbVertices * 3];
        computeNormals();
    }

    // Link with mappings
    x.setData((Vec<3,GLfloat>*)vertices,nbVertices);
}

void OglModel::applyTranslation(double dx, double dy, double dz)
{
    for (int i = 0; i < nbVertices; i++)
    {
        vertices[i * 3] += (GLfloat)dx;
        vertices[i * 3 + 1] += (GLfloat)dy;
        vertices[i * 3 + 2] += (GLfloat)dz;
    }
}

void OglModel::applyScale(double scale)
{
    for (int i = 0; i < nbVertices; i++)
    {
        vertices[i * 3] *= (float)scale;
        vertices[i * 3 + 1] *= (float)scale;
        vertices[i * 3 + 2] *= (float)scale;
    }
}

void OglModel::computeNormals()
{
    for (int i = 0; i < nbVertices * 3; i++)
        normals[i] = 0.0;

    for (int i = 0; i < nbFacets ; i++)
    {
        int indV1 = facets[i * 3] * 3;
        int indV2 = facets[i * 3 + 1] * 3;
        int indV3 = facets[i * 3 + 2] * 3;

        Vector3 s1(vertices[indV1], vertices[indV1 + 1], vertices[indV1 + 2]);
        Vector3 s2(vertices[indV2], vertices[indV2 + 1], vertices[indV2 + 2]);
        Vector3 s3(vertices[indV3], vertices[indV3 + 1], vertices[indV3 + 2]);

        computeNormal (s1, s2, s3, indV1);
        computeNormal (s3, s1, s2, indV2);
        computeNormal (s2, s3, s1, indV3);
    }

    for (int i = 0; i < nbVertices; i++)
    {
        Vector3 norm(normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]);
        norm.normalize();
        for (int j = 0; j < 3; j++)
            normals[i * 3 + j] = (GLfloat) norm[j];
    }
}

void OglModel::computeNormal(const Vector3& s1, const Vector3 &s2, const Vector3 &s3, int indVertex)
{
    Vector3 v1 = s2 - s1;
    Vector3 v2 = s1 - s3;
    Vector3 norm = cross(v2,v1);
    norm.normalize();

    normals[indVertex] += (GLfloat) norm[0];
    normals[indVertex + 1] += (GLfloat) norm[1];
    normals[indVertex + 2] += (GLfloat) norm[2];
}

void Material::setColor(float r, float g, float b, float a)
{
    float f[4] = { r, g, b, a };
    for (int i=0; i<4; i++)
    {
        ambient[i] *= f[i];
        diffuse[i] *= f[i];
        specular[i] *= f[i];
        emissive[i] *= f[i];
    }
}

void OglModel::setColor(float r, float g, float b, float a)
{
    material.setColor(r,g,b,a);
}

static int hexval(char c)
{
    if (c>='0' && c<='9') return c-'0';
    else if (c>='a' && c<='f') return (c-'a')+10;
    else if (c>='A' && c<='F') return (c-'A')+10;
    else return 0;
}

void OglModel::setColor(std::string color)
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
        std::cerr << "Unknown color "<<color<<std::endl;
        return;
    }
    setColor(r,g,b,a);
}

void OglModel::update()
{
    computeNormals();
}

void OglModel::initTextures()
{
    if (tex)
    {
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
        tex->init();
    }
}

} // namespace GL

} // namespace Components

} // namespace Sofa
