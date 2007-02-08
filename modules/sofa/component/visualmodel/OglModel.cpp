#include <sofa/component/visualmodel/OglModel.h>
#include <sofa/helper/gl/RAII.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sstream>

namespace sofa
{

namespace component
{

namespace visualmodel
{

using namespace sofa::defaulttype;

void create(OglModel*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    obj = new OglModel;

    if (arg->getAttribute("normals")!=NULL)
        obj->setUseNormals(atoi(arg->getAttribute("normals"))!=0);

    std::string filename = arg->getAttribute("filename","");
    std::string loader = arg->getAttribute("loader","");
    std::string texturename = arg->getAttribute("texturename","");
    obj->load(filename, loader, texturename);
    if (arg->getAttribute("flip")!=NULL)
        obj->flipFaces();

    if (arg->getAttribute("color"))
        obj->setColor(arg->getAttribute("color"));
    if (arg->getAttribute("scale")!=NULL)
        obj->applyScale(atof(arg->getAttribute("scale","1.0")));
    if (arg->getAttribute("scaleTex")!=NULL)
        obj->applyUVScale(atof(arg->getAttribute("scaleTex","1.0")), atof(arg->getAttribute("scaleTex","1.0")));
    if (arg->getAttribute("dx")!=NULL || arg->getAttribute("dy")!=NULL || arg->getAttribute("dz")!=NULL)
        obj->applyTranslation(atof(arg->getAttribute("dx","0.0")),atof(arg->getAttribute("dy","0.0")),atof(arg->getAttribute("dz","0.0")));
}

SOFA_DECL_CLASS(OglModel)

helper::Creator<simulation::tree::xml::ObjectFactory, OglModel > OglModelClass("OglModel");

Material& Material::operator=(const helper::io::Mesh::Material &matLoaded)
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
        ambient[i] = 0.75;
        diffuse[i] = 0.75;
        specular[i] = 1.0;
        emissive[i] = 0.0;
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

OglModel::OglModel() //const std::string &name, std::string filename, std::string loader, std::string textureName)
    : modified(false), useTopology(false), useNormals(true), tex(NULL)
{
    inputVertices = &vertices;
    //init (name, filename, loader, textureName);
}

OglModel::~OglModel()
{
    if (tex!=NULL) delete tex;
    if (inputVertices != &vertices) delete inputVertices;
}

void OglModel::draw()
{
    //std::cerr<<"	OglModel::draw()"<<std::endl;
    if (!getContext()->getShowVisualModels()) return;
    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glEnable(GL_LIGHTING);
    //Enable<GL_BLEND> blending;
    //glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
    glColor3f(1.0 , 1.0, 1.0);
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

    if (!triangles.empty())
        glDrawElements(GL_TRIANGLES, triangles.size() * 3, GL_UNSIGNED_INT, triangles.getData());
    if (!quads.empty())
        glDrawElements(GL_QUADS, quads.size() * 4, GL_UNSIGNED_INT, quads.getData());

//	if ((tex != NULL) && (texCoords != NULL))
//		tex->bind();
//	for (int i = 0; i < nbFacets; i++)
//	{
//		glBegin(GL_TRIANGLES);
//		for (int j = 0; j < 3; j++)
//		{
//			if ((tex != NULL) && (texCoords != NULL))
//				glTexCoord2d(texCoords[texCoordIndices[i * 3 + j] * 2], texCoords[(texCoordIndices[i * 3 + j] * 2) + 1]);
//			glNormal3d(normals[(normalsIndices[i * 3 + j]) * 3], normals[(normalsIndices[i * 3 + j] * 3) + 1], normals[(normalsIndices[i * 3 + j] * 3) + 2]);
//			glVertex3d(vertices[(facets[i * 3 + j]) * 3], vertices[(facets[i * 3 + j] * 3) + 1], vertices[(facets[i * 3 + j] * 3) + 2]);
//		}
//		glEnd();
//	}

    if (tex)
    {
        tex->unbind();
        glDisableClientState(GL_TEXTURE_COORD_ARRAY);
        glDisable(GL_TEXTURE_2D);
    }
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisable(GL_LIGHTING);

    if (getContext()->getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    if (getContext()->getShowNormals())
    {
        glColor3f (1.0, 1.0, 1.0);
        for (unsigned int i = 0; i < vertices.size(); i++)
        {
            glBegin(GL_LINES);
            glVertex3fv (vertices[i].ptr());
            Coord p = vertices[i] + vnormals[i];
            glVertex3fv (p.ptr());
            glEnd();
        }
    }
}

bool OglModel::addBBox(double* minBBox, double* maxBBox)
{
    const ResizableExtVector<Coord>& x = vertices;
    for (unsigned int i=0; i<x.size(); i++)
    {
        const Coord& p = x[i];
        for (int c=0; c<3; c++)
        {
            if (p[c] > maxBBox[c]) maxBBox[c] = p[c];
            if (p[c] < minBBox[c]) minBBox[c] = p[c];
        }
    }
    return true;
}

bool OglModel::load(const std::string& filename, const std::string& loader, const std::string& textureName)
{
    if (textureName != "")
    {
        helper::io::Image *img = helper::io::Image::Create(textureName);
        if (img)
        {
            tex = new helper::gl::Texture(img);
        }
    }

    if (filename != "")
    {
        helper::io::Mesh *objLoader;
        if (loader.empty())
            objLoader = helper::io::Mesh::Create(filename);
        else
            objLoader = helper::io::Mesh::Factory::CreateObject(loader, filename);

        if (!objLoader)
        {
            return false;
        }
        else
        {
            vector< vector< vector<int> > > &facetsImport = objLoader->getFacets();
            vector<Vector3> &verticesImport = objLoader->getVertices();
            vector<Vector3> &normalsImport = objLoader->getNormals();
            vector<Vector3> &texCoordsImport = objLoader->getTexCoords();

            helper::io::Mesh::Material &materialImport = objLoader->getMaterial();

            if (materialImport.activated)
                material = materialImport;

            std::cout << "Vertices Import size : " << verticesImport.size() << " (" << normalsImport.size() << " normals)." << std::endl;

            int nbVIn = verticesImport.size();
            // First we compute for each point how many pair of normal/texcoord indices are used
            // The map store the final index of each combinaison
            vector< std::map< std::pair<int,int>, int > > vertTexNormMap;
            vertTexNormMap.resize(nbVIn);
            for (unsigned int i = 0; i < facetsImport.size(); i++)
            {
                vector<vector <int> > vertNormTexIndex = facetsImport[i];
                if (vertNormTexIndex[0].size() < 3) continue; // ignore lines
                vector<int> verts = vertNormTexIndex[0];
                vector<int> texs = vertNormTexIndex[1];
                vector<int> norms = vertNormTexIndex[2];
                for (unsigned int j = 0; j < verts.size(); j++)
                {
                    vertTexNormMap[verts[j]][std::make_pair((tex!=NULL?texs[j]:-1), (useNormals?norms[j]:0))] = 0;
                }
            }

            // Then we can compute how many vertices are created
            int nbVOut = 0;
            bool vsplit = false;
            for (int i = 0; i < nbVIn; i++)
            {
                int s = vertTexNormMap[i].size();
                nbVOut += s;
                if (s!=1)
                    vsplit = true;
            }

            // Then we can create the final arrays

            vertices.resize(nbVOut);
            vnormals.resize(nbVOut);

            if (tex)
                vtexcoords.resize(nbVOut);

            if (vsplit)
            {
                inputVertices = new ResizableExtVector<Coord>;
                inputVertices->resize(nbVIn);
                vertPosIdx.resize(nbVOut);
                vertNormIdx.resize(nbVOut);
            }
            else
                inputVertices = &vertices;

            int nbNOut = 0; /// Number of different normals
            for (int i = 0, j = 0; i < nbVIn; i++)
            {
                if (vsplit)
                    (*inputVertices)[i] = verticesImport[i];
                std::map<int, int> normMap;
                for (std::map<std::pair<int, int>, int>::iterator it = vertTexNormMap[i].begin();
                        it != vertTexNormMap[i].end(); ++it)
                {
                    vertices[j] = verticesImport[i];
                    int t = it->first.first;
                    int n = it->first.second;
                    if ((unsigned)n < normalsImport.size())
                        vnormals[j] = normalsImport[n];
                    if ((unsigned)t < texCoordsImport.size())
                        vtexcoords[j] = texCoordsImport[t];
                    if (vsplit)
                    {
                        vertPosIdx[j] = i;
                        if (normMap.count(n))
                            vertNormIdx[j] = normMap[n];
                        else
                            vertNormIdx[j] = normMap[n] = nbNOut++;
                    }
                    it->second = j++;
                }
            }

            std::cout << "Vertices Export size : " << nbVOut << " (" << nbNOut << " normals)." << std::endl;

            std::cout << "Facets Import size : " << facetsImport.size() << std::endl;

            // Then we create the triangles and quads

            for (unsigned int i = 0; i < facetsImport.size(); i++)
            {
                vector<vector <int> > vertNormTexIndex = facetsImport[i];
                if (vertNormTexIndex[0].size() < 3) continue; // ignore lines
                vector<int> verts = vertNormTexIndex[0];
                vector<int> texs = vertNormTexIndex[1];
                vector<int> norms = vertNormTexIndex[2];
                vector<int> idxs;
                idxs.resize(verts.size());
                for (unsigned int j = 0; j < verts.size(); j++)
                    idxs[j] = vertTexNormMap[verts[j]][std::make_pair((tex!=NULL?texs[j]:-1), (useNormals?norms[j]:0))];

                if (verts.size() == 4)
                {
                    // quad
                    quads.push_back(helper::make_array(idxs[0],idxs[1],idxs[2],idxs[3]));
                }
                else
                {
                    // triangle(s)
                    for (unsigned int j = 2; j < verts.size(); j++)
                    {
                        triangles.push_back(helper::make_array(idxs[0],idxs[j-1],idxs[j]));
                    }
                }
            }

            std::cout << "Facets Export size : ";
            if (!triangles.empty())
                std::cout << triangles.size() << " triangles";
            if (!quads.empty())
                std::cout << quads.size() << " quads";
            std::cout << "." << std::endl;

            computeNormals();
        }
    }
    else
    {
        useTopology = true;
        modified = true;
    }
    return true;
}

void OglModel::applyTranslation(double dx, double dy, double dz)
{
    Vector3 d((GLfloat)dx,(GLfloat)dy,(GLfloat)dz);
    VecCoord& x = *getX();
    for (unsigned int i = 0; i < x.size(); i++)
    {
        x[i] += d;
    }
    update();
}

void OglModel::applyScale(double scale)
{
    VecCoord& x = *getX();
    for (unsigned int i = 0; i < x.size(); i++)
    {
        x[i] *= (GLfloat) scale;
    }
    update();
}

void OglModel::applyUVScale(double scaleU, double scaleV)
{
    for (unsigned int i = 0; i < vtexcoords.size(); i++)
    {
        vtexcoords[i][0] *= (GLfloat) scaleU;
        vtexcoords[i][1] *= (GLfloat) scaleV;
    }
}

void OglModel::init()
{
    update();
}

void OglModel::computeNormals()
{
    vector<Coord> normals;
    int nbn = 0;
    bool vsplit = !vertNormIdx.empty();
    if (vsplit)
    {
        for (unsigned int i = 0; i < vertNormIdx.size(); i++)
        {
            if (vertNormIdx[i] >= nbn)
                nbn = vertNormIdx[i]+1;
        }
    }
    else
    {
        nbn = vertices.size();
    }
    normals.resize(nbn);
    for (int i = 0; i < nbn; i++)
        normals[i].clear();

    for (unsigned int i = 0; i < triangles.size() ; i++)
    {
        const Coord & v1 = vertices[triangles[i][0]];
        const Coord & v2 = vertices[triangles[i][1]];
        const Coord & v3 = vertices[triangles[i][2]];
        Coord n = cross(v2-v1, v3-v1);
        n.normalize();
        normals[(vsplit ? vertNormIdx[triangles[i][0]] : triangles[i][0])] += n;
        normals[(vsplit ? vertNormIdx[triangles[i][1]] : triangles[i][1])] += n;
        normals[(vsplit ? vertNormIdx[triangles[i][2]] : triangles[i][2])] += n;
    }

    for (unsigned int i = 0; i < quads.size() ; i++)
    {
        const Coord & v1 = vertices[quads[i][0]];
        const Coord & v2 = vertices[quads[i][1]];
        const Coord & v3 = vertices[quads[i][3]];
        Coord n = cross(v2-v1, v3-v1);
        n.normalize();
        normals[(vsplit ? vertNormIdx[quads[i][0]] : quads[i][0])] += n;
        normals[(vsplit ? vertNormIdx[quads[i][1]] : quads[i][1])] += n;
        normals[(vsplit ? vertNormIdx[quads[i][2]] : quads[i][2])] += n;
        normals[(vsplit ? vertNormIdx[quads[i][3]] : quads[i][3])] += n;
    }

    for (unsigned int i = 0; i < normals.size(); i++)
    {
        normals[i].normalize();
    }
    vnormals.resize(vertices.size());
    for (unsigned int i = 0; i < vertices.size(); i++)
    {
        vnormals[i] = normals[(vsplit ? vertNormIdx[i] : i)];
    }
}

void OglModel::flipFaces()
{
    for (unsigned int i = 0; i < triangles.size() ; i++)
    {
        int temp = triangles[i][1];
        triangles[i][1] = triangles[i][2];
        triangles[i][2] = temp;
    }
    for (unsigned int i = 0; i < vnormals.size(); i++)
    {
        vnormals[i] = -vnormals[i];
    }
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

const OglModel::VecCoord* OglModel::getX()  const
{
    return inputVertices;
}

OglModel::VecCoord* OglModel::getX()
{
    modified = true;
    return inputVertices;
}

void OglModel::update()
{
    if (modified && !vertices.empty())
    {
        if (!vertPosIdx.empty())
        {
            // Need to transfer positions
            for (unsigned int i=0 ; i < vertices.size(); ++i)
                vertices[i] = (*inputVertices)[vertPosIdx[i]];
        }

        if (useTopology)
        {
            topology::MeshTopology* topology = dynamic_cast<topology::MeshTopology*>(getContext()->getTopology());
            if (topology != NULL)
            {
                const vector<topology::MeshTopology::Triangle>& inputTriangles = topology->getTriangles();
                triangles.resize(inputTriangles.size());
                for (unsigned int i=0; i<triangles.size(); ++i)
                    triangles[i] = inputTriangles[i];
                const vector<topology::MeshTopology::Quad>& inputQuads = topology->getQuads();
                quads.resize(inputQuads.size());
                for (unsigned int i=0; i<quads.size(); ++i)
                    quads[i] = inputQuads[i];
            }
        }

        computeNormals();
        modified = false;
    }
}

void OglModel::initTextures()
{
    if (tex)
    {
        tex->init();
    }
}

void OglModel::exportOBJ(std::string name, std::ostream* out, std::ostream* mtl, int& vindex, int& nindex, int& tindex)
{
    *out << "g "<<name<<"\n";

    if (mtl != NULL) // && !material.name.empty())
    {
        std::string name; // = material.name;
        if (name.empty())
        {
            static int count = 0;
            std::ostringstream o; o << "mat" << ++count;
            name = o.str();
        }
        *mtl << "newmtl "<<name<<"\n";
        *mtl << "illum 4\n";
        if (material.useAmbient)
            *mtl << "Ka "<<material.ambient[0]<<' '<<material.ambient[1]<<' '<<material.ambient[2]<<"\n";
        if (material.useDiffuse)
            *mtl << "Kd "<<material.diffuse[0]<<' '<<material.diffuse[1]<<' '<<material.diffuse[2]<<"\n";
        *mtl << "Tf 1.00 1.00 1.00\n";
        *mtl << "Ni 1.00\n";
        if (material.useSpecular)
            *mtl << "Ks "<<material.specular[0]<<' '<<material.specular[1]<<' '<<material.specular[2]<<"\n";
        if (material.useShininess)
            *mtl << "Ns "<<material.shininess<<"\n";

        *out << "usemtl "<<name<<'\n';
    }
    const ResizableExtVector<Coord>& x = *inputVertices;

    int nbv = x.size();

    for (unsigned int i=0; i<x.size(); i++)
    {
        *out << "v "<< std::fixed << x[i][0]<<' '<< std::fixed <<x[i][1]<<' '<< std::fixed <<x[i][2]<<'\n';
    }

    int nbn = 0;

    if (vertNormIdx.empty())
    {
        nbn = vnormals.size();
        for (unsigned int i=0; i<vnormals.size(); i++)
        {
            *out << "vn "<< std::fixed << vnormals[i][0]<<' '<< std::fixed <<vnormals[i][1]<<' '<< std::fixed <<vnormals[i][2]<<'\n';
        }
    }
    else
    {
        for (unsigned int i = 0; i < vertNormIdx.size(); i++)
        {
            if (vertNormIdx[i] >= nbn)
                nbn = vertNormIdx[i]+1;
        }
        vector<int> normVertIdx(nbn);
        for (unsigned int i = 0; i < vertNormIdx.size(); i++)
        {
            normVertIdx[vertNormIdx[i]]=i;
        }
        for (int i = 0; i < nbn; i++)
        {
            int j = normVertIdx[i];
            *out << "vn "<< std::fixed << vnormals[j][0]<<' '<< std::fixed <<vnormals[j][1]<<' '<< std::fixed <<vnormals[j][2]<<'\n';
        }
    }

    int nbt = 0;
    if (!vtexcoords.empty())
    {
        nbt = vtexcoords.size();
        for (unsigned int i=0; i<vtexcoords.size(); i++)
        {
            *out << "vt "<< std::fixed << vtexcoords[i][0]<<' '<< std::fixed <<vtexcoords[i][1]<<'\n';
        }
    }

    for (unsigned int i = 0; i < triangles.size() ; i++)
    {
        *out << "f";
        for (int j=0; j<3; j++)
        {
            int i0 = triangles[i][j];
            int i_p = vertPosIdx.empty() ? i0 : vertPosIdx[i0];
            int i_n = vertNormIdx.empty() ? i0 : vertNormIdx[i0];
            if (vtexcoords.empty())
                *out << ' ' << i_p+vindex+1 << "//" << i_n+nindex+1;
            else
                *out << ' ' << i_p+vindex+1 << '/' << i0+tindex+1 << '/' << i_n+nindex+1;
        }
        *out << '\n';
    }
    for (unsigned int i = 0; i < quads.size() ; i++)
    {
        *out << "f";
        for (int j=0; j<4; j++)
        {
            int i0 = quads[i][j];
            int i_p = vertPosIdx.empty() ? i0 : vertPosIdx[i0];
            int i_n = vertNormIdx.empty() ? i0 : vertNormIdx[i0];
            if (vtexcoords.empty())
                *out << ' ' << i_p+vindex+1 << "//" << i_n+nindex+1;
            else
                *out << ' ' << i_p+vindex+1 << '/' << i0+tindex+1 << '/' << i_n+nindex+1;
        }
        *out << '\n';
    }
    *out << std::endl;
    vindex+=nbv;
    nindex+=nbn;
    tindex+=nbt;
}

} // namespace visualmodel

} // namespace component

} // namespace sofa

