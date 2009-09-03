/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/loader/MeshObjLoader.h>
#include <sofa/helper/system/SetDirectory.h>

namespace sofa
{

namespace component
{

namespace loader
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(MeshObjLoader)

int MeshObjLoaderClass = core::RegisterObject("Specific mesh loader for Obj file format.")
        .add< MeshObjLoader >()
        ;


MeshObjLoader::Material::Material()
{
    ambient =  Vec4f( 0.2f,0.2f,0.2f,1.0f);
    diffuse =  Vec4f( 0.75f,0.75f,0.75f,1.0f);
    specular =  Vec4f( 1.0f,1.0f,1.0f,1.0f);
    emissive =  Vec4f( 0.0f,0.0f,0.0f,0.0f);

    shininess =  45.0f;
    name = "Default";
    useAmbient =  true;
    useDiffuse =  true;
    useSpecular =  false;
    useEmissive =  false;
    useShininess =  false;
    activated = false;
}

void MeshObjLoader::Material::setColor(float r, float g, float b, float a)
{
    float f[4] = { r, g, b, a };
    for (int i=0; i<4; i++)
    {
        ambient = Vec4f(f[0]*0.2f,f[1]*0.2f,f[2]*0.2f,f[3]);
        diffuse = Vec4f(f[0],f[1],f[2],f[3]);
        specular = Vec4f(f[0],f[1],f[2],f[3]);
        emissive = Vec4f(f[0],f[1],f[2],f[3]);
    }
}




MeshObjLoader::MeshObjLoader(): MeshLoader()
    , normalsList(initData(&normalsList,"normalsList","Vertices of the mesh loaded"))
    , texturesList(initData(&texturesList,"texturesList","Edges of the mesh loaded"))
{
    normalsList.setPersistent(false);
    texturesList.setPersistent(false);
}




bool MeshObjLoader::load()
{
    std::cout << "Loading OBJ file: " << m_filename << std::endl;

    FILE* file;
    bool fileRead = false;

    // -- Loading file
    const char* filename = m_filename.getFullPath().c_str();
    if ((file = fopen(filename, "r")) == NULL)
    {
        std::cerr << "Error: MeshObjLoader: Cannot read file '" << m_filename << "'." << std::endl;
        return false;
    }

    // -- Reading file
    fileRead = this->readOBJ (file,filename);
    fclose(file);

    return fileRead;
}



bool MeshObjLoader::readOBJ (FILE* file, const char* filename)
{
    std::cout << "MeshObjLoader::readOBJ" << std::endl;


    helper::vector<sofa::defaulttype::Vector3>& my_positions = *(positions.beginEdit());
    helper::vector<sofa::defaulttype::Vector3>& my_texCoords = *(texCoords.beginEdit());
    helper::vector<sofa::defaulttype::Vector3>& my_normals   = *(normals.beginEdit());

    helper::vector<Material>& my_materials = *(materials.beginEdit());
    helper::vector< helper::vector <int> >& my_normalsList = *(normalsList.beginEdit());
    helper::vector< helper::vector <int> >& my_texturesList   = *(texturesList.beginEdit());
    helper::vector<int> nodes, nIndices, tIndices;

    helper::vector<helper::fixed_array <unsigned int,2> >& my_edges = *(edges.beginEdit());
    helper::vector<helper::fixed_array <unsigned int,3> >& my_triangles = *(triangles.beginEdit());
    helper::vector<helper::fixed_array <unsigned int,4> >& my_quads = *(quads.beginEdit());


    int vtn[3];
    char buf[128], matName[1024];
    Vec3d result;
    const char *token;

    std::string face, tmp;
    while (fscanf(file, "%s", buf) != EOF)
    {
        switch (buf[0])
        {
        case '#':
            /* comment */
            /* eat up rest of line */
            if ( fgets(buf, sizeof(buf), file) == NULL)
            {
                if (feof (file) )
                    std::cerr << "Error: MeshObjLoader: fgets function has encounter end of file. case #." << std::endl;
                else
                    std::cerr << "Error: MeshObjLoader: fgets function has encounter an error. case #." << std::endl;
            }

            break;
        case 'v':
            /* v, vn, vt */
            switch (buf[1])
            {
            case '\0':
                /* vertex */
                /* eat up rest of line */
                if ( fgets(buf, sizeof(buf), file) == NULL)
                {
                    if (feof (file) )
                        std::cerr << "Error: MeshObjLoader: fgets function has encounter end of file. case \0." << std::endl;
                    else
                        std::cerr << "Error: MeshObjLoader: fgets function has encounter an error. case \0." << std::endl;
                }

                sscanf(buf, "%lf %lf %lf", &result[0], &result[1], &result[2]);
                my_positions.push_back(Vector3(result[0],result[1], result[2]));
                break;
            case 'n':
                /* normal */
                /* eat up rest of line */
                if ( fgets(buf, sizeof(buf), file) == NULL)
                {
                    if (feof (file) )
                        std::cerr << "Error: MeshObjLoader: fgets function has encounter end of file. case n." << std::endl;
                    else
                        std::cerr << "Error: MeshObjLoader: fgets function has encounter an error. case n." << std::endl;
                }


                sscanf(buf, "%lf %lf %lf", &result[0], &result[1], &result[2]);
                my_normals.push_back(Vector3(result[0],result[1], result[2]));
                break;
            case 't':
                /* texcoord */
                /* eat up rest of line */
                if ( fgets(buf, sizeof(buf), file) == NULL)
                {
                    if (feof (file) )
                        std::cerr << "Error: MeshObjLoader: fgets function has encounter end of file. case t." << std::endl;
                    else
                        std::cerr << "Error: MeshObjLoader: fgets function has encounter an error. case t." << std::endl;
                }

                sscanf (buf, "%lf %lf", &result[0], &result[1]);
                my_texCoords.push_back(Vector3(result[0],result[1], result[2]));
                break;
            default:
                printf("readObj : Unknown token \"%s\".\n", buf);
                exit(1);
                break;
            }
            break;
        case 'm':
        {
            if ( fgets(buf, sizeof(buf), file) == NULL)
            {
                if (feof (file) )
                    std::cerr << "Error: MeshObjLoader: fgets function has encounter end of file. case m." << std::endl;
                else
                    std::cerr << "Error: MeshObjLoader: fgets function has encounter an error. case m." << std::endl;
            }

            sscanf(buf, "%s %s", buf, buf);
            //mtllibname = strdup(buf);
            //fscanf(file, "%s", buf);
            std::string mtlfile = sofa::helper::system::SetDirectory::GetRelativeFromFile(buf, filename);
            //std::cerr << "Buf = " << buf << std::endl;
            //std::cerr << "Filename = " << filename << std::endl;
            this->readMTL(mtlfile.c_str(), my_materials);
        }
        break;
        case 'u':
        {
            /* eat up rest of line */
            if ( fgets(buf, sizeof(buf), file) == NULL)
            {
                if (feof (file) )
                    std::cerr << "Error: MeshObjLoader: fgets function has encounter end of file. case u." << std::endl;
                else
                    std::cerr << "Error: MeshObjLoader: fgets function has encounter an error. case u." << std::endl;
            }

            sscanf(buf, "%s", matName);
            helper::vector<MeshObjLoader::Material>::iterator it = my_materials.begin();
            helper::vector<MeshObjLoader::Material>::iterator itEnd = my_materials.end();
            for (; it != itEnd; it++)
            {
                if (it->name == matName)
                {
                    //  							std::cout << "Using material "<<it->name<<std::endl;
                    (*it).activated = true;
                    material = *it;
                }
            }
        }
        break;
        case 'g':
            /* group */
            /* eat up rest of line */
            if ( fgets(buf, sizeof(buf), file) == NULL)
            {
                if (feof (file) )
                    std::cerr << "Error: MeshObjLoader: fgets function has encounter end of file. case g." << std::endl;
                else
                    std::cerr << "Error: MeshObjLoader: fgets function has encounter an error. case g." << std::endl;
            }

            sscanf(buf, "%s", buf);
            break;
        case 'l': // for now we consider a line as a 2-vertices face
        case 'f':
            // face
            if ( fgets(buf, sizeof(buf), file) == NULL)
            {
                if (feof (file) )
                    std::cerr << "Error: MeshObjLoader: fgets function has encounter end of file. case f." << std::endl;
                else
                    std::cerr << "Error: MeshObjLoader: fgets function has encounter an error. case f." << std::endl;
            }
            token = strtok(buf, " ");

            nodes.clear();
            nIndices.clear();
            tIndices.clear();

            while(token!=NULL && token[0]>='0' && token[0]<='9')
            {
                face = token;
                for (int j = 0; j < 3; j++)
                {
                    vtn[j] = 0;
                    std::string::size_type pos = face.find('/');
                    tmp = face.substr(0, pos);
                    if (tmp != "")
                        vtn[j] = atoi(tmp.c_str()) - 1; // -1 because the numerotation begins at 1 and a vector begins at 0
                    if (pos == std::string::npos)
                        face = "";
                    else
                        face = face.substr(pos + 1);
                }

                nodes.push_back(vtn[0]);
                nIndices.push_back(vtn[1]);
                tIndices.push_back(vtn[2]);
                token = strtok(NULL, " ");
            }
            my_normalsList.push_back(nIndices);
            my_texturesList.push_back(tIndices);

            if (nodes.size() == 2) // Edge
            {
                if (nodes[0]<nodes[1])
                    my_edges.push_back (helper::fixed_array <unsigned int,2>(nodes[0], nodes[1]));
                else
                    my_edges.push_back (helper::fixed_array <unsigned int,2>(nodes[1], nodes[0]));
            }
            else if (nodes.size()==4) // Quad
            {
                my_quads.push_back (helper::fixed_array <unsigned int,4>(nodes[0], nodes[1], nodes[2], nodes[3]));
            }
            else // Triangularize
            {
                for (unsigned int j=2; j<nodes.size(); j++)
                    my_triangles.push_back (helper::fixed_array <unsigned int,3>(nodes[0], nodes[j-1], nodes[j]));
            }

            break;



        default:
            // eat up rest of line
            if ( fgets(buf, sizeof(buf), file) == NULL)
            {
                if (feof (file) )
                    std::cerr << "Error: MeshObjLoader: fgets function has encounter end of file. case default." << std::endl;
                else
                    std::cerr << "Error: MeshObjLoader: fgets function has encounter an error. case default." << std::endl;
            }
            break;
        }
    }
    // announce the model statistics
    // 	std::cout << " Vertices: " << vertices.size() << std::endl;
    // 	std::cout << " Normals: " << normals.size() << std::endl;
    // 	std::cout << " Texcoords: " << texCoords.size() << std::endl;
    // 	std::cout << " Triangles: " << facets.size() << std::endl;
//     if (my_positions.size()>0)
//     {
//       // compute bbox
//       Vector3 minBB = vertices[0];
//       Vector3 maxBB = vertices[0];
//       for (unsigned int i=1; i<vertices.size();i++)
//       {
// 	Vector3 p = vertices[i];
// 	for (int c=0;c<3;c++)
// 	{
// 	  if (minBB[c] > p[c])
// 	    minBB[c] = p[c];
// 	  if (maxBB[c] < p[c])
// 	    maxBB[c] = p[c];
// 	}
//       }

//       //std::cout << "BBox: <"<<minBB[0]<<','<<minBB[1]<<','<<minBB[2]<<">-<"<<maxBB[0]<<','<<maxBB[1]<<','<<maxBB[2]<<">\n";
//     }


    //Fill topology elements:

    positions.endEdit();
    edges.endEdit();
    triangles.endEdit();
    quads.endEdit();

    texCoords.endEdit();
    normals.endEdit();
    normalsList.endEdit();
    texturesList.endEdit();

    materials.endEdit();

    return true;
}



// -----------------------------------------------------
// readMTL: read a wavefront material library file
//
//    model - properly initialized GLMmodel structure
//    name  - name of the material library
// -----------------------------------------------------
bool MeshObjLoader::readMTL(const char* filename, helper::vector <Material>& materials)
{
    std::cout << "MeshObjLoader::readMTL" << std::endl;

    FILE* file;
    char buf[128];
    file = fopen(filename, "r");
    Material *mat = NULL;
    if (!file);//std::cerr << "readMTL() failed: can't open material file " << filename << std::endl;
    else
    {
        /* now, read in the data */
        while (fscanf(file, "%s", buf) != EOF)
        {

            switch (buf[0])
            {
            case '#':
                /* comment */
                /* eat up rest of line */
                if ( fgets(buf, sizeof(buf), file) == NULL)
                {
                    if (feof (file) )
                        std::cerr << "Error: MeshObjLoader: fgets function has encounter end of file. case #." << std::endl;
                    else
                        std::cerr << "Error: MeshObjLoader: fgets function has encounter an error. case #." << std::endl;
                }
                break;
            case 'n':
                /* newmtl */
                if (mat != NULL)
                {
                    materials.push_back(*mat);
                    delete mat;
                    mat = NULL;
                }
                mat = new Material();
                if ( fgets(buf, sizeof(buf), file) == NULL)
                {
                    if (feof (file) )
                        std::cerr << "Error: MeshObjLoader: fgets function has encounter end of file. case n." << std::endl;
                    else
                        std::cerr << "Error: MeshObjLoader: fgets function has encounter an error. case n." << std::endl;
                }

                sscanf(buf, "%s %s", buf, buf);
                mat->name = buf;
                break;
            case 'N':
                switch (buf[1])
                {
                case 'i':
                {
                    float optical_density;
                    if ( fscanf(file, "%f", &optical_density) == EOF )
                        std::cerr << "Error: MeshObjLoader: fscanf function has encounter an error. case N i." << std::endl;
                    break;
                }
                case 's':
                    if (fscanf(file, "%f", &mat->shininess) == EOF )
                        std::cerr << "Error: MeshObjLoader: fscanf function has encounter an error. case N s." << std::endl;
                    // wavefront shininess is from [0, 1000], so scale for OpenGL
                    //mat->shininess /= 1000.0;
                    //mat->shininess *= 128.0;
                    mat->useShininess = true;
                    break;
                default:
                    /* eat up rest of line */
                    if ( fgets(buf, sizeof(buf), file) == NULL)
                    {
                        if (feof (file) )
                            std::cerr << "Error: MeshObjLoader: fgets function has encounter end of file. case N." << std::endl;
                        else
                            std::cerr << "Error: MeshObjLoader: fgets function has encounter an error. case N." << std::endl;
                    }
                    break;
                }
                break;
            case 'K':
                switch (buf[1])
                {
                case 'd':
                    if ( fscanf(file, "%f %f %f", &mat->diffuse[0], &mat->diffuse[1], &mat->diffuse[2]) == EOF)
                        std::cerr << "Error: MeshObjLoader: fscanf function has encounter an error. case K d." << std::endl;
                    mat->useDiffuse = true;
                    /*std::cout << mat->name << " diffuse = "<<mat->diffuse[0]<<' '<<mat->diffuse[1]<<'*/ /*'<<mat->diffuse[2]<<std::endl;*/
                    break;
                case 's':
                    if ( fscanf(file, "%f %f %f", &mat->specular[0], &mat->specular[1], &mat->specular[2]) == EOF)
                        std::cerr << "Error: MeshObjLoader: fscanf function has encounter an error. case K s." << std::endl;
                    mat->useSpecular = true;
                    /*std::cout << mat->name << " specular = "<<mat->specular[0]<<' '<<mat->specular[1]<<'*/ /*'<<mat->specular[2]<<std::endl;*/
                    break;
                case 'a':
                    if ( fscanf(file, "%f %f %f", &mat->ambient[0], &mat->ambient[1], &mat->ambient[2]) == EOF)
                        std::cerr << "Error: MeshObjLoader: fscanf function has encounter an error. case K a." << std::endl;
                    mat->useAmbient = true;
                    /*std::cout << mat->name << " ambient = "<<mat->ambient[0]<<' '<<mat->ambient[1]<<'*/ /*'<<mat->ambient[2]<<std::endl;*/
                    break;
                default:
                    /* eat up rest of line */
                    if ( fgets(buf, sizeof(buf), file) == NULL)
                    {
                        if (feof (file) )
                            std::cerr << "Error: MeshObjLoader: fgets function has encounter end of file. case K." << std::endl;
                        else
                            std::cerr << "Error: MeshObjLoader: fgets function has encounter an error. case K." << std::endl;
                    }
                    break;
                }
                break;
            case 'd':
            case 'T':
                // transparency value
                if ( fscanf(file, "%f", &mat->diffuse[3]) == EOF)
                    std::cerr << "Error: MeshObjLoader: fscanf function has encounter an error. case T i." << std::endl;
                break;
            default:
                /* eat up rest of line */
                if ( fgets(buf, sizeof(buf), file) == NULL)
                {
                    if (feof (file) )
                        std::cerr << "Error: MeshObjLoader: fgets function has encounter end of file. case default." << std::endl;
                    else
                        std::cerr << "Error: MeshObjLoader: fgets function has encounter an error. case default." << std::endl;
                }
                break;
            }

        }
        fclose(file);
    }
    if (mat != NULL)
    {
        materials.push_back(*mat);
        delete mat;
        mat = NULL;
    }

    return true;
}



} // namespace loader

} // namespace component

} // namespace sofa

