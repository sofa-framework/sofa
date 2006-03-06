#include "MeshOBJ.h"
#include <stdlib.h>
#include <iostream>
#include <string>

namespace Sofa
{

namespace Components
{

using namespace Common;

Creator<Mesh::Factory,MeshOBJ> MeshOBJClass("obj");

void MeshOBJ::init (std::string filename)
{
    FILE *f = fopen(filename.c_str(), "r");
    if (f)
    {
        readOBJ (f);
        fclose(f);
    }
    else
        std::cerr << "File " << filename << " not found " << std::endl;
}

// -------------------------------------------------------------------------
// _glmFirstPass: first pass at a Wavefront OBJ file that gets all the
//  			  statistics of the model (such as #vertices, #normals, etc)
//
//  	model - properly initialized GLMmodel structure
//  	file  - (fopen'd) file descriptor
// --------------------------------------------------------------------------
void MeshOBJ::readOBJ (FILE* file)
{
    std::vector< std::vector<int> > vertNormTexIndices;
    std::vector<int>vIndices, nIndices, tIndices;
    int vtn[3];
    char buf[128], matName[1024];
    Vector3 result;
    Vector3 texCoord;
    Vector3 normal;
    const char *token;

    std::string face, tmp;

    while (fscanf(file, "%s", buf) != EOF)
    {
        switch (buf[0])
        {
        case '#':
            /* comment */
            /* eat up rest of line */
            fgets(buf, sizeof(buf), file);
            break;
        case 'v':
            /* v, vn, vt */
            switch (buf[1])
            {
            case '\0':
                /* vertex */
                /* eat up rest of line */
                fgets(buf, sizeof(buf), file);
                sscanf(buf, "%lf %lf %lf", &result[0], &result[1], &result[2]);
                vertices.push_back(result);
                break;
            case 'n':
                /* normal */
                /* eat up rest of line */
                fgets(buf, sizeof(buf), file);
                sscanf(buf, "%lf %lf %lf", &result[0], &result[1], &result[2]);
                normals.push_back(result);
                break;
            case 't':
                /* texcoord */
                /* eat up rest of line */
                fgets(buf, sizeof(buf), file);
                sscanf (buf, "%lf %lf", &result[0], &result[1]);
                texCoords.push_back(result);
                break;
            default:
                printf("readObj : Unknown token \"%s\".\n", buf);
                exit(1);
                break;
            }
            break;
        case 'm':
            fgets(buf, sizeof(buf), file);
            sscanf(buf, "%s %s", buf, buf);
            //mtllibname = strdup(buf);
            readMTL(buf);
            break;
        case 'u':
        {
            /* eat up rest of line */
            fgets(buf, sizeof(buf), file);
            sscanf(buf, "%s", matName);
            std::vector<Material>::iterator it = materials.begin();
            std::vector<Material>::iterator itEnd = materials.end();
            for (; it != itEnd; it++)
            {
                if (strcmp ((*it).name.c_str(), matName) == 0)
                {
                    material = (*it);
                    material.activated = true;
                }
            }
        }
        break;
        case 'g':
            /* group */
            /* eat up rest of line */
            fgets(buf, sizeof(buf), file);
            sscanf(buf, "%s", buf);
            break;
        case 'f':
            // face
            fgets(buf, sizeof(buf), file);
            token = strtok(buf, " ");

            vIndices.clear();
            nIndices.clear();
            tIndices.clear();
            vertNormTexIndices.clear();

            for (int i = 0; i < 3; i++)
            {
                face = token;
                for (int j = 0; j < 3; j++)
                {
                    vtn[j] = 0;

                    tmp = face.substr(0, face.find('/'));
                    if (tmp != "")
                        vtn[j] = atoi(tmp.c_str()) - 1; // -1 because the numerotation begins at 1 and a vector begins at 0
                    face = face.substr(face.find('/') + 1);
                }
                vIndices.push_back(vtn[0]);
                tIndices.push_back(vtn[1]);
                nIndices.push_back(vtn[2]);
                token = strtok(NULL, " ");
            }
            vertNormTexIndices.push_back (vIndices);
            vertNormTexIndices.push_back (nIndices);
            vertNormTexIndices.push_back (tIndices);
            facets.push_back(vertNormTexIndices);
            break;

        default:
            // eat up rest of line
            fgets(buf, sizeof(buf), file);
            break;
        }
    }

    // announce the model statistics
    std::cout << " Vertices: " << vertices.size() << std::endl;
    std::cout << " Normals: " << normals.size() << std::endl;
    std::cout << " Texcoords: " << texCoords.size() << std::endl;
    std::cout << " Triangles: " << facets.size() << std::endl;
}

// -----------------------------------------------------
// readMTL: read a wavefront material library file
//
//    model - properly initialized GLMmodel structure
//    name  - name of the material library
// -----------------------------------------------------
void MeshOBJ::readMTL(char* filename)
{
    FILE* file;
    char buf[128];
    /* open the file */
    char fName[1024];
    sprintf (fName, "Data/VisualModels/%s", filename);
    file = fopen(fName, "r");
    Material *mat = NULL;
    if (!file)
        std::cerr << "readMTL() failed: can't open material file " << fName << std::endl;
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
                fgets(buf, sizeof(buf), file);
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
                fgets(buf, sizeof(buf), file);
                sscanf(buf, "%s %s", buf, buf);
                mat->name = buf;
                break;
            case 'N':
                fscanf(file, "%lf", &mat->shininess);
                /* wavefront shininess is from [0, 1000], so scale for OpenGL */
                mat->shininess /= 1000.0;
                mat->shininess *= 128.0;
                mat->useShininess = true;
                break;
            case 'K':
                switch (buf[1])
                {
                case 'd':
                    fscanf(file, "%lf %lf %lf", &mat->diffuse[0], &mat->diffuse[1], &mat->diffuse[2]);
                    mat->useDiffuse = true;
                    break;
                case 's':
                    fscanf(file, "%lf %lf %lf", &mat->specular[0], &mat->specular[1], &mat->specular[2]);
                    mat->useSpecular = true;
                    break;
                case 'a':
                    fscanf(file, "%lf %lf %lf", &mat->ambient[0], &mat->ambient[1], &mat->ambient[2]);
                    mat->useAmbient = true;
                    break;
                default:
                    /* eat up rest of line */
                    fgets(buf, sizeof(buf), file);
                    break;
                }
                break;
            default:
                /* eat up rest of line */
                fgets(buf, sizeof(buf), file);
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
}

} // namespace Components

} // namespace Sofa
