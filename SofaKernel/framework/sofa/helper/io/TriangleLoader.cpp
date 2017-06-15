/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include <sofa/helper/io/TriangleLoader.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/Locale.h>

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <sofa/defaulttype/Vec.h>
#include <sstream>
#include <string.h>

namespace sofa
{

namespace helper
{

namespace io
{

using namespace sofa::defaulttype;

// static void skipToEOL(FILE* f)
// {
// 	int	ch;
// 	while ((ch = fgetc(f)) != EOF && ch != '\n');
// }

bool TriangleLoader::load(const char *filename)
{
    std::string fname = filename;
    if (!sofa::helper::system::DataRepository.findFile(fname)) return false;
    FILE*	file;

    /* open the file */
    file = fopen(fname.c_str(), "r");
    if (!file)
    {
        fprintf(stderr, "readOBJ() failed: can't open data file \"%s\".\n",
                filename);
        return false;
    }

    /* announce the model name */
    printf("Loading Triangle model: %s\n", filename);

    loadTriangles (file);
    fclose(file);

    return true;
}

void TriangleLoader::loadTriangles(FILE *file)
{
    // Make sure that fscanf() uses a dot '.' as the decimal separator.
    helper::system::TemporaryLocale locale(LC_NUMERIC, "C");

    assert (file != NULL);

    char buf[128];

    std::vector<Vector3> normals;
    Vector3 n;
    float x, y, z;
//	/* make a default group */
//
    
    std::ostringstream bufScanFormat;
    bufScanFormat << "%" << (sizeof(buf) - 1) << "s";

    while (fscanf(file, bufScanFormat.str().c_str(), buf) != EOF)
    {
        switch (buf[0])
        {
        case '#':
            /* comment */
            /* eat up rest of line */
            if ( fgets(buf, sizeof(buf), file) == NULL)
            {
                if (feof (file) )
                    std::cerr << "Error: TriangleLoader: fgets function has encountered end of file." << std::endl;
                else
                    std::cerr << "Error: TriangleLoader: fgets function has encountered an error." << std::endl;
            }
            break;
        case 'v':
            /* v, vn, vt */
            switch (buf[1])
            {
            case '\0':
                /* vertex */
                //p = new Vector3();
                if( fscanf(file, "%f %f %f", &x, &y, &z) == 3 )
                    addVertices(x, y, z);

                // fgets(buf, sizeof(buf), file);
                // numvertices++;
                else
                    std::cerr << "Error: TriangleLoader: fscanf function has encountered an error." << std::endl;
                break;
            case 'n':
                /* normal */
                /* eat up rest of line */
                if ( fgets(buf, sizeof(buf), file) == NULL)
                {
                    if (feof (file) )
                        std::cerr << "Error: TriangleLoader: fgets function has encountered end of file." << std::endl;
                    else
                        std::cerr << "Error: TriangleLoader: fgets function has encountered an error." << std::endl;
                }
                //fscanf(file, "%lf %lf %lf", &(n.x), &(n.y), &(n.z));
                //normals.push_back(n);
                break;
            case 't':
                /* texcoord */
                /* eat up rest of line */
                if ( fgets(buf, sizeof(buf), file) == NULL)
                {
                    if (feof (file) )
                        std::cerr << "Error: TriangleLoader: fgets function has encountered end of file." << std::endl;
                    else
                        std::cerr << "Error: TriangleLoader: fgets function has encountered an error." << std::endl;
                }
                break;
            default:
                printf("loadTriangles(): Unknown token \"%s\".\n", buf);
                exit(EXIT_FAILURE);
                break;
            }
            break;
        case 'm':
            /* eat up rest of line */
            if ( fgets(buf, sizeof(buf), file) == NULL)
            {
                if (feof (file) )
                    std::cerr << "Error: TriangleLoader: fgets function has encountered end of file." << std::endl;
                else
                    std::cerr << "Error: TriangleLoader: fgets function has encountered an error." << std::endl;
            }
            break;
        case 'u':
            /* eat up rest of line */
            if ( fgets(buf, sizeof(buf), file) == NULL)
            {
                if (feof (file) )
                    std::cerr << "Error: TriangleLoader: fgets function has encountered end of file." << std::endl;
                else
                    std::cerr << "Error: TriangleLoader: fgets function has encountered an error." << std::endl;
            }
            break;
        case 'g':
            /* eat up rest of line */
            if ( fgets(buf, sizeof(buf), file) == NULL)
            {
                if (feof (file) )
                    std::cerr << "Error: TriangleLoader: fgets function has encountered end of file." << std::endl;
                else
                    std::cerr << "Error: TriangleLoader: fgets function has encountered an error." << std::endl;
            }
            break;
        case 'f':
            /* face */
            if( fscanf(file, bufScanFormat.str().c_str(), buf) == 1 )
            {
                int n1, n2, n3, v1, v2, v3, t1, t2, t3;
                /* can be one of %d, %d//%d, %d/%d, %d/%d/%d %d//%d */
                if (strstr(buf, "//"))
                {
                    /* v//n */
                    sscanf(buf, "%d//%d", &v1, &n1);
                    if( fscanf(file, "%d//%d", &v2, &n2) == 2 && fscanf(file, "%d//%d", &v3, &n3) == 2 )
                    //Triangle *t = new Triangle(vertices[v1 - 1],
                    //						   vertices[v2 - 1],
                    //						   vertices[v3 - 1],
                    //						   velocityVertices[v1 - 1],
                    //						   velocityVertices[v2 - 1],
                    //						   velocityVertices[v3 - 1],
                    //						   (normals[n1] + normals[n2] + normals[n3]) / 3,
                    //						   this);
                    //elems.push_back(t);
                    addTriangle(v1 - 1, v2 - 1, v3 -1);
                    else
                        std::cerr << "Error: TriangleLoader: fscanf function has encountered an error." << std::endl;
                }
                else if (sscanf(buf, "%d/%d/%d", &v1, &t1, &n1) == 3)
                {
                    /* v/t/n */

                    if( fscanf(file, "%d/%d/%d", &v2, &t2, &n2) == 3 && fscanf(file, "%d/%d/%d", &v3, &t3, &n3) == 3 )
                    /* Triangle *t = new Triangle(vertices[v1 - 1],
                                               vertices[v2 - 1],
                                               vertices[v3 - 1],
                                               velocityVertices[v1 - 1],
                                               velocityVertices[v2 - 1],
                                               velocityVertices[v3 - 1],
                                               (normals[n1] + normals[n2] + normals[n3]) / 3,
                                               this);
                    elems.push_back(t); */
                    addTriangle(v1 - 1, v2 - 1, v3 -1);
                    else
                        std::cerr << "Error: TriangleLoader: fscanf function has encountered an error." << std::endl;
                }
                else if (sscanf(buf, "%d/%d", &v1, &t1) == 2)
                {
                    /* v/t */
                    if( fscanf(file, "%d/%d", &v2, &t2) == 2 && fscanf(file, "%d/%d", &v3, &t3) == 2 )
                    /* Triangle *t = new Triangle(vertices[v1 - 1],
                                               vertices[v2 - 1],
                                               vertices[v3 - 1],
                                               velocityVertices[v1 - 1],
                                               velocityVertices[v2 - 1],
                                               velocityVertices[v3 - 1],
                                               (normals[n1] + normals[n2] + normals[n3]) / 3,
                                               this);
                    elems.push_back(t); */
                    addTriangle(v1 - 1, v2 - 1, v3 -1);
                    else
                        std::cerr << "Error: TriangleLoader: fscanf function has encountered an error." << std::endl;
                }
                else
                {
                    /* v */
                    sscanf(buf, "%d", &v1);
                    if( fscanf(file, "%d", &v2) == 1 && fscanf(file, "%d", &v3) == 1 )

                    // compute the normal
                    /* Triangle *t = new Triangle(vertices[v1 - 1],
                                               vertices[v2 - 1],
                                               vertices[v3 - 1],
                                               velocityVertices[v1 - 1],
                                               velocityVertices[v2 - 1],
                                               velocityVertices[v3 - 1],
                                               this);
                    elems.push_back(t); */
                    addTriangle(v1 - 1, v2 - 1, v3 -1);
                    else
                        std::cerr << "Error: TriangleLoader: fscanf function has encountered an error." << std::endl;
                }
            }
            else
                std::cerr << "Error: TriangleLoader: fscanf function has encountered an error." << std::endl;
            break;

        default:
            /* eat up rest of line */
            if ( fgets(buf, sizeof(buf), file) == NULL)
            {
                if (feof (file) )
                    std::cerr << "Error: TriangleLoader: fgets function has encountered end of file." << std::endl;
                else
                    std::cerr << "Error: TriangleLoader: fgets function has encountered an error." << std::endl;
            }
            break;
        }
    }

    if (normals.empty())
    {
        // compute the normal for the triangles
        /*		std::vector<CollisionElement*>::iterator it = elems.begin()
        		std::vector<CollisionElement*>::iterator itEnd = elems.end();

        		for (; it != itEnd; it++)
        		{
        			Triangle *t = static_cast<Triangle*> (*it);
        			Vector3 u,v;
        			u = *(t->p2) - *(t->p1);
        			v = *(t->p3) - *(t->p1);

        			Vector3 uCrossV = u.Cross(v);

        			t->normal = uCrossV;
        		}*/
    }
}

} // namespace io

} // namespace helper

} // namespace sofa

