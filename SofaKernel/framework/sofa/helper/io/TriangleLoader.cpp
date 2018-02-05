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

/// This register the TriangleLoader object to the logging system so that we can use msg_*(this)
MSG_REGISTER_CLASS(sofa::helper::io::TriangleLoader, "TriangleLoader")

namespace sofa
{

namespace helper
{

namespace io
{

using namespace sofa::defaulttype;

bool TriangleLoader::load(const char *filename)
{
    std::string fname = filename;
    if (!sofa::helper::system::DataRepository.findFile(fname)) return false;
    FILE*	file;

    /* open the file */
    file = fopen(fname.c_str(), "r");
    if (!file)
    {
        msg_error() << "readOBJ() failed: can't open data file '"<<filename<<"'.";
        return false;
    }

    /* announce the model name */
    msg_info("TriangleLoader") << "Loading Triangle model: '"<<filename<<"'";

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

    /* make a default group */
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
                if (feof (file) ){
                    msg_error() << "fgets function has encountered end of file." ;
                }else{
                    msg_error() << "fgets function has encountered an error." ;
                }
            }
            break;
        case 'v':
            /* v, vn, vt */
            switch (buf[1])
            {
            case '\0':
                /* vertex */
                if( fscanf(file, "%f %f %f", &x, &y, &z) == 3 )
                    addVertices(x, y, z);

                else{
                    msg_error() << "Error: TriangleLoader: fscanf function has encountered an error." ;
                }break;
            case 'n':
                /* normal */
                /* eat up rest of line */
                if ( fgets(buf, sizeof(buf), file) == NULL)
                {
                    if (feof (file) ){
                        msg_error() << "fgets function has encountered end of file." ;
                    }else{
                        msg_error() << "fgets function has encountered an error." ;
                    }
                }
                break;
            case 't':
                /* texcoord */
                /* eat up rest of line */
                if ( fgets(buf, sizeof(buf), file) == NULL)
                {
                    if (feof (file) ){
                        msg_error() << "fgets function has encountered end of file." ;
                    }else{
                        msg_error() << "fgets function has encountered an error." ;
                    }
                }
                break;
            default:
                msg_fatal() << "Unknown token '" << buf << "'";
                exit(EXIT_FAILURE);
                break;
            }
            break;
        case 'm':
            /* eat up rest of line */
            if ( fgets(buf, sizeof(buf), file) == NULL)
            {
                if (feof (file) ){
                    msg_error() << "fgets function has encountered end of file."  ;
                }else{
                    msg_error() <<  "fgets function has encountered an error." ;
                }
            }
            break;
        case 'u':
            /* eat up rest of line */
            if ( fgets(buf, sizeof(buf), file) == NULL)
            {
                if (feof (file)){
                    msg_error() << "fgets function has encountered end of file." ;
                }else{
                    msg_error() << "fgets function has encountered an error." ;
                }
            }
            break;
        case 'g':
            /* eat up rest of line */
            if ( fgets(buf, sizeof(buf), file) == NULL)
            {
                if (feof (file) ){
                    msg_error() << "fgets function has encountered end of file." ;
                }else {
                    msg_error() << "fgets function has encountered an error." ;
                }
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
                    addTriangle(v1 - 1, v2 - 1, v3 -1);
                    else{
                        msg_error() << "fscanf function has encountered an error." ;
                    }
                }
                else if (sscanf(buf, "%d/%d/%d", &v1, &t1, &n1) == 3)
                {
                    /* v/t/n */

                    if( fscanf(file, "%d/%d/%d", &v2, &t2, &n2) == 3 && fscanf(file, "%d/%d/%d", &v3, &t3, &n3) == 3 )
                    addTriangle(v1 - 1, v2 - 1, v3 -1);
                    else{
                        msg_error() << "fscanf function has encountered an error." ;
                    }
                }
                else if (sscanf(buf, "%d/%d", &v1, &t1) == 2)
                {
                    /* v/t */
                    if( fscanf(file, "%d/%d", &v2, &t2) == 2 && fscanf(file, "%d/%d", &v3, &t3) == 2 )
                    addTriangle(v1 - 1, v2 - 1, v3 -1);
                    else{
                        msg_error() << "fscanf function has encountered an error." ;
                    }
                }
                else
                {
                    /* v */
                    sscanf(buf, "%d", &v1);
                    if( fscanf(file, "%d", &v2) == 1 && fscanf(file, "%d", &v3) == 1 )

                    // compute the normal
                    addTriangle(v1 - 1, v2 - 1, v3 -1);
                    else{
                        msg_error() << "fscanf function has encountered an error." ;
                    }
                }
            }
            else{
                msg_error() << "fscanf function has encountered an error." ;
            }
            break;

        default:
            /* eat up rest of line */
            if ( fgets(buf, sizeof(buf), file) == NULL)
            {
                if (feof (file) ){
                    msg_error() <<  "fgets function has encountered end of file." ;
                }else{
                    msg_error() <<  "fgets function has encountered an error." ;
                }
            }
            break;
        }
    }

    if (normals.empty())
    {
        // compute the normal for the triangles ?
    }
}

} // namespace io

} // namespace helper

} // namespace sofa

