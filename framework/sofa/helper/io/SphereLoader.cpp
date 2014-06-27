/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/helper/io/SphereLoader.h>
#include <sofa/helper/system/FileRepository.h>

#include <stdio.h>
#include <iostream>
#include <cstdio>
#include <string.h>
#include <sstream>

namespace sofa
{

namespace helper
{

namespace io
{

static void skipToEOL(FILE* f)
{
    int	ch;
    while ((ch = fgetc(f)) != EOF && ch != '\n') ;
}

bool SphereLoader::load(const char *filename)
{
    std::string fname = filename;
    if (!sofa::helper::system::DataRepository.findFile(fname)) return false;

    char cmd[64];
    FILE* file;

    static const char* SPH_FORMAT = "sph 1.0";

    if ((file = fopen(fname.c_str(), "r")) == NULL)
    {
        std::cerr << "ERROR: cannot read file '" << filename << "'. Exiting..." << std::endl;
        return false;
    }

#ifndef NDEBUG
    std::cout << "Loading model'" << filename << "'" << std::endl;
#endif

    int totalNumSpheres=0;

    // Check first line
    if (fgets(cmd, 7, file) == NULL || !strcmp(cmd,SPH_FORMAT))
    {
        fclose(file);
        return false;
    }
    skipToEOL(file);

    std::ostringstream cmdScanFormat;
    cmdScanFormat << "%" << (sizeof(cmd) - 1) << "s";

    while (fscanf(file,cmdScanFormat.str().c_str(), cmd) != EOF)
    {
        if (!strcmp(cmd,"nums"))
        {
            if (fscanf(file, "%d", &totalNumSpheres) == EOF)
                std::cerr << "Error: SphereLoader: fscanf function has encountered an error." << std::endl;
            setNumSpheres(totalNumSpheres);
        }
        else if (!strcmp(cmd,"sphe"))
        {
            int index;
            double cx=0,cy=0,cz=0,r=1;
            if (fscanf(file, "%d %lf %lf %lf %lf\n",
                    &index, &cx, &cy, &cz, &r) == EOF)
                std::cerr << "Error: SphereLoader: fscanf function has encountered an error." << std::endl;
            addSphere((SReal)cx,(SReal)cy,(SReal)cz,(SReal)r);
            ++totalNumSpheres;
        }
        else if (cmd[0]=='#')
        {
            skipToEOL(file);
        }
        else			// it's an unknown keyword
        {
            printf("%s: Unknown Sphere keyword: %s\n", filename, cmd);
            skipToEOL(file);
        }
    }
// 	printf("Model contains %d spheres\n", totalNumSpheres);

    (void) fclose(file);

    return true;
}

} // namespace io

} // namespace helper

} // namespace sofa

