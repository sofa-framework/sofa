/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/helper/io/SphereLoader.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/Locale.h>

#include <cstdio>
#include <iostream>
#include <cstdio>
#include <string>
#include <sstream>

/// This allow MeshTrian to interact with the messaging system.
MSG_REGISTER_CLASS(sofa::helper::io::SphereLoader, "SphereLoader")

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
    /// Make sure that fscanf() uses a dot '.' as the decimal separator.
    helper::system::TemporaryLocale locale(LC_NUMERIC, "C");

    std::string fname = filename;
    if (!sofa::helper::system::DataRepository.findFile(fname)) return false;

    char cmd[64];
    FILE* file;

    static const char* SPH_FORMAT = "sph 1.0";

    if ((file = fopen(fname.c_str(), "r")) == NULL)
    {
        msg_error() << "ERROR: cannot read file '" << filename << "'. (Aborting)";
        return false;
    }

    int totalNumSpheres=0;

    /// Check first line
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
                msg_error() << "fscanf function has encountered an error." ;
            setNumSpheres(totalNumSpheres);
        }
        else if (!strcmp(cmd,"sphe"))
        {
            int index;
            double cx=0,cy=0,cz=0,r=1;
            if (fscanf(file, "%d %lf %lf %lf %lf\n",
                    &index, &cx, &cy, &cz, &r) == EOF)
                msg_error() << "fscanf function has encountered an error." ;
            addSphere((SReal)cx,(SReal)cy,(SReal)cz,(SReal)r);
            ++totalNumSpheres;
        }
        else if (cmd[0]=='#')
        {
            skipToEOL(file);
        }
        else			// it's an unknown keyword
        {
            msg_info() << "'"<< filename << "' unknown Sphere keyword: " << cmd ;
            skipToEOL(file);
        }
    }

    (void) fclose(file);

    return true;
}

} // namespace io

} // namespace helper

} // namespace sofa

