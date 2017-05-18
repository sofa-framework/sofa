/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#include <SofaGeneralLoader/SphereLoader.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/Locale.h>
#include <sofa/core/ObjectFactory.h>
#include <sstream>

using namespace sofa::core::loader;
using namespace sofa::defaulttype;
namespace sofa
{
namespace component
{
namespace loader
{

int SphereLoaderClass = core::RegisterObject("Loader for sphere model description files")
        .add<SphereLoader>();

SphereLoader::SphereLoader()
    :BaseLoader(),
     positions(initData(&positions,"position","Sphere centers")),
     radius(initData(&radius,"listRadius","Radius of each sphere")),
     d_scale(initData(&d_scale,"scale","Scale applied to sphere positions & radius")),
     d_translation(initData(&d_translation,"translation","Translation applied to sphere positions"))

{
    addAlias(&positions,"sphere_centers");
}



bool SphereLoader::load()
{
    // Make sure that fscanf() uses a dot '.' as the decimal separator.
    helper::system::TemporaryLocale locale(LC_NUMERIC, "C");

    const char* filename = m_filename.getFullPath().c_str();
    std::string fname = std::string(filename);

    if (!sofa::helper::system::DataRepository.findFile(fname)) return false;

    char cmd[64];
    FILE* file;

    static const char* SPH_FORMAT = "sph 1.0";

    if ((file = fopen(fname.c_str(), "r")) == NULL)
    {
        msg_error("SphereLoader") << "cannot read file '" << filename << "'. ";
        return false;
    }

    helper::vector<sofa::defaulttype::Vec<3,SReal> >& my_positions = *positions.beginEdit();
    helper::vector<SReal>& my_radius = *radius.beginEdit();

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
    while (fscanf(file, cmdScanFormat.str().c_str(), cmd ) != EOF)
    {
        if (!strcmp(cmd,"nums"))
        {
            int total;
            if (fscanf(file, "%d", &total) == EOF)
                msg_error("SphereLoader") << "Problem while loading. fscanf function has encountered an error." ;
            my_positions.reserve(total);
        }
        else if (!strcmp(cmd,"sphe"))
        {
            int index;
            double cx=0,cy=0,cz=0,r=1;
            if (fscanf(file, "%d %lf %lf %lf %lf\n",
                    &index, &cx, &cy, &cz, &r) == EOF)
                msg_error("SphereLoader") << "Problem while loading. fscanf function has encountered an error." ;
            my_positions.push_back(Vector3((SReal)cx,(SReal)cy,(SReal)cz));
            my_radius.push_back((SReal)r);
            ++totalNumSpheres;
        }
        else if (cmd[0]=='#')
        {
            skipToEOL(file);
        }
        else			// it's an unknown keyword
        {
            msg_warning("SphereLoader") << "Unknown Sphere keyword: "<< cmd << " in file '"<<filename<< "'" ;
            skipToEOL(file);
        }
    }

    (void) fclose(file);

    if (d_scale.isSet())
    {
        const SReal sx = d_scale.getValue()[0];
        const SReal sy = d_scale.getValue()[1];
        const SReal sz = d_scale.getValue()[2];

        for (unsigned int i = 0; i < my_radius.size(); i++)
        {
            my_radius[i] *= sx;
        }

        for (unsigned int i = 0; i < my_positions.size(); i++)
        {
            my_positions[i].x() *= sx;
            my_positions[i].y() *= sy;
            my_positions[i].z() *= sz;
        }
    }

    if (d_translation.isSet())
    {
        const SReal dx = d_translation.getValue()[0];
        const SReal dy = d_translation.getValue()[1];
        const SReal dz = d_translation.getValue()[2];

        for (unsigned int i = 0; i < my_positions.size(); i++)
        {
            my_positions[i].x() += dx;
            my_positions[i].y() += dy;
            my_positions[i].z() += dz;
        }
    }

    positions.endEdit();
    radius.endEdit();

    return true;
}

}//loader

}//component

}//sofa
