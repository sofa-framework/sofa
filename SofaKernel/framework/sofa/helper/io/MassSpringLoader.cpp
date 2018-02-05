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
#include <sofa/helper/io/MassSpringLoader.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/Locale.h>
#include <sofa/defaulttype/Vec.h>

#include <cstdio>
#include <iostream>
#include <vector>
#include <sstream>
#include <string>

namespace sofa
{

namespace helper
{

namespace io
{

using namespace sofa::defaulttype;

static void skipToEOL(FILE* f)
{
    int	ch;
    while ((ch = fgetc(f)) != EOF && ch != '\n') ;
}

bool MassSpringLoader::load(const char *filename)
{
    // Make sure that fscanf() uses a dot '.' as the decimal separator.
    helper::system::TemporaryLocale locale(LC_NUMERIC, "C");

    std::string fname = filename;
    if (!sofa::helper::system::DataRepository.findFile(fname)) return false;

    char cmd[64];
    FILE* file;

    if ((file = fopen(fname.c_str(), "r")) == NULL)
    {
        msg_error("MassSpringLoader") << "cannot read file '" << filename << "'" ;
        return false;
    }

    dmsg_info("MassSpringLoader") << "Loading model '" << filename << "'" ;

    int totalNumMasses=0;
    int totalNumSprings=0;
    // Check first line

    //if (fgets(cmd, 7, file) == NULL || !strcmp(cmd,"Xsp 4.0"))
    if (fgets(cmd, 7, file) == NULL)
    {
        fclose(file);
        return false;
    }

    // paul--------------------------
    float version = 0.0;
    sscanf(cmd, "Xsp %f", &version);

    bool  vector_spring = false;
    if (version == 3.0) vector_spring = false;
    else if (version == 4.0) vector_spring = true;
    else
    {
        fclose(file);
        return false;
    }
    // paul----------------------------

    skipToEOL(file);

    // then find out number of masses and springs
    std::ostringstream cmdScanFormat;
    cmdScanFormat << "%" << (sizeof(cmd) - 1) << "s";
    int massAndSpringSet=0;
    while (massAndSpringSet != 2 && fscanf(file, cmdScanFormat.str().c_str(), cmd) != EOF )
    {
        if (!strcmp(cmd,"numm"))
        {
            if (fscanf(file, "%d", &totalNumMasses) == EOF){
                msg_error("MassSpringLoader") << "fscanf function has encountered an error." ;
                setNumMasses(0);
                setNumSprings(0);
                return false;
            }
            setNumMasses(totalNumMasses);
            massAndSpringSet+=1;
        }else if(!strcmp(cmd,"nums")){
            if (fscanf(file, "%d", &totalNumSprings) == EOF){
                msg_error("MassSpringLoader") << "fscanf function has encountered an error." ;
                setNumMasses(0);
                setNumSprings(0);
                return false;
            }
            setNumSprings(totalNumSprings);
            massAndSpringSet+=1;

        }else {
            msg_warning("MassSpringLoader") << "Unable to process Xsp command '"<< cmd << "'" ;
            skipToEOL(file);
        }
    }

    if(massAndSpringSet!=2){
        msg_error("MassSpringLoader") << "Unable to load punctual masses from file. "
                                      << "Either the file is broken or is a file describing a rigid object." ;
        setNumMasses(0);
        setNumSprings(0);
        return false;
    }

    std::vector<Vector3> masses;
    if (totalNumMasses>0)
        masses.reserve(totalNumMasses);

    while (fscanf(file, cmdScanFormat.str().c_str(), cmd) != EOF)
    {
        if (!strcmp(cmd,"mass"))
        {
            int index;
            char location;
            double px,py,pz,vx,vy,vz,mass=0.0,elastic=0.0;
            bool fixed=false;
            if (fscanf(file, "%d %c %lf %lf %lf %lf %lf %lf %lf %lf\n",
                    &index, &location,
                    &px, &py, &pz, &vx, &vy, &vz,
                    &mass, &elastic) == EOF)
                msg_error("MassSpringLoader") << "fscanf function has encountered an error." ;

            bool surface = (location == 's');

            if (mass < 0)
            {
                // fixed point initialization
                mass = -mass;
                fixed = true;
            }
            addMass((SReal)px,(SReal)py,(SReal)pz,(SReal)vx,(SReal)vy,(SReal)vz,(SReal)mass,(SReal)elastic,fixed,surface);
            masses.push_back(Vector3((SReal)px,(SReal)py,(SReal)pz));
        }
        else if (!strcmp(cmd,"lspg"))	// linear springs connector
        {
            int	index;
            int m1,m2;
            double ks=0.0,kd=0.0,initpos=-1;
            // paul-------------------------------------
            double restx=0.0,resty=0.0,restz=0.0;
            if (vector_spring)
            {
                if (fscanf(file, "%d %d %d %lf %lf %lf %lf %lf %lf\n",
                        &index,&m1,&m2,&ks,&kd,&initpos, &restx,&resty,&restz) == EOF)
                    msg_error("MassSpringLoader") << "fscanf function has encountered an error." ;
            }
            else
            {
                if (fscanf(file, "%d %d %d %lf %lf %lf\n",
                        &index,&m1,&m2,&ks,&kd,&initpos) == EOF)
                    msg_error("MassSpringLoader") << "fscanf function has encountered an error." ;
            }

            --m1;
            --m2;
            if (!masses.empty() && ((unsigned int)m1>=masses.size() || (unsigned int)m2>=masses.size()))
            {
                msg_error("MassSpringLoader") << "incorrect mass indexes in spring "<<index<<" "<<m1+1<<" "<<m2+1;
            }
            else
            {
                if (initpos==-1 && !masses.empty())
                {
                    initpos = (masses[m1]-masses[m2]).norm();
                    ks/=initpos;
                    kd/=initpos;

#ifndef NDEBUG
                    dmsg_info("MassSpringLoader") << "spring "<<m1<<" "<<m2<<" "<<ks<<" "<<kd<<" "<<initpos ;
#endif
                }

                //paul-----------------------------------------
                if (vector_spring)
                    addVectorSpring(m1,m2,(SReal)ks,(SReal)kd,(SReal)initpos,(SReal)restx,(SReal)resty,(SReal)restz);
                else
                    addSpring(m1,m2,(SReal)ks,(SReal)kd,(SReal)initpos);
            }
        }
        else if (!strcmp(cmd,"grav"))
        {
            double gx,gy,gz;
            if (fscanf(file, "%lf %lf %lf\n", &gx, &gy, &gz) == EOF)
                msg_error("MassSpringLoader") << "fscanf function has encountered an error." ;
            setGravity((SReal)gx,(SReal)gy,(SReal)gz);
        }
        else if (!strcmp(cmd,"visc"))
        {
            double viscosity;
            if (fscanf(file, "%lf\n", &viscosity) == EOF)
                msg_error("MassSpringLoader") << "fscanf function has encountered an error." ;
            setViscosity((SReal)viscosity);
        }
        else if (!strcmp(cmd,"step"))
        {
            skipToEOL(file);
        }
        else if (!strcmp(cmd,"frce"))
        {
            skipToEOL(file);
        }
        else if (cmd[0] == '#')	// it's a comment
        {
            skipToEOL(file);
        }
        else		// it's an unknown keyword
        {
            msg_info("LassSpringLoader") << "Unknown MassSpring keyword: " << cmd << msgendl
                                         << "From file: " << filename ;
            skipToEOL(file);
        }
    }
    fclose(file);
    return true;
}

} // namespace io

} // namespace helper

} // namespace sofa

