/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/helper/io/XspLoader.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/Locale.h>
#include <sofa/type/Vec.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/logging/Messaging.h>
using sofa::type::Vec3;

#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <cmath>

namespace sofa
{

namespace helper
{

namespace io
{

XspLoaderDataHook::~XspLoaderDataHook(){}

bool XspLoader::ReadXspContent(std::ifstream &file,
                               bool hasVectorSpring,
                               XspLoaderDataHook& data)
{
    bool hasTotalNumMasses=false;
    bool hasNumSprings=false;
    size_t numTotalMasses=0;
    size_t numTotalSpring=0;

    /// Temporarily stores the masses while loading for the initpos calculs in the
    /// 'lspg' command.
    std::vector<Vec3> masses;
    while (!file.eof())
    {
        std::string cmd {""};
        file  >> cmd;
        if(!file.good() && file.eof())
            break;

        if (cmd == "numm")
        {
            /// In case the file contains multiple totalMass we print a warning.
            if(hasTotalNumMasses)
            {
                msg_error("XspLoader") << "The file contains multiple 'numm' commands which is invalid.";
                return false;
            }
            file >> numTotalMasses;
            data.setNumMasses(numTotalMasses);
            hasTotalNumMasses=true;
        }
        else if (cmd=="nums")
        {
            /// In case the file contains multiple totalMass we print a warning.
            if(hasNumSprings)
            {
                msg_error("XspLoader") << "The file contains multiple 'nums' commands which is invalid.";
                return false;
            }
            file >> numTotalSpring;
            data.setNumSprings(numTotalSpring);
            hasNumSprings=true;
        }
        else if (cmd=="mass")
        {
            int index;
            char location;

            double px,py,pz,vx,vy,vz,mass=0.0,elastic=0.0;
            file >> index >> location >> px >> py >> pz >> vx >> vy >> vz >> mass >> elastic;
            if( !file.good() )
            {
                msg_error("XspLoader") << "Error while reading 'mass' command.";
                return false;
            }
            const bool isASurfacePoint = (location == 's');
            bool isAFixedPoint = false;
            if (mass < 0)
            {
                // fixed point initialization
                mass = -mass;
                isAFixedPoint = true;
            }

            /// The massses are needed because of springs.
            masses.push_back(Vec3(px,py,pz));
            data.addMass(px,py,pz,vx,vy,vz,mass, elastic, isAFixedPoint, isASurfacePoint);
        }
        else if (cmd=="lspg")	// linear springs connector
        {
            int	index;
            size_t m0,m1;
            double ks=0.0, kd=0.0, initpos=-1;
            double restx=0.0,resty=0.0,restz=0.0;
            if (hasVectorSpring)
            {
                file >> index >> m0 >> m1 >> ks >> kd >> initpos >> restx >> resty >> restz;
            }
            else
            {
                file >> index >> m0 >> m1 >> ks >> kd >> initpos;
            }
            if( !file.good() )
            {
                msg_error("XspLoader") << "Error while reading 'lspg' command.";
                return false;
            }

            --m0;
            --m1;

            if ( m0 >= masses.size() || m1 >= masses.size()  )
            {
                msg_error("XspLoader") << "incorrect mass indexes in spring "<<index<<" "<<m0+1<<" "<<m1+1;
                return false;
            }

            if (isEqual(initpos,-1.0))
            {
                initpos = (masses[m0]-masses[m1]).norm();
                ks/=initpos;
                kd/=initpos;
            }

            data.addVectorSpring(m0,m1,ks,kd,initpos,restx,resty,restz);
        }
        else if (cmd == "grav")
        {
            double gx,gy,gz;
            file >> gx >> gy >> gz;
            data.setGravity(gx,gy,gz);
        }
        else if (cmd == "visc")
        {
            double visc;
            file >> visc;
            data.setViscosity(visc);
        }
        else if (cmd == "step")
        {
            /// We ignore the line
            std::getline(file, cmd);
        }
        else if (cmd == "frce")
        {
            /// We ignore the line
            std::getline(file, cmd);
        }
        else if (cmd.at(0) == '#')	// it's a comment
        {            
            /// We ignore the line
            std::getline(file, cmd);
        }
        else /// it's an unknown keyword
        {
            msg_error("XspLoader") << "Unknown MassSpring keyword '" << cmd << "'.";
            return false;
        }
    }

    return true;
}


bool XspLoader::Load(const std::string& filename,
                        XspLoaderDataHook& data)
{
    /// Make sure that fscanf() uses a dot '.' as the decimal separator.
    helper::system::TemporaryLocale locale(LC_NUMERIC, "C");

    /// We need this fname because the findFile function is modifying the argument
    /// to store the full path.
    std::string fullFilePath = filename;
    if (!sofa::helper::system::DataRepository.findFile(fullFilePath)) return false;

    std::ifstream file(fullFilePath);
    if (!file.good())
    {
        msg_error("XspLoader") << "Cannot read file '" << fullFilePath << "'.";
        return false;
    }

    /// -- Reading the Xsp header to check it is a real xsp file.
    std::string cmd;
    file >> cmd;
    if(cmd!="Xsp")
    {
        msg_error("XspLoader") << "File '" << fullFilePath << "' finally appears not to be a Xsp file.";
        file.close();
        return false;
    }

    /// -- Reading file version
    float version = 0.0f;
    file >> version;

    bool isOk = false;
    if (isEqual(version, 3.0f))
        isOk = ReadXspContent(file, false, data);
    else if (isEqual(version, 4.0f))
        isOk = ReadXspContent(file, true, data);
    else
        msg_error("XspLoader") <<"Xsp version '"<<version<<"' is not supported yet.";

    if(!isOk)
        msg_error("XspLoader") << "Unable to read '" << fullFilePath << "'.";

    data.finalizeLoading(isOk);

    file.close();
    return isOk;
}

} // namespace io

} // namespace helper

} // namespace sofa

