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
#include <sofa/helper/io/File.h>
#include <sofa/helper/io/MeshXsp.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/Locale.h>
#include <sofa/helper/logging/Messaging.h>
#include <istream>
#include <fstream>

namespace sofa
{

namespace helper
{

namespace io
{

using namespace sofa::defaulttype;
using namespace sofa::core::loader;

SOFA_DECL_CLASS(MeshXsp)

Creator<Mesh::FactoryMesh, MeshXsp> MeshXspClass("xsp");

void MeshXsp::init (std::string filename)
{
    if (!sofa::helper::system::DataRepository.findFile(filename))
    {
        msg_error("MeshXsp") << "File " << filename << " not found.";
        return;
    }
    loaderType = "xsp";

    std::ifstream file(filename);
    if (!file.good()) return;

    int gmshFormat = 0;

    std::string cmd;
    file >> cmd;

    if (cmd == "Xsp")
    {
        float version = 0.0f;
        file >> version;

        if (version == 3.0)
            readXsp(file, false);
        else if (version == 4.0)
            readXsp(file, true);
    }

    file.close();
}

bool MeshXsp::readXsp(std::ifstream &file, bool vector_spring)
{
    std::string cmd;
    file >> cmd;

    // then find out number of masses and springs, not used.
    if (cmd == "numm")
    {
        int totalNumMasses = 0;
        file >> totalNumMasses;
    }

    if (cmd == "nums")
    {
        int totalNumSprings = 0;
        file >> totalNumSprings;
    }


    while (!file.eof())
    {
        file >> cmd;
        if (cmd == "mass")
        {
            int index;
            char location;
            double px, py, pz, vx, vy, vz, mass = 0.0, elastic = 0.0;
            file >> index >> location >> px >> py >> pz >> vx >> vy >> vz >> mass >> elastic;
            m_vertices.push_back(Vector3(px, py, pz));
        }
        else if (cmd == "lspg")	// linear springs connector
        {
            int	index;
            Topology::Edge m;
            double ks = 0.0, kd = 0.0, initpos = -1;

            if (vector_spring)
            {
                double restx = 0.0, resty = 0.0, restz = 0.0;
                file >> index >> m[0] >> m[1] >> ks >> kd >> initpos >> restx >> resty >> restz;
            }
            else
                file >> index >> m[0] >> m[1] >> ks >> kd >> initpos;
            --m[0];
            --m[1];

            m_edges.push_back(m);
        }
        else if (cmd == "grav")
        {
            double gx, gy, gz;
            file >> gx >> gy >> gz;
            // my_gravity.push_back(Vector3(gx, gy, gz)); //TODO: 2018-04-06 (unify loader api): This buffer is missing in the old loaders.
        }
        else if (cmd == "visc")
        {
            double viscosity;
            file >> viscosity;
            // my_viscosity.push_back(Vector3(gx, gy, gz)); //TODO: 2018-04-06 (unify loader api): This buffer is missing in the old loaders.
        }
        else if (cmd == "step")
        {
        }
        else if (cmd == "frce")
        {
        }
        else if (cmd[0] == '#')	// it's a comment
        {
        }
        else		// it's an unknown keyword
        {
            msg_error("MeshXsp") << "Unknown MassSpring keyword:";
            return false;
        }
    }

}

} // namespace io

} // namespace helper

} // namespace sofa

