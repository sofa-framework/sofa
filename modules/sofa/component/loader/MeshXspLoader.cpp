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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/loader/MeshXspLoader.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace loader
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(MeshXspLoader)

int MeshXspLoaderClass = core::RegisterObject("Specific mesh loader for Xsp file format.")
        .add< MeshXspLoader >()
        ;

MeshXspLoader::MeshXspLoader() : MeshLoader()
    , gravity(initData(&gravity,"gravity","Gravity coordinates loaded in this mesh."))
    , viscosity(initData(&viscosity,"viscosity","viscosity values loaded in this mesh."))
{
    gravity.setPersistent(false);
    viscosity.setPersistent(false);
}


bool MeshXspLoader::load()
{

    std::cout << "Loading Xsp file: " << m_filename << std::endl;

    FILE* file;
    char cmd[1024];
    bool fileRead = false;

    // -- Loading file
    const char* filename = m_filename.getFullPath().c_str();
    if ((file = fopen(filename, "r")) == NULL)
    {
        std::cerr << "Error: MeshXspLoader: Cannot read file '" << m_filename << "'." << std::endl;
        return false;
    }


    // -- Check first line.
    if (!readLine(cmd, sizeof(cmd), file))
    {
        std::cerr << "Error: MeshXspLoader: Cannot read first line in file '" << m_filename << "'." << std::endl;
        fclose(file);
        return false;
    }


    // -- Reading file version
    if (!strncmp(cmd,"Xsp",3))
    {
        float version = 0.0f;
        sscanf(cmd, "Xsp %f", &version);

        if (version == 3.0)
            fileRead = readXsp(file, false);
        else if (version == 4.0)
            fileRead = readXsp(file, true);
    }
    else
    {
        std::cerr << "Error: MeshXspLoader: File '" << m_filename << "' finally appears not to be a Xsp file." << std::endl;
        fclose(file);
        return false;

    }

    fclose (file);
    return fileRead;
}



bool MeshXspLoader::readXsp (FILE *file, bool vector_spring)
{
    std::cout << "Reading Xsp file: " << vector_spring << std::endl;


    char cmd[1024];
    int npoints = 0;
    int nlines = 0;

    int totalNumMasses;
    int totalNumSprings;


    // then find out number of masses and springs
    if (fscanf(file, "%s", cmd) != EOF && !strcmp(cmd,"numm"))
    {
        fscanf(file, "%d", &totalNumMasses);
        npoints=totalNumMasses;
    }

    if (fscanf(file, "%s", cmd) != EOF && !strcmp(cmd,"nums"))
    {
        fscanf(file, "%d", &totalNumSprings);
        nlines=totalNumSprings;
    }

    helper::vector<sofa::defaulttype::Vector3>& my_positions = *(positions.beginEdit());
    helper::vector<helper::fixed_array <unsigned int,2> >& my_edges = *(edges.beginEdit());

    helper::vector <Vector3>& my_gravity = *(gravity.beginEdit());
    helper::vector <double>& my_viscosity = *(viscosity.beginEdit());


    while (fscanf(file, "%s", cmd) != EOF)
    {
        if (!strcmp(cmd,"mass"))
        {
            int index;
            char location;
            double px,py,pz,vx,vy,vz,mass=0.0,elastic=0.0;
            bool fixed=false;
            fscanf(file, "%d %c %lf %lf %lf %lf %lf %lf %lf %lf\n",
                    &index, &location,
                    &px, &py, &pz, &vx, &vy, &vz,
                    &mass, &elastic);

            if (mass < 0)
            {
                // fixed point initialization
                mass = -mass;
                fixed = true;
            }

            // Adding positions
            my_positions.push_back(Vector3(px, py, pz));
        }
        else if (!strcmp(cmd,"lspg"))	// linear springs connector
        {
            int index;
            helper::fixed_array <unsigned int,2> m;
            double ks=0.0,kd=0.0,initpos=-1;
            double restx=0.0,resty=0.0,restz=0.0;
            if (vector_spring)
                fscanf(file, "%d %d %d %lf %lf %lf %lf %lf %lf\n",
                        &index,&m[0],&m[1],&ks,&kd,&initpos, &restx,&resty,&restz);
            else
                fscanf(file, "%d %d %d %lf %lf %lf\n",
                        &index,&m[0],&m[1],&ks,&kd,&initpos);
            --m[0];
            --m[1];

            my_edges.push_back (m);
        }
        else if (!strcmp(cmd,"grav"))
        {
            double gx,gy,gz;
            fscanf(file, "%lf %lf %lf\n", &gx, &gy, &gz);
            my_gravity.push_back(Vector3(gx, gy, gz));
        }
        else if (!strcmp(cmd,"visc"))
        {
            double visc;
            fscanf(file, "%lf\n", &visc);
            my_viscosity.push_back (visc);
        }
        else if (!strcmp(cmd,"step"))
        {
            //fscanf(file, "%lf\n", &(MSparams.default_dt));
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
            std::cerr << "Error: MeshXspLoader: Unknown MassSpring keyword '" << cmd << "'." << std::endl;
            skipToEOL(file);
            fclose(file);
            positions.endEdit();
            edges.endEdit();
            gravity.endEdit();
            viscosity.endEdit();

            return false;
        }
    }


    positions.endEdit();
    edges.endEdit();
    gravity.endEdit();
    viscosity.endEdit();

    fclose(file);
    return true;
}

} // namespace loader

} // namespace component

} // namespace sofa

