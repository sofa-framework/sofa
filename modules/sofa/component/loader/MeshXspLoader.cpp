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

    sout << "Loading Xsp file: " << m_filename << sendl;

    FILE* file;
    char cmd[1024];
    bool fileRead = false;

    // -- Loading file
    const char* filename = m_filename.getFullPath().c_str();

    if ((file = fopen(filename, "r")) == NULL)
    {
        serr << "Error: MeshXspLoader: Cannot read file '" << m_filename << "'." << sendl;
        return false;
    }


    // -- Check first line.
    if (!readLine(cmd, sizeof(cmd), file))
    {
        serr << "Error: MeshXspLoader: Cannot read first line in file '" << m_filename << "'." << sendl;
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
        serr << "Error: MeshXspLoader: File '" << m_filename << "' finally appears not to be a Xsp file." << sendl;
        fclose(file);
        return false;

    }

    fclose (file);
    return fileRead;
}



bool MeshXspLoader::readXsp (FILE *file, bool vector_spring)
{
    sout << "Reading Xsp file: " << vector_spring << sendl;


    char cmd[1024];
    int npoints = 0;
    int nlines = 0;

    int totalNumMasses;
    int totalNumSprings;


    // then find out number of masses and springs
    if (fscanf(file, "%s", cmd) != EOF && !strcmp(cmd,"numm"))
    {
        if( fscanf(file, "%d", &totalNumMasses) == EOF)
            serr << "Error: MeshXspLoader: fscanf function can't read element for total number of mass." << sendl;
        npoints=totalNumMasses;
    }

    if (fscanf(file, "%s", cmd) != EOF && !strcmp(cmd,"nums"))
    {
        if( fscanf(file, "%d", &totalNumSprings) == EOF)
            serr << "Error: MeshXspLoader: fscanf function can't read element for total number of springs." << sendl;
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
            if (fscanf(file, "%d %c %lf %lf %lf %lf %lf %lf %lf %lf\n",
                    &index, &location,
                    &px, &py, &pz, &vx, &vy, &vz,
                    &mass, &elastic) == EOF)
                serr << "Error: MeshXspLoader: fscanf function can't read elements in main loop." << sendl;

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
            {
                if(fscanf(file, "%d %d %d %lf %lf %lf %lf %lf %lf\n",
                        &index,&m[0],&m[1],&ks,&kd,&initpos, &restx,&resty,&restz) == EOF)
                    serr << "Error: MeshXspLoader: fscanf function can't read elements for linear springs connectors (vector_spring case)." << sendl;
            }
            else
            {
                if (fscanf(file, "%d %d %d %lf %lf %lf\n",
                        &index,&m[0],&m[1],&ks,&kd,&initpos) == EOF)
                    serr << "Error: MeshXspLoader: fscanf function can't read element for linear springs connectors." << sendl;
            }

            --m[0];
            --m[1];

            addEdge(&my_edges, m);
        }
        else if (!strcmp(cmd,"grav"))
        {
            double gx,gy,gz;
            if ( fscanf(file, "%lf %lf %lf\n", &gx, &gy, &gz) == EOF)
                serr << "Error: MeshXspLoader: fscanf function can't read element for gravity." << sendl;
            my_gravity.push_back(Vector3(gx, gy, gz));
        }
        else if (!strcmp(cmd,"visc"))
        {
            double visc;
            if ( fscanf(file, "%lf\n", &visc) == EOF)
                serr << "Error: MeshXspLoader: fscanf function can't read element for viscosity." << sendl;
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
            serr << "Error: MeshXspLoader: Unknown MassSpring keyword '" << cmd << "'." << sendl;
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

