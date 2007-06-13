/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/helper/io/MeshTopologyLoader.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/defaulttype/Vec.h>

#include <stdio.h>
#include <iostream>
#include <vector>

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
    while ((ch = fgetc(f)) != EOF && ch != '\n');
}

static bool readLine(char* buf, int size, FILE* f)
{
    buf[0] = '\0';
    if (fgets(buf, size, f) == NULL)
        return false;
    if ((int)strlen(buf)==size-1 && buf[size-1] != '\n')
        skipToEOL(f);
    return true;
}

bool MeshTopologyLoader::load(const char *filename)
{
    std::string fname = filename;
    if (!sofa::helper::system::DataRepository.findFile(fname)) return false;

    char cmd[1024];
    FILE* file;
    int npoints = 0;
    int nlines = 0;
    int ntris = 0;
    int nquads = 0;
    int ntetras = 0;
    int ncubes = 0;

    if ((file = fopen(fname.c_str(), "r")) == NULL)
    {
        std::cout << "ERROR: cannot read file '" << filename << "'. Exiting..." << std::endl;
        return false;
    }
    // Check first line
    if (!readLine(cmd, sizeof(cmd), file))
    {
        fclose(file);
        return false;
    }
    std::cout << cmd << std::endl;
    if (!strncmp(cmd,"$NOD",4)) // Gmsh format
    {
        std::cout << "Loading Gmsh topology '" << filename << "'" << std::endl;
        fscanf(file, "%d\n", &npoints);
        setNbPoints(npoints);
        std::vector<int> pmap;
        for (int i=0; i<npoints; ++i)
        {
            int index = i;
            double x,y,z;
            fscanf(file, "%d %lf %lf %lf\n", &index, &x, &y, &z);
            addPoint(x, y, z);
            if ((int)pmap.size() <= index) pmap.resize(index+1);
            pmap[index] = i;
        }

        readLine(cmd, sizeof(cmd), file);
        //std::cout << cmd << std::endl;
        if (strncmp(cmd,"$ENDNOD",7))
        {
            std::cerr << "'$ENDNOD' expected, found '" << cmd << "'" << std::endl;
            fclose(file);
            return false;
        }

        readLine(cmd, sizeof(cmd), file);
        //std::cout << cmd << std::endl;
        if (strncmp(cmd,"$ELM",4))
        {
            std::cerr << "'$ELM' expected, found '" << cmd << "'" << std::endl;
            fclose(file);
            return false;
        }

        int nelems = 0;
        fscanf(file, "%d\n", &nelems);

        for (int i=0; i<nelems; ++i)
        {
            int index, etype, rphys, relem, nnodes;
            fscanf(file, "%d %d %d %d %d", &index, &etype, &rphys, &relem, &nnodes);
            std::vector<int> nodes;
            nodes.resize(nnodes);
            for (int n=0; n<nnodes; ++n)
            {
                int t = 0;
                fscanf(file, "%d",&t);
                nodes[n] = (((unsigned int)t)<pmap.size())?pmap[t]:0;
            }
            switch (etype)
            {
            case 1: // Line
                if (nnodes == 2)
                {
                    addLine(nodes[0], nodes[1]);
                    ++nlines;
                }
                break;
            case 2: // Triangle
                if (nnodes == 3)
                {
                    addTriangle(nodes[0], nodes[1], nodes[2]);
                    ++ntris;
                }
                break;
            case 3: // Quad
                if (nnodes == 4)
                {
                    addQuad(nodes[0], nodes[1], nodes[2], nodes[3]);
                    ++nquads;
                }
                break;
            case 4: // Tetra
                if (nnodes == 4)
                {
                    addTetra(nodes[0], nodes[1], nodes[2], nodes[3]);
                    ++ntetras;
                }
                break;
            case 5: // Hexa
                if (nnodes == 8)
                {
                    addCube(nodes[0], nodes[1], nodes[2], nodes[3],
                            nodes[4], nodes[5], nodes[6], nodes[7]);
                    ++ncubes;
                }
                break;
            }
            skipToEOL(file);
        }
        readLine(cmd, sizeof(cmd), file);
        std::cout << cmd << std::endl;
        if (strncmp(cmd,"$ENDELM",7))
        {
            std::cerr << "'$ENDELM' expected, found '" << cmd << "'" << std::endl;
            fclose(file);
            return false;
        }
    }
    else if (!strncmp(cmd,"Xsp",3))
    {
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

        int totalNumMasses;
        int totalNumSprings;

        //		skipToEOL(file);

        // then find out number of masses and springs
        if (fscanf(file, "%s", cmd) != EOF && !strcmp(cmd,"numm"))
        {
            fscanf(file, "%d", &totalNumMasses);
            setNbPoints(totalNumMasses);
            npoints=totalNumMasses;
        }
        if (fscanf(file, "%s", cmd) != EOF && !strcmp(cmd,"nums"))
        {
            fscanf(file, "%d", &totalNumSprings);
            setNbLines(totalNumSprings);
            nlines=totalNumSprings;
            //		setNumSprings(totalNumSprings);
        }

        //std::cout << "Model contains "<< totalNumMasses <<" masses and "<< totalNumSprings <<" springs"<<std::endl;



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
                addPoint(px,py,pz);
            }
            else if (!strcmp(cmd,"lspg"))	// linear springs connector
            {
                int	index;
                int m1,m2;
                double ks=0.0,kd=0.0,initpos=-1;
                // paul-------------------------------------
                double restx=0.0,resty=0.0,restz=0.0;
                if (vector_spring)
                    fscanf(file, "%d %d %d %lf %lf %lf %lf %lf %lf\n",
                            &index,&m1,&m2,&ks,&kd,&initpos, &restx,&resty,&restz);
                else
                    fscanf(file, "%d %d %d %lf %lf %lf\n",
                            &index,&m1,&m2,&ks,&kd,&initpos);
                --m1;
                --m2;

                addLine(m1,m2);
            }
            else if (!strcmp(cmd,"grav"))
            {
                double gx,gy,gz;
                fscanf(file, "%lf %lf %lf\n", &gx, &gy, &gz);
            }
            else if (!strcmp(cmd,"visc"))
            {
                double viscosity;
                fscanf(file, "%lf\n", &viscosity);
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
                printf("%s: Unknown MassSpring keyword: %s\n", filename, cmd);
                skipToEOL(file);
            }
        }
    }
    else
    {
        std::cout << "Loading mesh topology '" << filename << "'" << std::endl;
        while (fscanf(file, "%s", cmd) != EOF)
        {
            if (!strcmp(cmd,"line"))
            {
                int p1,p2;
                fscanf(file, "%d %d\n",
                        &p1, &p2);
                addLine(p1, p2);
                ++nlines;
            }
            else if (!strcmp(cmd,"triangle"))
            {
                int p1,p2,p3;
                fscanf(file, "%d %d %d\n",
                        &p1, &p2, &p3);
                addTriangle(p1, p2, p3);
                ++ntris;
            }
            else if (!strcmp(cmd,"quad"))
            {
                int p1,p2,p3,p4;
                fscanf(file, "%d %d %d %d\n",
                        &p1, &p2, &p3, &p4);
                addQuad(p1, p2, p3, p4);
                ++nquads;
            }
            else if (!strcmp(cmd,"tetra"))
            {
                int p1,p2,p3,p4;
                fscanf(file, "%d %d %d %d\n",
                        &p1, &p2, &p3, &p4);
                addTetra(p1, p2, p3, p4);
                ++ntetras;
            }
            else if (!strcmp(cmd,"cube"))
            {
                int p1,p2,p3,p4,p5,p6,p7,p8;
                fscanf(file, "%d %d %d %d %d %d %d %d\n",
                        &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8);
                addCube(p1, p2, p3, p4, p5, p6, p7, p8);
                ++ncubes;
            }
            else if (!strcmp(cmd,"point"))
            {
                double px,py,pz;
                fscanf(file, "%lf %lf %lf\n",
                        &px, &py, &pz);
                addPoint(px, py, pz);
                ++npoints;
            }
            else if (!strcmp(cmd,"v"))
            {
                double px,py,pz;
                fscanf(file, "%lf %lf %lf\n",
                        &px, &py, &pz);
                addPoint(px, py, pz);
                ++npoints;
            }
            else if (!strcmp(cmd,"f"))
            {
                int p1,p2,p3,p4=0;
                fscanf(file, "%d %d %d %d\n",
                        &p1, &p2, &p3, &p4);
                if (p4)
                {
                    addQuad(p1-1, p2-1, p3-1, p4-1);
                    ++nquads;
                }
                else
                {
                    addTriangle(p1-1, p2-1, p3-1);
                    ++ntris;
                }
            }
            else if (!strcmp(cmd,"mass"))
            {
                int index;
                char location;
                double px,py,pz,vx,vy,vz,mass=0.0,elastic=0.0;
                fscanf(file, "%d %c %lf %lf %lf %lf %lf %lf %lf %lf\n",
                        &index, &location,
                        &px, &py, &pz, &vx, &vy, &vz,
                        &mass, &elastic);
                addPoint(px, py, pz);
                ++npoints;
            }
            else if (!strcmp(cmd,"lspg"))
            {
                int	index;
                int m1,m2;
                double ks=0.0,kd=0.0,initpos=-1;
                fscanf(file, "%d %d %d %lf %lf %lf\n", &index,
                        &m1,&m2,&ks,&kd,&initpos);
                --m1;
                --m2;
                addLine(m1,m2);
                ++nlines;
            }
            else if (cmd[0] == '#')	// it's a comment
            {
                skipToEOL(file);
            }
            else		// it's an unknown keyword
            {
                printf("%s: Unknown Mesh keyword: %s\n", filename, cmd);
                skipToEOL(file);
                return false;
            }
        }
    }
    std::cout << "Loading topology complete:";
    if (npoints>0) std::cout << ' ' << npoints << " points";
    if (nlines>0)  std::cout << ' ' << nlines  << " lines";
    if (ntris>0)   std::cout << ' ' << ntris   << " triangles";
    if (nquads>0)  std::cout << ' ' << nquads  << " quads";
    if (ntetras>0) std::cout << ' ' << ntetras << " tetrahedra";
    if (ncubes>0)  std::cout << ' ' << ncubes  << " cubes";
    std::cout << std::endl;
    fclose(file);
    return true;
}

} // namespace io

} // namespace helper

} // namespace sofa

