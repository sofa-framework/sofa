#include "Sofa/Components/MeshTopologyLoader.h"
#include "Sofa/Components/Common/Vec.h"

#include <stdio.h>
#include <iostream>
#include <vector>

namespace Sofa
{

namespace Components
{

using namespace Common;

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
    char cmd[1024];
    FILE* file;

    if ((file = fopen(filename, "r")) == NULL)
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
        int npoints = 0;
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
        std::cout << cmd << std::endl;
        if (strncmp(cmd,"$ENDNOD",7))
        {
            fclose(file);
            return false;
        }

        readLine(cmd, sizeof(cmd), file);
        std::cout << cmd << std::endl;
        if (strncmp(cmd,"$ELM",4))
        {
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
                    addLine(nodes[0], nodes[1]);
                break;
            case 2: // Triangle
                if (nnodes == 3)
                    addTriangle(nodes[0], nodes[1], nodes[2]);
                break;
            case 3: // Quad
                if (nnodes == 4)
                    addQuad(nodes[0], nodes[1], nodes[2], nodes[3]);
                break;
            case 4: // Tetra
                if (nnodes == 4)
                    addTetra(nodes[0], nodes[1], nodes[2], nodes[3]);
                break;
            case 5: // Hexa
                if (nnodes == 8)
                    addCube(nodes[0], nodes[1], nodes[2], nodes[3],
                            nodes[4], nodes[5], nodes[6], nodes[7]);
                break;
            }
        }
        readLine(cmd, sizeof(cmd), file);
        std::cout << cmd << std::endl;
        if (strncmp(cmd,"$ENDELM",7))
        {
            fclose(file);
            return false;
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
            }
            else if (!strcmp(cmd,"triangle"))
            {
                int p1,p2,p3;
                fscanf(file, "%d %d %d\n",
                        &p1, &p2, &p3);
                addTriangle(p1, p2, p3);
            }
            else if (!strcmp(cmd,"quad"))
            {
                int p1,p2,p3,p4;
                fscanf(file, "%d %d %d %d\n",
                        &p1, &p2, &p3, &p4);
                addQuad(p1, p2, p3, p4);
            }
            else if (!strcmp(cmd,"tetra"))
            {
                int p1,p2,p3,p4;
                fscanf(file, "%d %d %d %d\n",
                        &p1, &p2, &p3, &p4);
                addTetra(p1, p2, p3, p4);
            }
            else if (!strcmp(cmd,"cube"))
            {
                int p1,p2,p3,p4,p5,p6,p7,p8;
                fscanf(file, "%d %d %d %d %d %d %d %d\n",
                        &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8);
                addCube(p1, p2, p3, p4, p5, p6, p7, p8);
            }
            else if (!strcmp(cmd,"point"))
            {
                double px,py,pz;
                fscanf(file, "%lf %lf %lf\n",
                        &px, &py, &pz);
                addPoint(px, py, pz);
            }
            else if (!strcmp(cmd,"v"))
            {
                double px,py,pz;
                fscanf(file, "%lf %lf %lf\n",
                        &px, &py, &pz);
                addPoint(px, py, pz);
            }
            else if (!strcmp(cmd,"f"))
            {
                int p1,p2,p3,p4=0;
                fscanf(file, "%d %d %d %d\n",
                        &p1, &p2, &p3, &p4);
                if (p4)
                    addQuad(p1-1, p2-1, p3-1, p4-1);
                else
                    addTriangle(p1-1, p2-1, p3-1);
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
            }
            else if (cmd[0] == '#')	// it's a comment
            {
                skipToEOL(file);
            }
            else		// it's an unknown keyword
            {
                printf("Unknown keyword: %s\n", cmd);
                skipToEOL(file);
            }
        }
    }
    std::cout << "Loading topology complete." << std::endl;
    fclose(file);
    return true;
}

} // namespace Components

} // namespace Sofa
