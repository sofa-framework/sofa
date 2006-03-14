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

bool MeshTopologyLoader::load(const char *filename)
{
    char cmd[64];
    FILE* file;

    if ((file = fopen(filename, "r")) == NULL)
    {
        std::cout << "ERROR: cannot read file '" << filename << "'. Exiting..." << std::endl;
        return false;
    }
    std::cout << "Loading mesh topology '" << filename << "'" << std::endl;
    // Check first line
    if (fgets(cmd, 7, file) == NULL || !strcmp(cmd,"Xsp 3.0"))
    {
        //fclose(file);
        //return false;
    }
    skipToEOL(file);
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
    fclose(file);
    return true;
}

} // namespace Components

} // namespace Sofa
