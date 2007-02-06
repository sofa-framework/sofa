#include "Sofa/Components/MassSpringLoader.h"
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

bool MassSpringLoader::load(const char *filename)
{
    char cmd[64];
    FILE* file;

    if ((file = fopen(filename, "r")) == NULL)
    {
        std::cout << "ERROR: cannot read file '" << filename << "'. Exiting..." << std::endl;
        return false;
    }
    std::cout << "Loading model '" << filename << "'" << std::endl;
    int totalNumMasses=0;
    int totalNumSprings=0;
    // Check first line
    if (fgets(cmd, 7, file) == NULL || !strcmp(cmd,"Xsp 3.0"))
    {
        fclose(file);
        return false;
    }
    skipToEOL(file);

    // then find out number of masses and springs
    if (fscanf(file, "%s", cmd) != EOF && !strcmp(cmd,"numm"))
    {
        fscanf(file, "%d", &totalNumMasses);
        setNumMasses(totalNumMasses);
    }
    if (fscanf(file, "%s", cmd) != EOF && !strcmp(cmd,"nums"))
    {
        fscanf(file, "%d", &totalNumSprings);
        setNumSprings(totalNumSprings);
    }

    std::cout << "Model contains "<< totalNumMasses <<" masses and "<< totalNumSprings <<" springs"<<std::endl;

    std::vector<Vec3d> masses;
    if (totalNumMasses>0)
        masses.reserve(totalNumMasses);

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
            bool surface = (location == 's');

            if (mass < 0)
            {
                // fixed point initialization
                mass = -mass;
                fixed = true;
            }
            addMass(px,py,pz,vx,vy,vz,mass,elastic,fixed,surface);
            masses.push_back(Vec3d(px,py,pz));
        }
        else if (!strcmp(cmd,"lspg"))	// linear springs connector
        {
            int	index;
            int m1,m2;
            double ks=0.0,kd=0.0,initpos=-1;
            fscanf(file, "%d %d %d %lf %lf %lf\n", &index,
                    &m1,&m2,&ks,&kd,&initpos);
            --m1;
            --m2;
            if (!masses.empty() && ((unsigned int)m1>=masses.size() || (unsigned int)m2>=masses.size()))
            {
                std::cerr << "ERROR: incorrect mass indexes in spring "<<index<<" "<<m1+1<<" "<<m2+1<<std::endl;
            }
            else
            {
                if (initpos==-1 && !masses.empty())
                {
                    initpos = (masses[m1]-masses[m2]).norm();
                    ks/=initpos;
                    kd/=initpos;
                    //std::cout << "spring "<<m1<<" "<<m2<<" "<<ks<<" "<<kd<<" "<<initpos<<"\n";
                }
                addSpring(m1,m2,ks,kd,initpos);
            }
        }
        else if (!strcmp(cmd,"grav"))
        {
            double gx,gy,gz;
            fscanf(file, "%lf %lf %lf\n", &gx, &gy, &gz);
            setGravity(gx,gy,gz);
        }
        else if (!strcmp(cmd,"visc"))
        {
            double viscosity;
            fscanf(file, "%lf\n", &viscosity);
            setViscosity(viscosity);
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
    fclose(file);
    return true;
}

} // namespace Components

} // namespace Sofa
