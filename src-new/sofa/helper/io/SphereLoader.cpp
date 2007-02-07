#include "Sofa-old/Components/SphereLoader.h"

#include <stdio.h>
#include <iostream>

namespace Sofa
{

namespace Components
{

static void skipToEOL(FILE* f)
{
    int	ch;
    while ((ch = fgetc(f)) != EOF && ch != '\n');
}

bool SphereLoader::load(const char *filename)
{
    char cmd[64];
    FILE* file;

    static const char* SPH_FORMAT = "sph 1.0";

    if ((file = fopen(filename, "r")) == NULL)
    {
        std::cout << "ERROR: cannot read file '" << filename << "'. Exiting..." << std::endl;
        return false;
    }
    std::cout << "Loading model'" << filename << "'" << std::endl;

    int totalNumSpheres=0;

    // Check first line
    if (fgets(cmd, 7, file) == NULL || !strcmp(cmd,SPH_FORMAT))
    {
        fclose(file);
        return false;
    }
    skipToEOL(file);

    while (fscanf(file, "%s", cmd) != EOF)
    {
        if (!strcmp(cmd,"nums"))
        {
            fscanf(file, "%d", &totalNumSpheres);
            setNumSpheres(totalNumSpheres);
        }
        else if (!strcmp(cmd,"sphe"))
        {
            int index;
            double cx=0,cy=0,cz=0,r=1;
            fscanf(file, "%d %lf %lf %lf %lf\n",
                    &index, &cx, &cy, &cz, &r);
            addSphere(cx,cy,cz,r);
            ++totalNumSpheres;
        }
        else if (cmd[0]=='#')
        {
            skipToEOL(file);
        }
        else			// it's an unknown keyword
        {
            printf("%s: Unknown Sphere keyword: %s\n", filename, cmd);
            skipToEOL(file);
        }
    }
    printf("Model contains %d spheres\n", totalNumSpheres);

    (void) fclose(file);

    return true;
}

} // namespace Components

} // namespace Sofa
