#include <sofa/component/loader/SphereLoader.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/core/ObjectFactory.h>

using namespace sofa::core::loader;
using namespace sofa::defaulttype;
namespace sofa
{
namespace component
{
namespace loader
{

int SphereLoaderClass = core::RegisterObject("Loader for sphere model description files")
        .add<SphereLoader>();

SphereLoader::SphereLoader()
    :BaseLoader(),
     positions(initData(&positions,"position","Sphere centers")),
     radius(initData(&radius,"listRadius","Radius of each sphere"))
{
    addAlias(&positions,"sphere_centers");
}



bool SphereLoader::load()
{
    const char* filename = m_filename.getFullPath().c_str();
    std::string fname = std::string(filename);

    if (!sofa::helper::system::DataRepository.findFile(fname)) return false;

    char cmd[64];
    FILE* file;

    static const char* SPH_FORMAT = "sph 1.0";

    if ((file = fopen(fname.c_str(), "r")) == NULL)
    {
        std::cout << "ERROR: cannot read file '" << filename << "'. Exiting..." << std::endl;
        return false;
    }

    helper::vector<sofa::defaulttype::Vec<3,SReal> >& my_positions = *positions.beginEdit();
    helper::vector<SReal>& my_radius = *radius.beginEdit();
// 	std::cout << "Loading model'" << filename << "'" << std::endl;

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
            int total;
            if (fscanf(file, "%d", &total) == EOF)
                std::cerr << "Error: SphereLoader: fscanf function has encountered an error." << std::endl;
            my_positions.reserve(total);
        }
        else if (!strcmp(cmd,"sphe"))
        {
            int index;
            double cx=0,cy=0,cz=0,r=1;
            if (fscanf(file, "%d %lf %lf %lf %lf\n",
                    &index, &cx, &cy, &cz, &r) == EOF)
                std::cerr << "Error: SphereLoader: fscanf function has encountered an error." << std::endl;
            my_positions.push_back(Vector3((SReal)cx,(SReal)cy,(SReal)cz));
            my_radius.push_back((SReal)r);
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
// 	printf("Model contains %d spheres\n", totalNumSpheres);


    (void) fclose(file);

    positions.endEdit();
    radius.endEdit();

    return true;
}

}//loader

}//component

}//sofa
