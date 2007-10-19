#include "GenerateRigid.h"
#include <sofa/simulation/tree/init.h>
#include <iostream>
#include <fstream>

using namespace sofa::defaulttype;

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3)
    {
        std::cout <<"USAGE: "<<argv[0]<<" inputfile.obj [outputfile.rigid]\n";
        return 1;
    }

    sofa::simulation::tree::init();

    sofa::helper::io::Mesh* mesh = sofa::helper::io::Mesh::Create(argv[1]);

    if (mesh == NULL)
    {
        std::cout << "ERROR loading mesh "<<argv[1]<<std::endl;
        return 2;
    }

    Vec3d center;
    Rigid3Mass mass;


    projects::GenerateRigid(mass, center, mesh);

    std::ostream* out = &std::cout;
    if (argc >= 3)
    {
        out = new std::ofstream(argv[2]);
    }

    out->setf(std::ios::fixed);
    out->precision(10);

    *out << "Xsp 3.0\n";
    *out << "mass " << mass.mass << "\n";
    *out << "volm " << mass.volume << "\n";
    *out << "inrt " << mass.inertiaMatrix[0][0] << " " << mass.inertiaMatrix[0][1] << " " << mass.inertiaMatrix[0][2] << " "
            << mass.inertiaMatrix[1][0] << " " << mass.inertiaMatrix[1][1] << " " << mass.inertiaMatrix[1][2] << " "
            << mass.inertiaMatrix[2][0] << " " << mass.inertiaMatrix[2][1] << " " << mass.inertiaMatrix[2][2] << "\n";
    *out << "cntr " << center[0] << " " << center[1] << " " << center[2] << "\n";
    *out << std::flush;

    if (out != &std::cout)
        delete out;

    return 0;
}
