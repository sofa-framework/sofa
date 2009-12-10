/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "GenerateRigid.h"
#include <sofa/component/init.h>
#include <sofa/simulation/common/xml/initXml.h>
#include <iostream>
#include <fstream>

using namespace sofa::defaulttype;

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 4)
    {
        std::cout <<"USAGE: "<<argv[0]<<" inputfile.obj [outputfile.rigid] [density]\n";
        return 1;
    }

    sofa::component::init();
    sofa::simulation::xml::initXml();

    sofa::helper::io::Mesh* mesh = sofa::helper::io::Mesh::Create(argv[1]);

    if (mesh == NULL)
    {
        std::cout << "ERROR loading mesh "<<argv[1]<<std::endl;
        return 2;
    }

    Vec3d center;
    Rigid3Mass mass;


    projects::GenerateRigid(mass, center, mesh);

    double density = 1;
    if (argc >= 4) density = atof( argv[3] );
    std::cout << "Using density = " << density << std::endl;
    double correctedDensity = 1000 * density; // using standard metrics, 1 cubic meter of a material with density 1 weights 1000 kg
    mass.mass *= correctedDensity;
    //mass.inertiaMatrix *= correctedDensity;

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
