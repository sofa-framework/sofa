/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/helper/GenerateRigid.h>
#include <iostream>
#include <fstream>

#include <SofaComponentGeneral/initComponentGeneral.h>
#include <SofaComponentAdvanced/initComponentAdvanced.h>
#include <SofaComponentMisc/initComponentMisc.h>

using namespace sofa::defaulttype;

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 10)
    {
        std::cout <<"USAGE: "<<argv[0]<<" inputfile.obj [outputfile.rigid] [density] [scaleX] [scaleY] [scaleZ] [rotationX] [rotationY] [rotationZ]\n";
        return 1;
    }


    sofa::component::initComponentGeneral();
    sofa::component::initComponentAdvanced();
    sofa::component::initComponentMisc();

//////// SCALE //////
    Vector3 scale(1, 1, 1);

    for( int i = 0; i < 3; ++i) {
		if( argc > 4 + i ) scale[i] = std::atof(argv[4 + i]);
	}


//////// ROTATION from euler angles in degrees //////
    Vector3 rotation(0,0,0);

    for( int i = 0; i < 3; ++i) {
        if( argc > 7 + i ) rotation[i] = std::atof(argv[7 + i]);
    }

////// DENSITY /////////////
    SReal density = 1000;
    if (argc >= 4) density = atof( argv[3] );
    std::cout << "Using density = " << density << "kg/m^3" << std::endl;

    Vector3 center;
    Rigid3Mass mass;

    if( !generateRigid(mass, center, argv[1], density, scale, rotation) )
    {
        return 2;
    }


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
