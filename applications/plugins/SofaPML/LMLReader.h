/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

//-------------------------------------------------------------------------
//						--   Description   --
//	LMLReader is used to import a LML document to the sofa structure.
//  It builds forcefields and constraints on the objects of the scene,
//  reading a LML file, and using LMLConstraint and LMLForce classes.
//-------------------------------------------------------------------------

#ifndef SOFAPML_LMLREADER_H
#define SOFAPML_LMLREADER_H

#include <Loads.h>
#include <map>

#include "sofa/core/behavior/MechanicalState.h"
#include "sofa/defaulttype/Vec3Types.h"
#include <sofa/simulation/tree/GNode.h>
#include "initSofaPML.h"

namespace sofa
{

namespace filemanager
{

namespace pml
{

using namespace sofa::defaulttype;
using namespace sofa::simulation::tree;

class PMLReader;


class SOFA_BUILD_FILEMANAGER_PML_API LMLReader
{
public :
    LMLReader(char* filename=NULL);

    void BuildStructure(const char* filename, PMLReader * pmlreader);
    void BuildStructure(Loads * loads, PMLReader * pmlreader);
    void BuildStructure(PMLReader * pmlreader);

    void updateStructure(Loads * loads, PMLReader * pmlreader);

    void saveAsLML(const char * filename);

    unsigned int numberOfLoads() { if(loadsList)return loadsList->numberOfLoads(); else return 0;}


private :
    Loads * loadsList;
    const char * lmlFile;
};

}
}
}

#endif //LMLREADER_H
