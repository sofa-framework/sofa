/***************************************************************************
								  LMLReader
                             -------------------
    begin             : August 9th, 2006
    copyright         : (C) 2006 TIMC-INRIA (Michael Adam)
    author            : Michael Adam
    Date              : $Date: 2007/02/25 13:51:44 $
    Version           : $Revision: 0.2 $
 ***************************************************************************/

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

#ifndef LMLREADER_H
#define LMLREADER_H


#include <Loads.h>
#include <map>

#include "sofa/core/componentmodel/behavior/MechanicalState.h"
#include "sofa/defaulttype/Vec3Types.h"
using namespace sofa::defaulttype;
#include <sofa/simulation/tree/GNode.h>
using namespace sofa::simulation::tree;

namespace sofa
{

namespace filemanager
{

namespace pml
{

class PMLReader;


class LMLReader
{
public :
    LMLReader(char* filename=NULL);

    void BuildStructure(const char* filename, PMLReader * pmlreader);
    void BuildStructure(Loads * loads, PMLReader * pmlreader);
    void BuildStructure(PMLReader * pmlreader);

    void updateStructure(Loads * loads, PMLReader * pmlreader);

private :
    Loads * loadsList;
    const char * lmlFile;
};

}
}
}

#endif //LMLREADER_H
