/***************************************************************************
								  PMLReader
                             -------------------
    begin             : August 17th, 2006
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

/**-------------------------------------------------------------------------
*						--   Description   --
*	PMLReader is used to import a PML document to the sofa structure.
*  It builds the scenegraph with DOFs, mechanical models, and Forcefields,
*  reading a PML file, and using PMLRigid and PML ForceFields classes.
-------------------------------------------------------------------------**/

#ifndef PMLREADER_H
#define PMLREADER_H


#include <PhysicalModel.h>
#include <StructuralComponent.h>
#include "PMLBody.h"

#include <sofa/simulation/tree/GNode.h>
using namespace sofa::simulation::tree;

namespace sofa
{

namespace filemanager
{

namespace pml
{


class PMLReader
{
public :
    PMLReader() {pm = NULL;}

    ///build all the scene graph under the GNode root, from the pml filename
    void BuildStructure(const char* filename, GNode* root);
    ///build all the scene graph under the GNode root, from the a specified physicalmodel
    void BuildStructure(PhysicalModel * model, GNode* root);
    void BuildStructure(GNode* root);

    ///create a body (all object structure) from a PML StructuralComponent
    PMLBody* createBody(StructuralComponent* SC, GNode * root);

    ///Merge the bodies of same type which share any DOFS
    void processFusions(GNode * root);

    ///return a point position giving its pml's index
    Vector3 getAtomPos(unsigned int atomindex);

    ///save the structure under a pml file
    void saveAsPML(const char * filename);

    ///update all pml points positions
    void updatePML();

    ///the list of the bodies created
    std::vector<PMLBody*> bodiesList;

private :

    ///the physical model from which strucutre is created
    PhysicalModel * pm;


};

}
}
}

#endif //PMLREADER_H

