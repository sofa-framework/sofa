/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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

/**-------------------------------------------------------------------------
*						--   Description   --
*	PMLReader is used to import a PML document to the sofa structure.
*  It builds the scenegraph with DOFs, mechanical models, and Forcefields,
*  reading a PML file, and using PMLRigid and PML ForceFields classes.
-------------------------------------------------------------------------**/

#ifndef SOFAPML_PMLREADER_H
#define SOFAPML_PMLREADER_H


#include <PhysicalModel.h>
#include <StructuralComponent.h>
#include "PMLBody.h"
#include <SofaPML/config.h>

#include <sofa/simulation/tree/GNode.h>

namespace sofa
{

namespace filemanager
{

namespace pml
{

using namespace sofa::simulation::tree;

class SOFA_BUILD_FILEMANAGER_PML_API PMLReader
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

