/***************************************************************************
								PMLMappedBody
                             -------------------
    begin             : June 12th, 2007
    copyright         : (C) 2006 TIMC-INRIA (Michael Adam)
    author            : Michael Adam
    Date              : $Date: 2007/12/06 9:32:44 $
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
//	PMLMappedModel references points wich are mapped on an other mechanical model
//-------------------------------------------------------------------------


#ifndef PMLMAPPEDBODY_H
#define PMLMAPPEDBODY_H

#include "PMLBody.h"

#include <map>


namespace sofa
{

namespace filemanager
{

namespace pml
{
using namespace std;

class PMLMappedBody: public PMLBody
{
public :

    PMLMappedBody(StructuralComponent* body, PMLBody* fromBody, GNode * parent);

    ~PMLMappedBody();

    string isTypeOf() { return "mapped"; }

    bool FusionBody(PMLBody*) {return false;}

    Vector3 getDOF(unsigned int );

    GNode* getPointsNode() {return parentNode;}

private :

    /// creation of the scene graph
    /// only a mapping and mechanical model are created
    void createForceField() {}
    void createMechanicalState(StructuralComponent* );
    void createTopology(StructuralComponent* ) {}
    void createMass(StructuralComponent* ) {}
    void createVisualModel(StructuralComponent* ) {}
    void createCollisionModel() {}

    //structure
    PMLBody * bodyRef;
    BaseMapping * mapping;


};

}
}
}

#endif

