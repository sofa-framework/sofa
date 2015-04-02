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
//	PMLMappedModel references points wich are mapped on an other mechanical model
//-------------------------------------------------------------------------


#ifndef SOFAPML_PMLMAPPEDBODY_H
#define SOFAPML_PMLMAPPEDBODY_H

#include "PMLBody.h"
#include "initSofaPML.h"

#include <SofaBaseMechanics/MechanicalObject.h>
#include <map>


namespace sofa
{

namespace filemanager
{

namespace pml
{
using namespace std;
using namespace sofa::component::container;

class SOFA_BUILD_FILEMANAGER_PML_API PMLMappedBody: public PMLBody
{
public :

    PMLMappedBody(StructuralComponent* body, PMLBody* fromBody, GNode * parent);

    ~PMLMappedBody();

    string isTypeOf() { return "mapped"; }

    bool FusionBody(PMLBody*) {return false;}

    Vector3 getDOF(unsigned int );

    GNode::SPtr getPointsNode() {return parentNode;}

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
    BaseMapping::SPtr mapping;


};

}
}
}

#endif

