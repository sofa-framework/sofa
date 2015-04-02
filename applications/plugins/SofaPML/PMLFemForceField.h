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
//	PMLFemForceField translate an FEM object from Physical Model structure
//  to sofa structure, using TetrahedronFEMForcefield.
//  It inherits from PMLBody abstract class.
//-------------------------------------------------------------------------


#ifndef SOFAPML_PMLFEMFORCEFIELD_H
#define SOFAPML_PMLFEMFORCEFIELD_H

#include "PMLBody.h"

#include <StructuralComponent.h>
#include "sofa/core/topology/BaseMeshTopology.h"
#include <SofaMeshCollision/TriangleModel.h>
//#include <SofaMeshCollision/LineModel.h>
//#include <SofaMeshCollision/PointModel.h>
#include "initSofaPML.h"



#include <map>


namespace sofa
{

namespace filemanager
{

namespace pml
{
using namespace sofa::component::container;
using namespace sofa::core::topology;
using namespace sofa::component::collision;
using namespace std;

class SOFA_BUILD_FILEMANAGER_PML_API PMLFemForceField: public PMLBody
{
public :

    PMLFemForceField(StructuralComponent* body, GNode * parent);

    ~PMLFemForceField();

    string isTypeOf() { return "FEM"; }

    ///accessors
    TriangleModel::SPtr getTriangleModel() { return tmodel; }
    //LineModel * getLineModel() { return lmodel; }
    //PointModel * getPointModel() { return pmodel; }

    ///merge a body with current object
    bool FusionBody(PMLBody*);

    Vector3 getDOF(unsigned int index);

    GNode::SPtr getPointsNode() {return parentNode;}

private :

    /// creation of the scene graph
    void createMechanicalState(StructuralComponent* body);
    void createTopology(StructuralComponent* body);
    void createMass(StructuralComponent* body);
    void createVisualModel(StructuralComponent* body);
    void createForceField();
    void createCollisionModel();

    //initialization of properties
    void initMass(string m);
    void initDensity(string m);

    //tesselation of hexahedron to 5 tetrahedrons
    BaseMeshTopology::Tetra * Tesselate(Cell* pCell);
    
    //structure
    TriangleModel::SPtr tmodel;
    //LineModel * lmodel;
    //PointModel * pmodel;

    //members for the mass (only one of the 2 vectors is filled)
    std::vector<SReal> massList;
    std::vector<SReal> density;

    //members for FEM properties
    SReal young;
    SReal poisson;
    std::string deformationType;

};

}
}
}

#endif

