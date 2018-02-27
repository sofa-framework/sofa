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
#include "SceneCreator.h"
#include "SceneUtils.h"

#include <sofa/simulation/Simulation.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include "GetVectorVisitor.h"
#include "GetAssembledSizeVisitor.h"

#include <sofa/defaulttype/Vec3Types.h>
using sofa::defaulttype::Vec3Types ;

#include <SofaDeformable/StiffSpringForceField.h>

#include <SofaBaseMechanics/MechanicalObject.h>
typedef sofa::component::container::MechanicalObject<Vec3Types> MechanicalObject3;

#include <sofa/helper/system/FileRepository.h>
using sofa::helper::system::DataRepository ;

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory ;

#ifdef SOFA_HAVE_METIS
#define ARE_METIS_FEATURE_ENABLED true
#else
#define ARE_METIS_FEATURE_ENABLED false
#endif //

namespace sofa
{
namespace modeling {
using sofa::component::interactionforcefield::StiffSpringForceField ;
typedef StiffSpringForceField<Vec3Types>   StiffSpringForceField3;


/////////////////// IMPORTING THE DEPENDENCIES INTO THE NAMESPACE ///////////////////////////
using namespace sofa::defaulttype ;
typedef component::linearsolver::FullVector<SReal> FullVector ;

using helper::vector;

using sofa::simulation::graph::DAGSimulation ;
using sofa::simulation::GetAssembledSizeVisitor ;
using sofa::simulation::GetVectorVisitor ;
using sofa::simulation::Simulation ;
using sofa::simulation::Node ;


Vector getVector( core::ConstVecId id, bool indep )
{
    GetAssembledSizeVisitor getSizeVisitor;
    getSizeVisitor.setIndependentOnly(indep);
    sofa::modeling::getRoot()->execute(getSizeVisitor);
    unsigned size;
    if (id.type == sofa::core::V_COORD)
        size =  getSizeVisitor.positionSize();
    else
        size = getSizeVisitor.velocitySize();
    FullVector v(size);
    GetVectorVisitor getVec( core::MechanicalParams::defaultInstance(), &v, id);
    getVec.setIndependentOnly(indep);
    sofa::modeling::getRoot()->execute(getVec);

    Vector ve(size);
    for(size_t i=0; i<size; i++)
        ve(i)=v[i];
    return ve;
}

} /// modeling
} /// sofa
