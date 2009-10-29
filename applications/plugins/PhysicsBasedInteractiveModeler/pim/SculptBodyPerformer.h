/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef PLUGINS_PIM_SCULPTBODYPERFORMER_H
#define PLUGINS_PIM_SCULPTBODYPERFORMER_H

#include <sofa/component/collision/InteractionPerformer.h>
#include <sofa/component/collision/MouseInteractor.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/topology/TetrahedronSetTopologyContainer.h>
#include <sofa/component/misc/MeshTetraStuffing.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/forcefield/TetrahedronFEMForceField.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>
#include <sofa/component/mass/UniformMass.h>
#include <set>
#include <map>

namespace plugins
{

namespace pim
{

using sofa::component::linearsolver::GraphScatteredMatrix;
using sofa::component::linearsolver::GraphScatteredVector;
using namespace sofa::component::collision;
using namespace sofa;
using namespace sofa::component;

class SculptBodyPerformerConfiguration
{
public:
    SculptBodyPerformerConfiguration() {}
    void setForce(double f) {force=f;}
    void setScale(double s) {scale=s;}
    void setCheckedFix(bool b) {checkedFix = b;}

protected:
    SReal force;
    SReal scale;
    bool checkedFix;
};

template <class DataTypes>
class SculptBodyPerformer: public TInteractionPerformer<DataTypes>, public SculptBodyPerformerConfiguration
{
    typedef typename DataTypes::Coord                                         Coord;
public:
    SculptBodyPerformer(BaseMouseInteractor *i);
    ~SculptBodyPerformer() {};

    void start() {};
    void execute();
    void end();
    void draw();
    void animate(bool checked);

protected:
    std::set<unsigned int> vertexNeighborhood, vertexInInfluenceZone, alreadyCheckedVertex, fixedPoints, drawFacets, fixedFacets;
    BodyPicked picked;
    void computeNeighborhood();
    core::componentmodel::behavior::MechanicalState<DataTypes>* mstateCollision;

    unsigned int vertexIndex;
    sofa::component::topology::MeshTopology* fatMesh;
    std::set<unsigned int> modifiedVertex, triangleChecked;
    misc::MeshTetraStuffing* meshStuffed;
    sofa::component::topology::TetrahedronSetTopologyContainer* tetraMesh;
    vector<defaulttype::Vec<3,SReal> > seqPoints;
    simulation::Node *addedMateriaNode, *root;
    std::multimap<unsigned int, unsigned int> vertexMap;
    container::MechanicalObject<defaulttype::Vec3Types>* mstate;
    forcefield::TetrahedronFEMForceField<defaulttype::Vec3Types>* fem;
    odesolver::EulerImplicitSolver* ei;
    linearsolver::CGLinearSolver<GraphScatteredMatrix,GraphScatteredVector>* cgls;
    mass::UniformMass<defaulttype::Vec3dTypes,double>* mass;
};



#if defined(WIN32) && !defined(SOFA_COMPONENT_COLLISION_SCULPTBODYPERFORMER_CPP)
extern template class SOFA_COMPONENT_MISC_API SculptBodyPerformer<defaulttype::Vec3Types>;
#endif


}
}

#endif
