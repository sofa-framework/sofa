/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once
#include <SofaUserInteraction/config.h>

#include <SofaUserInteraction/InteractionPerformer.h>

#include <SofaDeformable/StiffSpringForceField.h>
#include <SofaUserInteraction/MouseInteractor.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <sofa/simulation/Node.h>

#include <unordered_map>
#include <typeindex>

namespace sofa::component::collision
{

class FixParticlePerformerConfiguration
{
public:
    void setStiffness(SReal s) {stiffness=s;}
protected:
    SReal stiffness;
};

template <class DataTypes>
class FixParticlePerformer: public TInteractionPerformer<DataTypes>, public FixParticlePerformerConfiguration
{
public:
    typedef sofa::component::interactionforcefield::StiffSpringForceField< DataTypes >   MouseForceField;
    typedef sofa::component::container::MechanicalObject< DataTypes >         MouseContainer;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;

    FixParticlePerformer(BaseMouseInteractor *i);

    void start();
    void execute();
    void draw(const core::visual::VisualParams* vparams);

    using GetFixationPointsOnModelFunction = std::function<void(sofa::core::sptr<sofa::core::CollisionModel>, const Index, helper::vector<Index>&, Coord&)>;

    template<typename TCollisionModel>
    static int RegisterSupportedModel(GetFixationPointsOnModelFunction func)
    {
        s_mapSupportedModels[std::type_index(typeid(TCollisionModel))] = func;

        return 1;
    }

    template<typename TTriangleCollisionModel>
    static void getFixationPointsTriangle(sofa::core::sptr<sofa::core::CollisionModel> model, const Index idx, helper::vector<Index>& points, Coord& fixPoint)
    {
        auto* triangle = static_cast<TTriangleCollisionModel*>(model.get());

        Triangle t(triangle, idx);
        fixPoint = (t.p1() + t.p2() + t.p3()) / 3.0;
        points.push_back(t.p1Index());
        points.push_back(t.p2Index());
        points.push_back(t.p3Index());
    }

    static void getFixationPointsSphere(sofa::core::sptr<sofa::core::CollisionModel> model, const Index idx, helper::vector<Index>& points, Coord& fixPoint)
    {
        auto* collisionState = model->getContext()->getMechanicalState();
        fixPoint[0] = collisionState->getPX(idx);
        fixPoint[1] = collisionState->getPY(idx);
        fixPoint[2] = collisionState->getPZ(idx);

        points.push_back(idx);
    }

protected:
    MouseContainer* getFixationPoints(const BodyPicked &b, helper::vector<unsigned int> &points, typename DataTypes::Coord &fixPoint);

    std::vector< simulation::Node * > fixations;

    // inline initialization of templated static members works on VS2019/gcc/clang (>5?)
    // but not on VS2017 and crash the compilation itself using clang5.
    // TODO: once VS2017 and clang5 support is dropped, just uncomment the inline static initialization 
    // and remove the initialization in the inl file (linux/mac) and cpp (windows)
    //inline static std::unordered_map<std::type_index, GetFixationPointsOnModelFunction > s_mapSupportedModels;
    static std::unordered_map<std::type_index, GetFixationPointsOnModelFunction > s_mapSupportedModels;

};

#if  !defined(SOFA_COMPONENT_COLLISION_FIXPARTICLEPERFORMER_CPP)
extern template class SOFA_SOFAUSERINTERACTION_API FixParticlePerformer<defaulttype::Vec3Types>;

#endif


} // namespace sofa::component::collision
