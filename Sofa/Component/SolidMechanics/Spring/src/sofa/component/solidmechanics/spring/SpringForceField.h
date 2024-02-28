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
#include <sofa/component/solidmechanics/spring/config.h>

#include <sofa/core/behavior/PairInteractionForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/component/solidmechanics/spring/LinearSpring.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/type/vector.h>
#include <sofa/helper/accessor.h>

#include <sofa/core/objectmodel/DataFileName.h>

#include <sofa/core/topology/TopologySubsetIndices.h>

namespace sofa::component::solidmechanics::spring
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class DataTypes>
class SpringForceFieldInternalData
{
public:
};

/// Set of simple springs between particles
template<class DataTypes>
class SpringForceField : public core::behavior::PairInteractionForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(SpringForceField,DataTypes), SOFA_TEMPLATE(core::behavior::PairInteractionForceField,DataTypes));

    typedef typename core::behavior::PairInteractionForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

    typedef helper::ReadAccessor< Data< VecCoord > > RDataRefVecCoord;
    typedef helper::WriteAccessor< Data< VecCoord > > WDataRefVecCoord;
    typedef helper::ReadAccessor< Data< VecDeriv > > RDataRefVecDeriv;
    typedef helper::WriteAccessor< Data< VecDeriv > > WDataRefVecDeriv;

    typedef core::behavior::MechanicalState<DataTypes> MechanicalState;

    typedef LinearSpring<Real> Spring;

    Data<SReal> ks; ///< uniform stiffness for the all springs
    Data<SReal> kd; ///< uniform damping for the all springs
    Data<float> showArrowSize; ///< size of the axis
    Data<int> drawMode;             ///Draw Mode: 0=Line - 1=Cylinder - 2=Arrow
    Data<sofa::type::vector<Spring> > springs; ///< pairs of indices, stiffness, damping, rest length

protected:
    core::objectmodel::DataFileName fileSprings;

    std::array<sofa::core::topology::TopologySubsetIndices, 2> d_springsIndices
    {
        sofa::core::topology::TopologySubsetIndices {initData ( &d_springsIndices[0], "springsIndices1", "List of indices in springs from the first mstate", true, true)},
        sofa::core::topology::TopologySubsetIndices {initData ( &d_springsIndices[1], "springsIndices2", "List of indices in springs from the second mstate", true, true)}
    };
    bool areSpringIndicesDirty { true };

    void initializeTopologyHandler(sofa::core::topology::TopologySubsetIndices& indices, core::topology::BaseMeshTopology* topology, sofa::Index mstateId);
    void updateTopologyIndicesFromSprings();
    void applyRemovedPoints(const sofa::core::topology::PointsRemoved* pointsRemoved, sofa::Index mstateId);
    void applyRemovedEdges(const sofa::core::topology::EdgesRemoved* edgesRemoved, sofa::Index mstateId);

protected:
    bool maskInUse;
    Real m_potentialEnergy;
    class Loader;

    SpringForceFieldInternalData<DataTypes> data;
    friend class SpringForceFieldInternalData<DataTypes>;

    virtual void addSpringForce(Real& potentialEnergy, VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1, VecDeriv& f2, const VecCoord& p2, const VecDeriv& v2, sofa::Index /*i*/, const Spring& spring);

    struct SpringForce
    {
        using DPos = typename DataTypes::DPos;
        Real energy;
        std::pair<DPos, DPos> force;
    };

    virtual std::unique_ptr<SpringForce> computeSpringForce(const VecCoord& p1, const VecDeriv& v1, const VecCoord& p2, const VecDeriv& v2, const Spring& spring);

    SpringForceField(SReal _ks=100.0, SReal _kd=5.0);
    SpringForceField(MechanicalState* object1, MechanicalState* object2, SReal _ks=100.0, SReal _kd=5.0);

public:
    bool load(const char *filename);

    core::behavior::MechanicalState<DataTypes>* getObject1() { return this->mstate1; }
    core::behavior::MechanicalState<DataTypes>* getObject2() { return this->mstate2; }

    const sofa::type::vector< Spring >& getSprings() const {return springs.getValue();}

    void reinit() override;
    void init() override;

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f1, DataVecDeriv& f2, const DataVecCoord& x1, const DataVecCoord& x2, const DataVecDeriv& v1, const DataVecDeriv& v2) override;
    void addDForce(const core::MechanicalParams*, DataVecDeriv& df1, DataVecDeriv& df2, const DataVecDeriv& dx1, const DataVecDeriv& dx2 ) override;

    // Make other overloaded version of getPotentialEnergy() to show up in subclass.
    using Inherit::getPotentialEnergy;
    SReal getPotentialEnergy(const core::MechanicalParams* /* PARAMS FIRST */, const DataVecCoord& data_x1, const DataVecCoord& data_x2) const override;

    using Inherit::addKToMatrix;
    virtual void addKToMatrix(sofa::linearalgebra::BaseMatrix * /*mat*/, SReal /*kFact*/, unsigned int &/*offset*/);

    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;
    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) override;

    SReal getStiffness() const { return ks.getValue(); }
    SReal getDamping() const { return kd.getValue(); }
    void setStiffness(SReal _ks) { ks.setValue(_ks); }
    void setDamping(SReal _kd) { kd.setValue(_kd); }
    SReal getArrowSize() const {return showArrowSize.getValue();}
    void setArrowSize(float s) {showArrowSize.setValue(s);}
    int getDrawMode() const {return drawMode.getValue();}
    void setDrawMode(int m) {drawMode.setValue(m);}

    void draw(const core::visual::VisualParams* vparams) override;
    void computeBBox(const core::ExecParams* params, bool onlyVisible=false) override;

    // -- Modifiers

    void clear(sofa::Size reserve=0);

    void removeSpring(sofa::Index idSpring);

    void addSpring(sofa::Index m1, sofa::Index m2, SReal ks, SReal kd, SReal initlen);

    void addSpring(const Spring & spring);

    /// initialization to export kinetic, potential energy  and force intensity to gnuplot files format
    void initGnuplot(const std::string path) override;

    /// export kinetic and potential energy state at "time" to a gnuplot file
    void exportGnuplot(SReal time) override;

    protected:
    /// stream to export Potential Energy to gnuplot files
    std::ofstream* m_gnuplotFileEnergy;
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_SPRINGFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API SpringForceField<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API SpringForceField<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API SpringForceField<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API SpringForceField<defaulttype::Vec6Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API SpringForceField<defaulttype::Rigid3Types>;
#endif

} // namespace sofa::component::solidmechanics::spring
