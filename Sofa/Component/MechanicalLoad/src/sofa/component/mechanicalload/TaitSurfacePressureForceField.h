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
#include <sofa/component/mechanicalload/config.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>

#include <sofa/core/objectmodel/lifecycle/RenamedData.h>

namespace sofa::component::mechanicalload
{


/**
 * This component computes the volume enclosed by a surface mesh
 * and apply a pressure force following Tait's equation: $P = P_0 - B((V/V_0)^\gamma - 1)$.
 * This ForceField can be used to apply :
 *  * a constant pressure (set $B=0$ and use $P_0$)
 *  * an ideal gas pressure (set $\gamma=1$ and use $B$)
 *  * a pressure from water (set $\gamma=7$ and use $B$)
 */
template<class DataTypes>
class TaitSurfacePressureForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TaitSurfacePressureForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    enum { DerivSize = DataTypes::deriv_total_size };
    typedef type::Mat<DerivSize, DerivSize, Real> MatBloc;

    typedef core::topology::BaseMeshTopology::Edge Edge;
    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::Quad Quad;
    typedef core::topology::BaseMeshTopology::SeqEdges SeqEdges;
    typedef core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef core::topology::BaseMeshTopology::SeqQuads SeqQuads;

    typedef core::topology::BaseMeshTopology::PointID PointID;
    typedef core::topology::BaseMeshTopology::EdgeID EdgeID;

protected:

    TaitSurfacePressureForceField();
    virtual ~TaitSurfacePressureForceField();

public:

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> m_p0;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> m_B;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> m_gamma;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> m_injectedVolume;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> m_maxInjectionRate;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> m_initialVolume;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> m_currentInjectedVolume;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> m_v0;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> m_currentVolume;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> m_currentPressure;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> m_currentStiffness;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    sofa::core::objectmodel::lifecycle::RenamedData<SeqTriangles> m_pressureTriangles;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> m_initialSurfaceArea;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> m_currentSurfaceArea;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> m_drawForceScale;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    sofa::core::objectmodel::lifecycle::RenamedData<sofa::type::RGBAColor> m_drawForceColor;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> m_volumeAfterTC;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> m_surfaceAreaAfterTC;


    Data< Real > d_p0; ///< IN: Rest pressure when V = V0
    Data< Real > d_B; ///< IN: Bulk modulus (resistance to uniform compression)
    Data< Real > d_gamma; ///< IN: Bulk modulus (resistance to uniform compression)
    Data< Real > d_injectedVolume; ///< IN: Injected (or extracted) volume since the start of the simulation
    Data< Real > d_maxInjectionRate; ///< IN: Maximum injection rate (volume per second)

    Data< Real > d_initialVolume; ///< OUT: Initial volume, as computed from the surface rest position
    Data< Real > d_currentInjectedVolume; ///< OUT: Current injected (or extracted) volume (taking into account maxInjectionRate)
    Data< Real > d_v0; ///< OUT: Rest volume (as computed from initialVolume + injectedVolume)
    Data< Real > d_currentVolume; ///< OUT: Current volume, as computed from the last surface position
    Data< Real > d_currentPressure; ///< OUT: Current pressure, as computed from the last surface position
    Data< Real > d_currentStiffness; ///< OUT: dP/dV at current volume and pressure

    Data< SeqTriangles > d_pressureTriangles; ///< OUT: list of triangles where a pressure is applied (mesh triangles + tessellated quads)

    Data< Real > d_initialSurfaceArea; ///< OUT: Initial surface area, as computed from the surface rest position
    Data< Real > d_currentSurfaceArea; ///< OUT: Current surface area, as computed from the last surface position

    Data< Real > d_drawForceScale; ///< DEBUG: scale used to render force vectors
    Data< sofa::type::RGBAColor > d_drawForceColor; ///< DEBUG: color used to render force vectors

    Data< Real > d_volumeAfterTC; ///< OUT: Volume after a topology change
    Data< Real > d_surfaceAreaAfterTC; ///< OUT: Surface area after a topology change

    /// Link to be set to the topology container in the component graph.
    SingleLink<TaitSurfacePressureForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

    void init() override;
    void storeResetState() override;
    void reset() override;
    void handleEvent(core::objectmodel::Event *event) override;


    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) override;
    void addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix ) override;
    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override;
    template<class MatrixWriter>
    void addKToMatrixT(const core::MechanicalParams* mparams, MatrixWriter mwriter);
    void buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix) override;
    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;

    void draw(const core::visual::VisualParams* vparams) override;

protected:

    sofa::core::topology::BaseMeshTopology* m_topology;
    int lastTopologyRevision;
    Real reset_injectedVolume;
    Real reset_currentInjectedVolume;

    // volume gradients: normals array, scaled by area covered by each point
    VecDeriv gradV;

    virtual void updateFromTopology();
    virtual void computePressureTriangles();

    virtual void computeMeshVolumeAndArea(Real& volume, Real& area, const helper::ReadAccessor<DataVecCoord>& x);
    void computePressureAndStiffness(Real& pressure, Real& stiffness, Real currentVolume, Real v0);
    virtual void computeStatistics(const helper::ReadAccessor<DataVecCoord>& x);
};


#if !defined(SOFA_COMPONENT_FORCEFIELD_TAITSURFACEPRESSUREFORCEFIELD_CPP)

extern template class SOFA_COMPONENT_MECHANICALLOAD_API TaitSurfacePressureForceField<defaulttype::Vec3Types>;

#endif //  !defined(SOFA_COMPONENT_FORCEFIELD_TAITSURFACEPRESSUREFORCEFIELD_CPP)

} // namespace sofa::component::mechanicalload
