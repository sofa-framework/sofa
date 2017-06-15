/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_TAITSURFACEPRESSUREFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_TAITSURFACEPRESSUREFORCEFIELD_H
#include "config.h"

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa
{

namespace component
{

namespace forcefield
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
    typedef defaulttype::Mat<DerivSize, DerivSize, Real> MatBloc;

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

    Data< Real > m_p0;                  ///< IN: Rest pressure when V = V0
    Data< Real > m_B;                   ///< IN: Bulk modulus (resistance to uniform compression)
    Data< Real > m_gamma;               ///< IN: Bulk modulus (resistance to uniform compression)
    Data< Real > m_injectedVolume;      ///< IN: Injected (or extracted) volume since the start of the simulation
    Data< Real > m_maxInjectionRate;    ///< IN: Maximum injection rate (volume per second)

    Data< Real > m_initialVolume;       ///< OUT: Initial volume, as computed from the surface rest position
    Data< Real > m_currentInjectedVolume; ///< OUT: Current injected (or extracted) volume (taking into account maxInjectionRate)
    Data< Real > m_v0;                  ///< OUT: Rest volume (as computed from initialVolume + currentInjectedVolume)
    Data< Real > m_currentVolume;       ///< OUT: Current volume, as computed from the last surface position
    Data< Real > m_currentPressure;     ///< OUT: Current pressure, as computed from the last surface position
    Data< Real > m_currentStiffness;    ///< OUT: dP/dV at current volume and pressure

    Data< SeqTriangles > m_pressureTriangles; ///< OUT: list of triangles where a pressure is applied (mesh triangles + tesselated quads)

    Data< Real > m_initialSurfaceArea;  ///< OUT: Initial surface area, as computed from the surface rest position
    Data< Real > m_currentSurfaceArea;  ///< OUT: Current surface area, as computed from the last surface position

    Data< Real > m_drawForceScale;  ///< DEBUG: scale used to render force vectors
    Data< defaulttype::Vec4f > m_drawForceColor;  ///< DEBUG: color used to render force vectors

    Data< Real > m_volumeAfterTC;  ///< OUT: Volume after a topology change
    Data< Real > m_surfaceAreaAfterTC;  ///< OUT: Surface area after a topology change

    virtual void init();
    virtual void storeResetState();
    virtual void reset();
    virtual void handleEvent(core::objectmodel::Event *event);


    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);
    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx);
    virtual void addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix );
    virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const
    {
        serr << "Get potentialEnergy not implemented" << sendl;
        return 0.0;
    }

    template<class MatrixWriter>
    void addKToMatrixT(const core::MechanicalParams* mparams, MatrixWriter m);

    virtual void draw(const core::visual::VisualParams* vparams);

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


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_TAITSURFACEPRESSUREFORCEFIELD_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_BOUNDARY_CONDITION_API TaitSurfacePressureForceField<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BOUNDARY_CONDITION_API TaitSurfacePressureForceField<defaulttype::Vec3fTypes>;
#endif
#endif // defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_TAITSURFACEPRESSUREFORCEFIELD_CPP)


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_TAITSURFACEPRESSUREFORCEFIELD_H
