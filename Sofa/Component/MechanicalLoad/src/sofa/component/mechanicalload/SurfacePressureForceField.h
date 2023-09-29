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


namespace sofa::core::topology
{
class BaseMeshTopology;
} // namespace sofa::core::topology


namespace sofa::component::mechanicalload
{

/**
 * @brief SurfacePressureForceField Class
 *
 * Implements a pressure force applied on a triangle or quad surface.
 * Each surfel receives a pressure in the direction of its normal.
 */
template <class DataTypes>
class SurfacePressureForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(SurfacePressureForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef type::Mat<3, 3, Real> Mat33;
    typedef type::vector<Mat33> Vec3DerivValues;
    typedef type::vector<unsigned int> Vec3DerivIndices;
    typedef type::vector<Vec3DerivValues> VecVec3DerivValues;
    typedef type::vector<Vec3DerivIndices> VecVec3DerivIndices;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    typedef sofa::core::topology::BaseMeshTopology::Triangle Triangle;
    typedef sofa::core::topology::BaseMeshTopology::Quad Quad;
    typedef sofa::core::topology::BaseMeshTopology::Index Index;
    typedef sofa::type::vector<Index> VecIndex;


    enum State { INCREASE, DECREASE };

    Data<Real> m_pressure; ///< Scalar pressure value applied on the surfaces.
    Data<Coord> m_min; ///< Lower bound of the pressured box.
    Data<Coord> m_max; ///< Upper bound of the pressured box.
    Data<VecIndex> m_triangleIndices; ///< Specify triangles by indices.
    Data<VecIndex> m_quadIndices; ///< Specify quads by indices.
    Data<bool> m_pulseMode; ///< In this mode, the pressure increases (or decreases) from 0 to m_pressure cyclicly.
    Data<Real> m_pressureLowerBound; ///< In pulseMode, the pressure increases(or decreases) from m_pressureLowerBound to m_pressure cyclicly.
    Data<Real> m_pressureSpeed; ///< Pressure variation in Pascal by second.
    Data<bool> m_volumeConservationMode; ///< In this mode, pressure variation is related to the object volume variation.
    Data<bool> m_useTangentStiffness; ///< The tangent stiffness matrix is not symmetric and could create issues with some linear solvers. Set to false to not use it.
    Data<Real> m_defaultVolume; ///< Default Volume.
    Data<Deriv> m_mainDirection; ///< Main axis for pressure application.

    Data<Real> m_drawForceScale; ///< scale used to render force vectors

protected:
    type::vector<Deriv> m_f; ///< store forces for visualization

    /// Link to be set to the topology container in the component graph.
    SingleLink<SurfacePressureForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

    State state; ///< In pulse mode, says wether pressure is increasing or decreasing.
    Real m_pulseModePressure; ///< Current pressure computed in pulse mode.

    SurfacePressureForceField();
    virtual ~SurfacePressureForceField();

public:
    void init() override;

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& /* d_df */, const DataVecDeriv& /* d_dx */) override;
    void addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;
    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;
    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord& /* x */) const override;

    void draw(const core::visual::VisualParams* vparams) override;

    void setPressure(const Real _pressure);

protected:
    /**
     * @brief Compute mesh volume.
     */
    Real computeMeshVolume(const VecDeriv& f, const VecCoord& x);

    /**
     * @brief Triangle based surface pressure computation method.
     * Each vertice receives a force equal to 1/3 of the pressure applied on its belonging triangle.
     */
    virtual void addTriangleSurfacePressure(unsigned int triId, VecDeriv& /*f*/, const VecCoord& /*x*/, const VecDeriv& /*v*/, const Real& /*pressure*/, bool computeDerivatives);


    /**
     * @brief Quad based surface pressure computation method.
     * Each vertice receives a force equal to 1/4 of the pressure applied on its belonging quad.
     */
    virtual void addQuadSurfacePressure(unsigned int quadId, VecDeriv& /*f*/, const VecCoord& /*x*/, const VecDeriv& /*v*/, const Real& /*pressure*/);


    /**
     * @brief Returns true if the x parameters belongs to the pressured box.
     */
    inline bool isInPressuredBox(const Coord& /*x*/) const;


    /**
     * @brief Returns next pressure value in pulse mode.
     * Pressure is computed according to the pressureSpeed attribute and the simulation time step.
     */
    Real computePulseModePressure();


    /**
     * @brief Computation of the derivative values
     */
    VecVec3DerivValues derivTriNormalValues;
    VecVec3DerivIndices derivTriNormalIndices;
    void verifyDerivative(VecDeriv& v_plus, VecDeriv& v, VecVec3DerivValues& DVval, VecVec3DerivIndices& DVind, const VecDeriv& Din);

    sofa::core::topology::BaseMeshTopology* m_topology;
};


template <>
void SurfacePressureForceField<defaulttype::Rigid3Types>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& /* d_df */, const DataVecDeriv& /* d_dx */);

template <>
SurfacePressureForceField<defaulttype::Rigid3Types>::Real SurfacePressureForceField<defaulttype::Rigid3Types>::computeMeshVolume(const VecDeriv& f, const VecCoord& x);

template <>
void SurfacePressureForceField<defaulttype::Rigid3Types>::addTriangleSurfacePressure(unsigned int triId, VecDeriv& /*f*/, const VecCoord& /*x*/, const VecDeriv& /*v*/, const Real& /*pressure*/, bool computeDerivatives);

template <>
void SurfacePressureForceField<defaulttype::Rigid3Types>::addQuadSurfacePressure(unsigned int quadId, VecDeriv& /*f*/, const VecCoord& /*x*/, const VecDeriv& /*v*/, const Real& /*pressure*/);

template <>
void SurfacePressureForceField<defaulttype::Rigid3Types>::verifyDerivative(VecDeriv& v_plus, VecDeriv& v, VecVec3DerivValues& DVval, VecVec3DerivIndices& DVind, const VecDeriv& Din);


#if !defined(SOFA_COMPONENT_FORCEFIELD_SURFACEPRESSUREFORCEFIELD_CPP)

extern template class SOFA_COMPONENT_MECHANICALLOAD_API SurfacePressureForceField<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_MECHANICALLOAD_API SurfacePressureForceField<defaulttype::Rigid3Types>;

#endif //  !defined(SOFA_COMPONENT_FORCEFIELD_SURFACEPRESSUREFORCEFIELD_CPP)

} // namespace sofa::component::mechanicalload
