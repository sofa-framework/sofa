/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_SURFACEPRESSUREFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_SURFACEPRESSUREFORCEFIELD_H
#include "config.h"

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>

namespace sofa { namespace core { namespace topology { class BaseMeshTopology; } } }


namespace sofa
{

namespace component
{

namespace forcefield
{


/**
 * @brief SurfacePressureForceField Class
 *
 * Implements a pressure force applied on a triangle or quad surface.
 * Each surfel receives a pressure in the direction of its normal.
 */
template<class DataTypes>
class SurfacePressureForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(SurfacePressureForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;
    typedef defaulttype::Mat<3,3,Real> Mat33;
    typedef helper::vector< Mat33 > Vec3DerivValues;
    typedef helper::vector< unsigned int > Vec3DerivIndices;
    typedef helper::vector< Vec3DerivValues> VecVec3DerivValues;
    typedef helper::vector< Vec3DerivIndices > VecVec3DerivIndices;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    typedef sofa::core::topology::BaseMeshTopology::Triangle Triangle;
    typedef sofa::core::topology::BaseMeshTopology::Quad Quad;
    typedef sofa::core::topology::BaseMeshTopology::index_type Index;
    typedef sofa::helper::vector< Index > VecIndex;

    enum State { INCREASE, DECREASE };

protected:

    sofa::core::topology::BaseMeshTopology* m_topology;

    Data< Real >	m_pressure;					///< Scalar pressure value applied on the surfaces.
    Data< Coord >	m_min;						///< Lower bound of the pressured box.
    Data< Coord >	m_max;						///< Upper bound of the pressured box.
    Data< VecIndex >	m_triangleIndices;		///< Specify triangles by indices.
    Data< VecIndex >	m_quadIndices;			///< Specify quads by indices.
    Data< bool >	m_pulseMode;				///< In this mode, the pressure increases (or decreases) from 0 to m_pressure cyclicly.
    Data< Real >	m_pressureLowerBound;		///< In pulseMode, the pressure increases(or decreases) from m_pressureLowerBound to m_pressure cyclicly.
    Data< Real >	m_pressureSpeed;			///< Pressure variation in Pascal by second.
    Data< bool >	m_volumeConservationMode;	///< In this mode, pressure variation is related to the object volume variation.
    Data< bool >	m_useTangentStiffness;	    ///< The tangent stiffness matrix is not symmetric and could create issues with some linear solvers. Set to false to not use it.
    Data< Real >	m_defaultVolume;			///< Default Volume.
    Data< Deriv >	m_mainDirection;			///< Main axis for pressure application.

    Data< Real > m_drawForceScale;  ///< scale used to render force vectors
    helper::vector< Deriv> m_f;             ///< store forces for visualization

    State state;								///< In pulse mode, says wether pressure is increasing or decreasing.
    Real m_pulseModePressure;					///< Current pressure computed in pulse mode.

    SurfacePressureForceField();
    virtual ~SurfacePressureForceField();
public:
    virtual void init() override;

    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& /* d_df */, const DataVecDeriv& /* d_dx */) override;
    virtual void addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix ) override;
    virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override
    {
        serr << "Get potentialEnergy not implemented" << sendl;
        return 0.0;
    }

    void draw(const core::visual::VisualParams* vparams) override;

    void setPressure(const Real _pressure)
    {
        this->m_pressure.setValue(_pressure);
    }

protected:

    /**
     * @brief Compute mesh volume.
     */
    Real computeMeshVolume(const VecDeriv& f,const VecCoord& x);

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
    const Real computePulseModePressure(void);


    /**
     * @brief Computation of the derivative values
     */
    VecVec3DerivValues derivTriNormalValues;
    VecVec3DerivIndices derivTriNormalIndices;
    void verifyDerivative(VecDeriv& v_plus, VecDeriv& v,  VecVec3DerivValues& DVval, VecVec3DerivIndices& DVind, const VecDeriv& Din);
};

#ifndef SOFA_FLOAT
template<>
void SurfacePressureForceField<defaulttype::Rigid3dTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& /* d_df */, const DataVecDeriv& /* d_dx */);

template<>
SurfacePressureForceField<defaulttype::Rigid3dTypes>::Real SurfacePressureForceField<defaulttype::Rigid3dTypes>::computeMeshVolume(const VecDeriv& f,const VecCoord& x);

template<>
void SurfacePressureForceField<defaulttype::Rigid3dTypes>::addTriangleSurfacePressure(unsigned int triId, VecDeriv& /*f*/, const VecCoord& /*x*/, const VecDeriv& /*v*/, const Real& /*pressure*/, bool computeDerivatives);

template<>
void SurfacePressureForceField<defaulttype::Rigid3dTypes>::addQuadSurfacePressure(unsigned int quadId, VecDeriv& /*f*/, const VecCoord& /*x*/, const VecDeriv& /*v*/, const Real& /*pressure*/);

template<>
void SurfacePressureForceField<defaulttype::Rigid3dTypes>::verifyDerivative(VecDeriv& v_plus, VecDeriv& v,  VecVec3DerivValues& DVval, VecVec3DerivIndices& DVind, const VecDeriv& Din);


template<>
void SurfacePressureForceField<defaulttype::Rigid3dTypes>::draw(const core::visual::VisualParams* vparams);

#endif

#ifndef SOFA_DOUBLE
template<>
void SurfacePressureForceField<defaulttype::Rigid3fTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& /* d_df */, const DataVecDeriv& /* d_dx */);

template<>
SurfacePressureForceField<defaulttype::Rigid3fTypes>::Real SurfacePressureForceField<defaulttype::Rigid3fTypes>::computeMeshVolume(const VecDeriv& f,const VecCoord& x);

template<>
void SurfacePressureForceField<defaulttype::Rigid3fTypes>::addTriangleSurfacePressure(unsigned int triId, VecDeriv& /*f*/, const VecCoord& /*x*/, const VecDeriv& /*v*/, const Real& /*pressure*/, bool computeDerivatives);

template<>
void SurfacePressureForceField<defaulttype::Rigid3fTypes>::addQuadSurfacePressure(unsigned int quadId, VecDeriv& /*f*/, const VecCoord& /*x*/, const VecDeriv& /*v*/, const Real& /*pressure*/);

template<>
void SurfacePressureForceField<defaulttype::Rigid3fTypes>::verifyDerivative(VecDeriv& v_plus, VecDeriv& v,  VecVec3DerivValues& DVval, VecVec3DerivIndices& DVind, const VecDeriv& Din);

template<>
void SurfacePressureForceField<defaulttype::Rigid3fTypes>::draw(const core::visual::VisualParams* vparams);

#endif


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_SURFACEPRESSUREFORCEFIELD_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_BOUNDARY_CONDITION_API SurfacePressureForceField<defaulttype::Vec3dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API SurfacePressureForceField<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BOUNDARY_CONDITION_API SurfacePressureForceField<defaulttype::Vec3fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API SurfacePressureForceField<defaulttype::Rigid3fTypes>;
#endif
#endif // defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_SURFACEPRESSUREFORCEFIELD_CPP)


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_SURFACEPRESSUREFORCEFIELD_H
