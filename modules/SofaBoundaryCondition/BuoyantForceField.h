/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_BUOYANTFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_BUOYANTFORCEFIELD_H
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
 * @brief BuoyantForceField Class
 *
 * Simulates the effect of buoyancy on an object.
 * Applies an upward force exerted by a fluid, that opposes the object's weight
 * The fluid is modeled by a static axis-aligned bounding box
 */
template<class DataTypes>
class BuoyantForceField : public core::behavior::ForceField<DataTypes>
{

public:
    SOFA_CLASS(SOFA_TEMPLATE(BuoyantForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    typedef core::topology::BaseMeshTopology::index_type ID;
    typedef core::topology::BaseMeshTopology::Tetra Tetra;
    typedef core::topology::BaseMeshTopology::SeqTetrahedra SeqTetrahedra;
    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::SeqTriangles seqTriangles;



    enum FLUID { AABOX, PLANE };

protected:



    FLUID fluidModel;
    Data< Real > m_fluidModel;

    Data< Coord >   m_minBox;                       ///< Lower bound of the liquid box.
    Data< Coord >   m_maxBox;                       ///< Upper bound of the liquid box.

    Data <Real>     m_heightPlane;              //orthogonal to the gravity

    Data <Real>     m_fluidDensity;
    Data <Real>     m_fluidViscosity;
    Data <Real>     m_atmosphericPressure;

    Data<bool>      m_enableViscosity;
    Data<bool>      m_turbulentFlow;    //1 for turbulent, 0 for laminar

    sofa::helper::vector<ID> m_triangles;

//    Data<Real>      m_immersedVolume;
//    Data<Real>      m_immersedArea;
//    Data<Real>      m_globalForce;

    Data<bool>      m_flipNormals;

    Data<bool>      m_showPressureForces;
    Data<bool>      m_showViscosityForces;
    Data<bool>      m_showBoxOrPlane;
    Data<Real>      m_showFactorSize;

    sofa::core::topology::BaseMeshTopology* m_topology;

    sofa::helper::vector<Deriv> m_showForce;
    sofa::helper::vector<Deriv> m_showViscosityForce;
    sofa::helper::vector<Deriv> m_showPosition;

    Coord      m_fluidSurfaceOrigin; //in case of a box, indicates which face is the surface
    Coord      m_fluidSurfaceDirection; //in case of a box, indicates which face is the surface

    Deriv       m_gravity; //store the local gravity to check if it changes during the simulation
    Real         m_gravityNorm;

    Coord       m_minBoxPrev;
    Coord       m_maxBoxPrev;



    BuoyantForceField();
    virtual ~BuoyantForceField();
public:
    virtual void init() override;

    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv&  d_df , const DataVecDeriv&  d_dx ) override;
    virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override
    {
        serr << "Get potentialEnergy not implemented" << sendl;
        return 0.0;
    }

    using core::behavior::ForceField<DataTypes>::addKToMatrix;
    virtual void addKToMatrix(const sofa::core::behavior::MultiMatrixAccessor* /*matrix*/, SReal /*kFact*/) {}

    void draw(const core::visual::VisualParams* vparams) override;

protected:

    /**
     * @brief Returns true if the x parameters belongs to the liquid modeled as a box.
     */
    inline bool isPointInFluid(const Coord& /*x*/);
    /**
     * @brief Returns the number of point of a tetra included in the liquid
     */
    //inline int isTetraInFluid(const Tetra& /*tetra*/, const VecCoord& x);
    inline int isTriangleInFluid(const Triangle& /*tetra*/, const VecCoord& x);

    //inline Real getImmersedVolume(const Tetra &tetra, const VecCoord& x);

    //inline bool isCornerInTetra(const Tetra &tetra, const VecCoord& x) const;

    inline Real distanceFromFluidSurface(const Coord& x);
    inline Real D_distanceFromFluidSurface(const Deriv& dx);

    inline bool checkParameters();
};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_BUOYANTFORCEFIELD_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_BOUNDARY_CONDITION_API BuoyantForceField<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BOUNDARY_CONDITION_API BuoyantForceField<defaulttype::Vec3fTypes>;
#endif
#endif // defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_BuoyantForceField_CPP)


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_BUOYANTFORCEFIELD_H
