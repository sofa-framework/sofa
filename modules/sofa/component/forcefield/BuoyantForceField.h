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
#ifndef SOFA_COMPONENT_FORCEFIELD_BUOYANTFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_BUOYANTFORCEFIELD_H


#include <sofa/core/behavior/ForceField.h>
#include <sofa/component/component.h>
#include <sofa/core/topology/BaseMeshTopology.h>

#include <sofa/component/topology/TriangleSetTopologyContainer.h>
#include <sofa/component/topology/TriangleSetTopologyAlgorithms.h>
#include <sofa/component/topology/TriangleSetTopologyModifier.h>
#include <sofa/component/topology/TriangleSetTopologyChange.h>

#include <sofa/component/topology/TetrahedronSetTopologyContainer.h>
#include <sofa/component/topology/TetrahedronSetTopologyModifier.h>
#include <sofa/component/topology/TetrahedronSetTopologyChange.h>
#include <sofa/component/topology/TetrahedronSetGeometryAlgorithms.h>
//#include <sofa/component/topology/EdgeSetTopologyChange.h>

namespace sofa { namespace core { namespace topology { class BaseMeshTopology; } } }


namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;


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

    typedef core::topology::BaseMeshTopology::Tetra Tetra;
    typedef core::topology::BaseMeshTopology::Triangle Triangle;

    enum FLUID { BOX, PLANE };

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

    sofa::helper::vector<int> m_surfaceTriangles;

    Data<Real>      m_immersedVolume;
    Data<Real>      m_immersedArea;
    Data<Real>      m_globalForce;

    sofa::core::topology::BaseMeshTopology* m_tetraTopology;

    sofa::component::topology::TetrahedronSetTopologyContainer* m_tetraContainer;
    sofa::component::topology::TetrahedronSetGeometryAlgorithms<DataTypes>* m_tetraGeo;

    sofa::helper::vector<Deriv> m_debugForce;
    sofa::helper::vector<Deriv> m_debugPosition;



public:

    BuoyantForceField();
    virtual ~BuoyantForceField();

    virtual void init();

    virtual void addForce(DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v, const core::MechanicalParams* mparams);
    virtual void addDForce(DataVecDeriv& /* d_df */, const DataVecDeriv& /* d_dx */, const core::MechanicalParams* mparams)
    {
        //TODO: remove this line (avoid warning message) ...
        mparams->kFactor();
    };

    void draw();

protected:

    /**
     * @brief Returns true if the x parameters belongs to the liquid modeled as a box.
     */
    inline bool isPointInFluid(const Coord& /*x*/) const;
    /**
     * @brief Returns the number of point of a tetra included in the liquid
     */
    inline int isTetraInFluid(const Tetra& /*tetra*/, const VecCoord& x) const;
    inline int isTriangleInFluid(const Triangle& /*tetra*/, const VecCoord& x) const;

    inline Real getImmersedVolume(const Tetra &tetra, const VecCoord& x) const;

    inline bool isCornerInTetra(const Tetra &tetra, const VecCoord& x) const;
};


#if defined(WIN32) && !defined(SOFA_COMPONENT_FORCEFIELD_CONICALFORCEFIELD_CPP)
#pragma warning(disable : 4231)

#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_FORCEFIELD_API BuoyantForceField<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_FORCEFIELD_API BuoyantForceField<defaulttype::Vec3fTypes>;
#endif
#endif // defined(WIN32) && !defined(SOFA_COMPONENT_FORCEFIELD_BuoyantForceField_CPP)


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_BUOYANTFORCEFIELD_H
