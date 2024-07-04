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
#include <sofa/core/topology/TopologySubsetData.h>
#include <fstream>

namespace sofa::component::mechanicalload
{

template<class DataTypes>
class OscillatingTorsionPressureForceField : public core::behavior::ForceField<DataTypes>
{
public:

    SOFA_CLASS(SOFA_TEMPLATE(OscillatingTorsionPressureForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;

    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

    using Index = sofa::Index;

protected:

    class TrianglePressureInformation
    {
    public:
        Real area {0.0};

        TrianglePressureInformation() = default;
        TrianglePressureInformation(const TrianglePressureInformation &e)
            : area(e.area)
        { }

        TrianglePressureInformation & operator= (const TrianglePressureInformation & other) {
            area = other.area;
            return *this;
        }

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const TrianglePressureInformation& /*ei*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, TrianglePressureInformation& /*ei*/ )
        {
            return in;
        }
    };

    std::ofstream file;

public:
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    Data<sofa::Index> trianglePressureMap;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    Data<Real> moment;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    Data<sofa::type::vector<Index> > triangleList;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    Data<Deriv> axis;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    Data<Coord> center;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    Data<Real> penalty;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    Data<Real> frequency;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    Data<Real> dmin;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    Data<Real> dmax;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MECHANICALLOAD()
    Data<bool> p_showForces;

    sofa::core::topology::TriangleSubsetData<sofa::type::vector <TrianglePressureInformation> > d_trianglePressureMap; ///< Map between triangle indices and their pressure

    Data<Real> d_moment; ///< Moment force applied on the entire surface
    Data<sofa::type::vector<Index> > d_triangleList; ///< Indices of triangles separated with commas where a pressure is applied
    Data<Deriv> d_axis; ///< Axis of rotation and normal direction for the plane selection of triangles
    Data<Coord> d_center; ///< Center of rotation
    Data<Real> d_penalty; ///< Strength of the penalty force
    Data<Real> d_frequency; ///< frequency of oscillation
    Data<Real> d_dmin; ///< Minimum distance from the origin along the normal direction
    Data<Real> d_dmax; ///< Maximum distance from the origin along the normal direction
    Data<bool> d_showForces; ///< draw triangles which have a given pressure

    /// Link to be set to the topology container in the component graph.
    SingleLink<OscillatingTorsionPressureForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

    void init() override;
    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& /* d_df */, const DataVecDeriv& /* d_dx */) override;

    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override;

    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;
    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;

    void draw(const core::visual::VisualParams* vparams) override;

    void setDminAndDmax(const SReal _dmin, const SReal _dmax){d_dmin.setValue(static_cast<Real>(_dmin));
                                                              d_dmax.setValue(static_cast<Real>(_dmax));}
    void setAxis(const Coord n) { d_axis.setValue(n);}
    void setMoment(Real x) { d_moment.setValue(x ); }

    // returns the amplitude/modifier of the set moment (dependent on frequency)
    SReal getAmplitude();

    // returns the rotation of the driven part of the object relative to the original state (in radians)
    // this value is updated in addForce()
    SReal getRotationAngle() const { return rotationAngle; }

protected :
    OscillatingTorsionPressureForceField();
    ~OscillatingTorsionPressureForceField() override;

    void selectTrianglesAlongPlane();
    void selectTrianglesFromString();
    void initTriangleInformation();
    bool isPointInPlane(Coord p);
    Coord getVecFromRotAxis( const Coord &x );
    Real getAngle( const Coord &v1, const Coord &v2 );

    sofa::type::vector<Real> relMomentToApply;   // estimated share of moment to apply to each point
    sofa::type::vector<bool> pointActive;        // true if moment is applied to specific point (surface)
    sofa::type::vector<Coord> vecFromCenter;     // vector from rotation axis for all points
    sofa::type::vector<Real> distFromCenter;     // norm of vecFromCenter
    sofa::type::vector<Coord> momentDir;         // direction in which to apply a moment
    sofa::type::vector<Coord> origVecFromCenter; // vector from rotation axis for all points in original state
    sofa::type::vector<Coord> origCenter;        // center of rotation for original points
    SReal rotationAngle;

    sofa::core::topology::BaseMeshTopology* m_topology; ///< Pointer to the current topology
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_OSCILLATINGTORSIONPRESSUREFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_MECHANICALLOAD_API OscillatingTorsionPressureForceField<sofa::defaulttype::Vec3Types>;
#endif //  !defined(SOFA_COMPONENT_FORCEFIELD_OSCILLATINGTORSIONPRESSUREFORCEFIELD_CPP)

} // namespace sofa::component::mechanicalload
