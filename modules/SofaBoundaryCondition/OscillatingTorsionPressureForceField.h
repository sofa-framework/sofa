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
#ifndef SOFA_COMPONENT_FORCEFIELD_OSCILLATINGTORSIONPRESSUREFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_OSCILLATINGTORSIONPRESSUREFORCEFIELD_H
#include "config.h"


#include <sofa/core/behavior/ForceField.h>
#include <SofaBaseTopology/TopologySparseData.h>
#include <fstream>



namespace sofa
{

namespace component
{

namespace forcefield
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

protected:

    class TrianglePressureInformation
    {
    public:
        Real area;

        TrianglePressureInformation() {}
        TrianglePressureInformation(const TrianglePressureInformation &e)
            : area(e.area)
        { }

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
    sofa::component::topology::TriangleSparseData<sofa::helper::vector <TrianglePressureInformation> > trianglePressureMap;
    sofa::core::topology::BaseMeshTopology* _topology;

    Data<Real> moment;   // total moment/torque applied
    Data<sofa::helper::vector<unsigned int> > triangleList;
    Data<Deriv> axis;    // axis of rotation and normal used to define the edge subjected to the pressure force
    Data<Coord> center;  // center of rotation
    Data<Real> penalty;  // strength of penalty force
    Data<Real> frequency; // frequency of change
    Data<Real> dmin;     // coordinates min of the plane for the vertex selection
    Data<Real> dmax;     // coordinates max of the plane for the vertex selection
    Data<bool> p_showForces;

protected:

    std::vector<Real> relMomentToApply;   // estimated share of moment to apply to each point
    std::vector<bool> pointActive;        // true if moment is applied to specific point (surface)
    std::vector<Coord> vecFromCenter;     // vector from rotation axis for all points
    std::vector<Real> distFromCenter;     // norm of vecFromCenter
    std::vector<Coord> momentDir;         // direction in which to apply a moment
    std::vector<Coord> origVecFromCenter; // vector from rotation axis for all points in original state
    std::vector<Coord> origCenter;        // center of rotation for original points
    SReal rotationAngle;


    OscillatingTorsionPressureForceField()
        : trianglePressureMap(initData(&trianglePressureMap, "trianglePressureMap", "map between edge indices and their pressure"))
        , moment(initData(&moment, "moment", "Moment force applied on the entire surface"))
        , triangleList(initData(&triangleList, "triangleList", "Indices of triangles separated with commas where a pressure is applied"))
        , axis(initData(&axis, Coord(0,0,1), "axis", "Axis of rotation and normal direction for the plane selection of triangles"))
        , center(initData(&center,"center", "Center of rotation"))
        , penalty(initData(&penalty, (Real)1000, "penalty", "Strength of the penalty force"))
        , frequency(initData(&frequency, (Real)1, "frequency", "frequency of oscillation"))
        , dmin(initData(&dmin,(Real)0.0, "dmin", "Minimum distance from the origin along the normal direction"))
        , dmax(initData(&dmax,(Real)0.0, "dmax", "Maximum distance from the origin along the normal direction"))
        , p_showForces(initData(&p_showForces, (bool)false, "showForces", "draw triangles which have a given pressure"))
    {
        rotationAngle = 0;
    }

    virtual ~OscillatingTorsionPressureForceField();
public:
    virtual void init();

    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);
    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& /* d_df */, const DataVecDeriv& /* d_dx */)
    {
        //TODO: remove this line (avoid warning message) ...
        mparams->setKFactorUsed(true);
    }

    virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const
    {
        serr << "Get potentialEnergy not implemented" << sendl;
        return 0.0;
    }


    void draw(const core::visual::VisualParams* vparams);

    void setDminAndDmax(const SReal _dmin, const SReal _dmax)
    {
        dmin.setValue((Real)_dmin); dmax.setValue((Real)_dmax);
    }
    void setAxis(const Coord n) { axis.setValue(n);}

    void setMoment(Real x) { moment.setValue( x ); }

    // returns the amplitude/modifier of the set moment (dependent on frequency)
    SReal getAmplitude();

    // returns the rotation of the driven part of the object relative to the original state (in radians)
    // this value is updated in addForce()
    SReal getRotationAngle() { return rotationAngle; }

protected :

    void selectTrianglesAlongPlane();

    void selectTrianglesFromString();

    void initTriangleInformation();

    bool isPointInPlane(Coord p)
    {
        Real d=dot(p,axis.getValue());
        if ((d>dmin.getValue())&& (d<dmax.getValue()))
            return true;
        else
            return false;
    }

    Coord getVecFromRotAxis( const Coord &x )
    {
        Coord vecFromCenter = x - center.getValue();
        Coord axisProj = axis.getValue() * dot( vecFromCenter, axis.getValue() ) + center.getValue();
        return (x - axisProj);
    }

    Real getAngle( const Coord &v1, const Coord &v2 )
    {
        Real dp = dot( v1, v2 ) / (v1.norm()*v2.norm());
        if (dp>1.0) dp=1.0; else if (dp<-1.0) dp=-1.0;
        Real angle = acos( dp );
        // check direction!
        if (dot( axis.getValue(), v1.cross( v2 ) ) > 0) angle *= -1;
        return angle;
    }

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_OSCILLATINGTORSIONPRESSUREFORCEFIELD_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_BOUNDARY_CONDITION_API OscillatingTorsionPressureForceField<sofa::defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BOUNDARY_CONDITION_API OscillatingTorsionPressureForceField<sofa::defaulttype::Vec3fTypes>;
#endif

#endif // defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_OSCILLATINGTORSIONPRESSUREFORCEFIELD_CPP)


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_OSCILLATINGTORSIONPRESSUREFORCEFIELD_H
