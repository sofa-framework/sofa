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
#ifndef SOFA_COMPONENT_FORCEFIELD_QUADPRESSUREFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_QUADPRESSUREFORCEFIELD_H
#include "config.h"

#include <sofa/core/behavior/ForceField.h>
#include <SofaBaseTopology/TopologySparseData.h>



namespace sofa
{

namespace component
{

namespace forcefield
{

/**
 * @brief QuadPressureForceField Class
 *
 * Implements a pressure force applied on a quad surface.
 * The force applied to each vertex of a quad is equal to the quad surface*Pressure/4.
 * The force is constant during animation. 
 */
template<class DataTypes>
class QuadPressureForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(QuadPressureForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    Data<Deriv> pressure; ///< Pressure force per unit area

    Data<sofa::helper::vector<unsigned int> > quadList; ///< Indices of quads separated with commas where a pressure is applied

    /// the normal used to define the edge subjected to the pressure force.
    Data<Deriv> normal;

    Data<Real> dmin; ///< coordinates min of the plane for the vertex selection
    Data<Real> dmax;///< coordinates max of the plane for the vertex selection
    Data<bool> p_showForces; ///< draw quads which have a given pressure

protected:

    class QuadPressureInformation
    {
    public:
        Real area;
        Deriv force;

        QuadPressureInformation() {}
        QuadPressureInformation(const QuadPressureInformation &e)
            : area(e.area),force(e.force)
        { }

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const QuadPressureInformation& /*ei*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, QuadPressureInformation& /*ei*/ )
        {
            return in;
        }
    };

    sofa::component::topology::QuadSparseData<sofa::helper::vector<QuadPressureInformation> > quadPressureMap; ///< map between edge indices and their pressure

    sofa::core::topology::BaseMeshTopology* _topology;

    QuadPressureForceField()
        : pressure(initData(&pressure, "pressure", "Pressure force per unit area"))
        , quadList(initData(&quadList,"quadList", "Indices of quads separated with commas where a pressure is applied"))
        , normal(initData(&normal,"normal", "Normal direction for the plane selection of quads"))
        , dmin(initData(&dmin,(Real)0.0, "dmin", "Minimum distance from the origin along the normal direction"))
        , dmax(initData(&dmax,(Real)0.0, "dmax", "Maximum distance from the origin along the normal direction"))
        , p_showForces(initData(&p_showForces, (bool)false, "showForces", "draw quads which have a given pressure"))
        , quadPressureMap(initData(&quadPressureMap, "quadPressureMap", "map between edge indices and their pressure"))
    {
    }

    virtual ~QuadPressureForceField();
public:
    virtual void init() override;

    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) override;

    /// Constant pressure has null variation
    virtual void addKToMatrix(sofa::defaulttype::BaseMatrix * /*m*/, SReal /*kFactor*/, unsigned int & /*offset*/) override {}

    /// Constant pressure has null variation
    virtual void addKToMatrix(const core::MechanicalParams* /*mparams*/, const sofa::core::behavior::MultiMatrixAccessor* /*matrix*/ ) override {}

    virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override { serr << "Get potentialEnergy not implemented" << sendl; return 0.0; }

    void draw(const core::visual::VisualParams* vparams) override;

    void setDminAndDmax(const SReal _dmin, const SReal _dmax)
    {
        dmin.setValue((Real)_dmin); dmax.setValue((Real)_dmax);
    }

    void setNormal(const Coord n) { normal.setValue(n);}

    void setPressure(Deriv _pressure) { this->pressure = _pressure; updateQuadInformation(); }

protected :
    void selectQuadsAlongPlane();
    void selectQuadsFromString();
    void updateQuadInformation();
    void initQuadInformation();
    bool isPointInPlane(Coord p)
    {
        Real d=dot(p,normal.getValue());
        if ((d>dmin.getValue())&& (d<dmax.getValue()))
            return true;
        else
            return false;
    }
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_QUADPRESSUREFORCEFIELD_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_BOUNDARY_CONDITION_API QuadPressureForceField<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BOUNDARY_CONDITION_API QuadPressureForceField<defaulttype::Vec3fTypes>;
#endif

#endif // defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_QUADPRESSUREFORCEFIELD_CPP)


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_QUADPRESSUREFORCEFIELD_H
