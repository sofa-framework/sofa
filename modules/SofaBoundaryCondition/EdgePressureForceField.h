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
#ifndef SOFA_COMPONENT_FORCEFIELD_EDGEPRESSUREFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_EDGEPRESSUREFORCEFIELD_H
#include "config.h"

#include <sofa/core/behavior/ForceField.h>
#include <SofaBaseTopology/TopologySparseData.h>
#include <SofaBaseTopology/EdgeSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TriangleSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
class EdgePressureForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(EdgePressureForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef typename DataTypes::Real        Real        ;
    typedef typename DataTypes::Coord       Coord       ;
    typedef typename DataTypes::Deriv       Deriv       ;
    typedef typename DataTypes::VecCoord    VecCoord    ;
    typedef typename DataTypes::VecDeriv    VecDeriv    ;
    typedef typename DataTypes::VecReal     VecReal     ;
    typedef Data<VecCoord>                  DataVecCoord;
    typedef Data<VecDeriv>                  DataVecDeriv;
    typedef sofa::defaulttype::Vec3d        Vec3d;
protected:

    class EdgePressureInformation
    {
    public:
        Real length;
        Deriv force;

        EdgePressureInformation(): length(0) {}
        EdgePressureInformation(const EdgePressureInformation &e)
            : length(e.length),force(e.force)
        { }


        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const EdgePressureInformation& /*ei*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, EdgePressureInformation& /*ei*/ )
        {
            return in;
        }
    };

    sofa::component::topology::EdgeSparseData<sofa::helper::vector< EdgePressureInformation> > edgePressureMap;

    sofa::core::topology::BaseMeshTopology* _topology;
    sofa::component::topology::TriangleSetTopologyContainer* _completeTopology;
    sofa::component::topology::EdgeSetGeometryAlgorithms<DataTypes>* edgeGeo;

    Data<Deriv> pressure;
    Data<helper::vector<unsigned int> > edgeIndices;
    Data<helper::vector<sofa::core::topology::Edge> > edges;
    Data<Deriv> normal; // the normal used to define the edge subjected to the pressure force
    Data<Real> dmin; // coordinates min of the plane for the vertex selection
    Data<Real> dmax;// coordinates max of the plane for the vertex selection
    Data< SReal > arrowSizeCoef; // for drawing. The sign changes the direction, 0 doesn't draw arrow
    Data< helper::vector<Real> > p_intensity; // pressure intensity on edge normal
    Data<Coord> p_binormal; // binormal of the 2D plane
    Data<bool> p_showForces;



    EdgePressureForceField()
        : edgePressureMap(initData(&edgePressureMap, "edgePressureMap", "map between edge indices and their pressure"))
        ,pressure(initData(&pressure, "pressure", "Pressure force per unit area"))
        , edgeIndices(initData(&edgeIndices,"edgeIndices", "Indices of edges separated with commas where a pressure is applied"))
        , edges(initData(&edges, "edges", "List of edges where a pressure is applied"))
        , normal(initData(&normal,"normal", "Normal direction for the plane selection of edges"))
        , dmin(initData(&dmin,(Real)0.0, "dmin", "Minimum distance from the origin along the normal direction"))
        , dmax(initData(&dmax,(Real)0.0, "dmax", "Maximum distance from the origin along the normal direction"))
        , arrowSizeCoef(initData(&arrowSizeCoef,(SReal)0.0, "arrowSizeCoef", "Size of the drawn arrows (0->no arrows, sign->direction of drawing"))
        , p_intensity(initData(&p_intensity,"p_intensity", "pressure intensity on edge normal"))
        , p_binormal(initData(&p_binormal,"binormal", "Binormal of the 2D plane"))
        , p_showForces(initData(&p_showForces, (bool)false, "showForces", "draw arrows of edge pressures"))
    {
        _completeTopology = NULL;
    }

    virtual ~EdgePressureForceField();
public:
    virtual void init();

    virtual void addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv &  dataF, const DataVecCoord &  dataX , const DataVecDeriv & dataV ) ;
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
    void setNormal(const Coord n) { normal.setValue(n);}
    void setPressure(Deriv _pressure) { this->pressure = _pressure; updateEdgeInformation(); }

protected :
    void selectEdgesAlongPlane();
    void selectEdgesFromIndices(const helper::vector<unsigned int>& inputIndices);
    void selectEdgesFromString();
    void selectEdgesFromEdgeList();
    void updateEdgeInformation();
    void initEdgeInformation();
    bool isPointInPlane(Coord p)
    {
        Real d=dot(p,normal.getValue());
        if ((d>dmin.getValue())&& (d<dmax.getValue()))
            return true;
        else
            return false;
    }
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_EDGEPRESSUREFORCEFIELD_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_BOUNDARY_CONDITION_API EdgePressureForceField<sofa::defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BOUNDARY_CONDITION_API EdgePressureForceField<sofa::defaulttype::Vec3fTypes>;
#endif

#endif //defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_EDGEPRESSUREFORCEFIELD_CPP)

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_EDGEPRESSUREFORCEFIELD_H
