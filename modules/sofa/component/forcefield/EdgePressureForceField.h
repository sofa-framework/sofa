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
#ifndef SOFA_COMPONENT_FORCEFIELD_EDGEPRESSUREFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_EDGEPRESSUREFORCEFIELD_H


#include <sofa/core/behavior/ForceField.h>
#include <sofa/component/topology/EdgeSubsetData.h>
#include <sofa/component/topology/EdgeSetGeometryAlgorithms.h>
#include <sofa/component/topology/TriangleSetTopologyContainer.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using namespace sofa::component::topology;

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
    };

    EdgeSubsetData<EdgePressureInformation> edgePressureMap;

    sofa::core::topology::BaseMeshTopology* _topology;
    sofa::component::topology::TriangleSetTopologyContainer* _completeTopology;
    sofa::component::topology::EdgeSetGeometryAlgorithms<DataTypes>* edgeGeo;

    Data<Deriv> pressure;
    Data<helper::vector<int> > edgeList;
    Data<Deriv> normal; // the normal used to define the edge subjected to the pressure force
    Data<Real> dmin; // coordinates min of the plane for the vertex selection
    Data<Real> dmax;// coordinates max of the plane for the vertex selection
    Data< double > arrowSizeCoef; // for drawing. The sign changes the direction, 0 doesn't draw arrow
    Data< helper::vector<Real> > p_intensity; // pressure intensity on edge normal
    Data<Coord> p_binormal; // binormal of the 2D plane

public:

    EdgePressureForceField():
        pressure(initData(&pressure, "pressure", "Pressure force per unit area"))
        , edgeList(initData(&edgeList,"edgeList", "Indices of edges separated with commas where a pressure is applied"))
        , normal(initData(&normal,"normal", "Normal direction for the plane selection of edges"))
        , dmin(initData(&dmin,(Real)0.0, "dmin", "Minimum distance from the origin along the normal direction"))
        , dmax(initData(&dmax,(Real)0.0, "dmax", "Maximum distance from the origin along the normal direction"))
        , arrowSizeCoef(initData(&arrowSizeCoef,0.0, "arrowSizeCoef", "Size of the drawn arrows (0->no arrows, sign->direction of drawing"))
        , p_intensity(initData(&p_intensity,"p_intensity", "pressure intensity on edge normal"))
        , p_binormal(initData(&p_binormal,"binormal", "Binormal of the 2D plane"))
    {
        _completeTopology = NULL;
    }

    virtual ~EdgePressureForceField();

    virtual void init();

    virtual void addForce(DataVecDeriv &  dataF, const DataVecCoord &  dataX , const DataVecDeriv & dataV, const sofa::core::MechanicalParams* /*mparams*/ ) ;
    virtual void addDForce(DataVecDeriv& /* d_df */, const DataVecDeriv& /* d_dx */, const core::MechanicalParams* mparams)
    {
        //TODO: remove this line (avoid warning message) ...
        mparams->kFactor();
    };

    // Handle topological changes
    virtual void handleTopologyChange();

    void draw();

    void setDminAndDmax(const double _dmin, const double _dmax)
    {
        dmin.setValue((Real)_dmin); dmax.setValue((Real)_dmax);
    }
    void setNormal(const Coord n) { normal.setValue(n);}
    void setPressure(Deriv _pressure) { this->pressure = _pressure; updateEdgeInformation(); }

protected :
    void selectEdgesAlongPlane();
    void selectEdgesFromString();
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

using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec3fTypes;

#if defined(WIN32) && !defined(SOFA_COMPONENT_FORCEFIELD_EDGEPRESSUREFORCEFIELD_CPP)

#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_FORCEFIELD_API EdgePressureForceField<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_FORCEFIELD_API EdgePressureForceField<Vec3fTypes>;
#endif

#endif //defined(WIN32) && !defined(SOFA_COMPONENT_FORCEFIELD_EDGEPRESSUREFORCEFIELD_CPP)

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_EDGEPRESSUREFORCEFIELD_H
