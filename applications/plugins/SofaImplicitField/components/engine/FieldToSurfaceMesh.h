/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2025 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaImplicitField/config.h>
#include <SofaImplicitField/components/geometry/ScalarField.h>
#include <SofaImplicitField/MarchingCube.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <future>

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace sofaimplicitfield::component::engine
{
using namespace sofa;

typedef sofa::core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
typedef sofa::core::topology::BaseMeshTopology::Triangle Triangle;
typedef sofa::type::vector<sofa::type::Vec3d> VecCoord;

using sofa::component::geometry::ScalarField;
using sofa::core::visual::VisualParams ;
using sofa::core::objectmodel::BaseObject ;
using sofa::type::Vec3d ;

class FieldToSurfaceMesh : public BaseObject
{
public:
    SOFA_CLASS(FieldToSurfaceMesh, BaseObject);

    virtual void init() override ;
    virtual void draw(const VisualParams*params) override ;

    double getStep() const { return d_step.getValue(); }
    void setStep(double val) { d_step.setValue(val); }

    double getIsoValue() const { return d_IsoValue.getValue(); }
    void setIsoValue(double val) { d_IsoValue.setValue(val); }

    const Vec3d& getGridMin() const { return d_gridMin.getValue(); }
    void setGridMin(const Vec3d& val) { d_gridMin.setValue(val); }
    void setGridMin(double x, double y, double z) { d_gridMin.setValue( Vec3d(x,y,z)); }

    const Vec3d& getGridMax() const { return d_gridMax.getValue(); }
    void setGridMax(const Vec3d& val) { d_gridMax.setValue(val); }
    void setGridMax(double x, double y, double z) { d_gridMax.setValue( Vec3d(x,y,z)); }

protected:
    SingleLink<FieldToSurfaceMesh, ScalarField,
               BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> l_field ;

    Data <double > d_step;
    Data <double > d_IsoValue;

    Data< Vec3d > d_gridMin;
    Data< Vec3d > d_gridMax;

    /// Output
    Data<VecCoord>      d_outPoints;
    Data<SeqTriangles>  d_outTriangles;
    Data<bool>          d_debugDraw;

protected:
    FieldToSurfaceMesh() ;
    virtual ~FieldToSurfaceMesh() ;

private:
    void computeBBox(const core::ExecParams* /* params */, bool /*onlyVisible*/=false) override;

    void checkInputs();

    void generateSurfaceMesh(double isoval, double mstep, double invStep,
                             Vec3d gridmin, Vec3d gridmax,
                             sofa::component::geometry::ScalarField*);
    void updateMeshIfNeeded();

    bool hasChanged {true} ;
    VecCoord tmpPoints;
    SeqTriangles tmpTriangles;

    MarchingCube marchingCube;
};

}

