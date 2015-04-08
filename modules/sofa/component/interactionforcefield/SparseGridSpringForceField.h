/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_SPARSEGRIDSPRINGFORCEFIELD_H
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_SPARSEGRIDSPRINGFORCEFIELD_H

#include <SofaDeformable/StiffSpringForceField.h>
#include <sofa/component/topology/MultiResSparseGridTopology.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

template<class DataTypes>
class SparseGridSpringForceField : public sofa::component::interactionforcefield::StiffSpringForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(SparseGridSpringForceField, DataTypes), SOFA_TEMPLATE(StiffSpringForceField, DataTypes));

    typedef StiffSpringForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

protected:

    DataField< Real > linesStiffness;
    DataField< Real > linesDamping;
    DataField< Real > quadsStiffness;
    DataField< Real > quadsDamping;
    DataField< Real > cubesStiffness;
    DataField< Real > cubesDamping;

    DataField< std::string > filename;
    typedef topology::MultiResSparseGridTopology::SparseGrid Voxels;
    typedef topology::MultiResSparseGridTopology::SparseGrid::Index3D Index3D;


public:
    SparseGridSpringForceField(core::behavior::MechanicalState<DataTypes>* object1, core::behavior::MechanicalState<DataTypes>* object2)
        : StiffSpringForceField<DataTypes>(object1, object2),
          linesStiffness  (dataField(&linesStiffness,Real(100),"linesStiffness","Lines Stiffness"))
          , linesDamping  (dataField(&linesDamping  ,Real(5),"linesDamping"  ,"Lines Damping"))
          , quadsStiffness(dataField(&quadsStiffness,Real(100),"quadsStiffness","Quads Stiffness"))
          , quadsDamping  (dataField(&quadsDamping  ,Real(5),"quadsDamping"  ,"Quads Damping"))
          , cubesStiffness(dataField(&cubesStiffness,Real(100),"cubesStiffness","Cubes Stiffness"))
          , cubesDamping  (dataField(&cubesDamping  ,Real(5),"cubesDamping"  ,"Cubes Damping"))
    {
        addAlias(&linesStiffness,    "stiffness"); addAlias(&linesDamping,    "damping");
        addAlias(&quadsStiffness,    "stiffness"); addAlias(&quadsDamping,    "damping");
        addAlias(&cubesStiffness,    "stiffness"); addAlias(&cubesDamping,    "damping");
    }

    SparseGridSpringForceField()
        :
        linesStiffness  (dataField(&linesStiffness,Real(100),"linesStiffness","Lines Stiffness"))
        , linesDamping  (dataField(&linesDamping  ,Real(5),"linesDamping"  ,"Lines Damping"))
        , quadsStiffness(dataField(&quadsStiffness,Real(100),"quadsStiffness","Quads Stiffness"))
        , quadsDamping  (dataField(&quadsDamping  ,Real(5),"quadsDamping"  ,"Quads Damping"))
        , cubesStiffness(dataField(&cubesStiffness,Real(100),"cubesStiffness","Cubes Stiffness"))
        , cubesDamping  (dataField(&cubesDamping  ,Real(5),"cubesDamping"  ,"Cubes Damping"))
    {
        addAlias(&linesStiffness,    "stiffness"); addAlias(&linesDamping,    "damping");
        addAlias(&quadsStiffness,    "stiffness"); addAlias(&quadsDamping,    "damping");
        addAlias(&cubesStiffness,    "stiffness"); addAlias(&cubesDamping,    "damping");
    }

    Real getStiffness() const { return linesStiffness.getValue(); }
    Real getLinesStiffness() const { return linesStiffness.getValue(); }
    Real getQuadsStiffness() const { return quadsStiffness.getValue(); }
    Real getCubesStiffness() const { return cubesStiffness.getValue(); }
    void setStiffness(Real val)
    {
        linesStiffness.setValue(val);
        quadsStiffness.setValue(val);
        cubesStiffness.setValue(val);
    }
    void setLinesStiffness(Real val)
    {
        linesStiffness.setValue(val);
    }
    void setQuadsStiffness(Real val)
    {
        quadsStiffness.setValue(val);
    }
    void setCubesStiffness(Real val)
    {
        cubesStiffness.setValue(val);
    }


    Real getDamping() const { return linesDamping.getValue(); }
    Real getLinesDamping() const { return linesDamping.getValue(); }
    Real getQuadsDamping() const { return quadsDamping.getValue(); }
    Real getCubesDamping() const { return cubesDamping.getValue(); }
    void setDamping(Real val)
    {
        linesDamping.setValue(val);
        quadsDamping.setValue(val);
        cubesDamping.setValue(val);
    }
    void setLinesDamping(Real val)
    {
        linesDamping.setValue(val);
    }
    void setQuadsDamping(Real val)
    {
        quadsDamping.setValue(val);
    }
    void setCubesDamping(Real val)
    {
        cubesDamping.setValue(val);
    }

    virtual void addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2);

    virtual void addDForce(VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2, double kFactor, double bFactor);

    virtual void draw(const core::visual::VisualParams* vparams);

    void setFileName(char* name)
    {
        filename.setValue( std::string(name) );
    }

    char * getFileName()const {return filename.getValue();}
};

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif  /* SOFA_COMPONENT_INTERACTIONFORCEFIELD_SPARSEGRIDSPRINGFORCEFIELD_H */
