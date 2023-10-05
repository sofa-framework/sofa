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
#include <sofa/component/solidmechanics/spring/config.h>

#include <sofa/component/solidmechanics/spring/StiffSpringForceField.h>
#include <set>

namespace sofa::component::solidmechanics::spring
{

template<class DataTypes>
class MeshSpringForceField : public virtual StiffSpringForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(MeshSpringForceField, DataTypes), SOFA_TEMPLATE(StiffSpringForceField, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

    using Inherit1::mstate1;
    using Inherit1::mstate2;
    using Inherit1::springs;

protected:
    Data< Real >  d_linesStiffness; ///< Stiffness for the Lines
    Data< Real >  d_linesDamping; ///< Damping for the Lines
    Data< Real >  d_trianglesStiffness; ///< Stiffness for the Triangles
    Data< Real >  d_trianglesDamping; ///< Damping for the Triangles
    Data< Real >  d_quadsStiffness; ///< Stiffness for the Quads
    Data< Real >  d_quadsDamping; ///< Damping for the Quads
    Data< Real >  d_tetrahedraStiffness; ///< Stiffness for the Tetrahedra
    Data< Real >  d_tetrahedraDamping; ///< Damping for the Tetrahedra
    Data< Real >  d_cubesStiffness; ///< Stiffness for the Cubes
    Data< Real >  d_cubesDamping; ///< Damping for the Cubes
    Data< bool >  d_noCompression; ///< Only consider elongation
    Data< Real >  d_drawMinElongationRange; ///< Min range of elongation (red eongation - blue neutral - green compression)
    Data< Real >  d_drawMaxElongationRange; ///< Max range of elongation (red eongation - blue neutral - green compression)
    Data< Real >  d_drawSpringSize; ///< Size of drawed lines

    /// optional range of local DOF indices. Any computation involving only indices outside of this range are discarded (useful for parallelization using mesh partitionning)
    Data< type::Vec<2, sofa::Index> > d_localRange;

    /// Link to be set to the topology container in the component graph.
    SingleLink<MeshSpringForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

    void addSpring(std::set<std::pair<sofa::Index, sofa::Index> >& sset, sofa::Index m1, sofa::Index m2, Real stiffness, Real damping);

    MeshSpringForceField() ;
    virtual ~MeshSpringForceField();
public:
    Real getStiffness() const { return d_linesStiffness.getValue(); }
    Real getLinesStiffness() const { return d_linesStiffness.getValue(); }
    Real getTrianglesStiffness() const { return d_trianglesStiffness.getValue(); }
    Real getQuadsStiffness() const { return d_quadsStiffness.getValue(); }
    Real getTetrahedraStiffness() const { return d_tetrahedraStiffness.getValue(); }
    Real getCubesStiffness() const { return d_cubesStiffness.getValue(); }
    void setStiffness(Real val)
    {
        d_linesStiffness.setValue(val);
        d_trianglesStiffness.setValue(val);
        d_quadsStiffness.setValue(val);
        d_tetrahedraStiffness.setValue(val);
        d_cubesStiffness.setValue(val);
    }
    void setLinesStiffness(Real val)
    {
        d_linesStiffness.setValue(val);
    }
    void setTrianglesStiffness(Real val)
    {
        d_trianglesStiffness.setValue(val);
    }
    void setQuadsStiffness(Real val)
    {
        d_quadsStiffness.setValue(val);
    }
    void setTetrahedraStiffness(Real val)
    {
        d_tetrahedraStiffness.setValue(val);
    }
    void setCubesStiffness(Real val)
    {
        d_cubesStiffness.setValue(val);
    }

    Real getDamping() const { return d_linesDamping.getValue(); }
    Real getLinesDamping() const { return d_linesDamping.getValue(); }
    Real getTrianglesDamping() const { return d_trianglesDamping.getValue(); }
    Real getQuadsDamping() const { return d_quadsDamping.getValue(); }
    Real getTetrahedraDamping() const { return d_tetrahedraDamping.getValue(); }
    Real getCubesDamping() const { return d_cubesDamping.getValue(); }
    void setDamping(Real val)
    {
        d_linesDamping.setValue(val);
        d_trianglesDamping.setValue(val);
        d_quadsDamping.setValue(val);
        d_tetrahedraDamping.setValue(val);
        d_cubesDamping.setValue(val);
    }
    void setLinesDamping(Real val)
    {
        d_linesDamping.setValue(val);
    }
    void setTrianglesDamping(Real val)
    {
        d_trianglesDamping.setValue(val);
    }
    void setQuadsDamping(Real val)
    {
        d_quadsDamping.setValue(val);
    }
    void setTetrahedraDamping(Real val)
    {
        d_tetrahedraDamping.setValue(val);
    }
    void setCubesDamping(Real val)
    {
        d_cubesDamping.setValue(val);
    }

    void init() override;

    void draw(const core::visual::VisualParams* vparams) override;
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_MESHSPRINGFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API MeshSpringForceField<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API MeshSpringForceField<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API MeshSpringForceField<defaulttype::Vec1Types>;

#endif

} // namespace sofa::component::solidmechanics::spring
