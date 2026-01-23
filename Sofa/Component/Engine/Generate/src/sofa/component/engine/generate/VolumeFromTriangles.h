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
#include <sofa/component/engine/generate/config.h>

#include <sofa/core/DataEngine.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/MechanicalState.h>


namespace sofa::component::engine::generate
{

/**
 * This class computes the volume of a given closed surfacic mesh.
 * Based on the divergence theorem with F(x,y,z)=<x,0,0> so that div(F)=1: https://en.wikipedia.org/wiki/Divergence_theorem
 */
template <class DataTypes>
class VolumeFromTriangles : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(VolumeFromTriangles,DataTypes), sofa::core::DataEngine);

    typedef typename DataTypes::VecCoord VecCoord;

    typedef typename DataTypes::Coord  Coord;
    typedef typename Coord::value_type Real;

    typedef typename sofa::core::behavior::MechanicalState<DataTypes> MechanicalState;
    typedef typename sofa::core::topology::BaseMeshTopology           BaseMeshTopology;

    typedef sofa::core::topology::BaseMeshTopology::Triangle      Triangle;
    typedef sofa::core::topology::BaseMeshTopology::Quad          Quad;

    typedef sofa::core::topology::BaseMeshTopology::SeqTriangles      VecTriangles;
    typedef sofa::core::topology::BaseMeshTopology::SeqQuads          VecQuads;

public:

    VolumeFromTriangles();
    ~VolumeFromTriangles() override;


    ////////////////////////// Inherited from BaseObject ///////////////////
    void init() override;
    void reinit() override;
    void parse(core::objectmodel::BaseObjectDescription* arg) override;

    ////////////////////////////////////////////////////////////////////////

    ////////////////////////// Inherited from DataEngine////////////////////
    void doUpdate() override;
    ///////////////////////////////////////////////////////////////////////

    SReal getVolume() {return d_volume.getValue();}

protected:

    SingleLink<VolumeFromTriangles<DataTypes>, BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;
    SingleLink<VolumeFromTriangles<DataTypes>, MechanicalState, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_state;

    sofa::Data<VecCoord>     d_positions;
    sofa::Data<VecTriangles> d_triangles;
    sofa::Data<VecQuads>     d_quads;

    sofa::Data<Real>         d_volume;
    sofa::Data<bool>         d_doUpdate;

    void updateVolume();

private:

    void initTopology();
    void checkTopology();

};

#if !defined(SOFA_COMPONENT_ENGINE_VOLUMEFROMTRIANGLES_CPP)
extern template class VolumeFromTriangles<sofa::defaulttype::Vec3Types>;
#endif

} // namespace

