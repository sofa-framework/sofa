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

#include <sofa/component/solidmechanics/fem/elastic/config.h>



#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>
#include <sofa/component/solidmechanics/fem/elastic/TriangularFEMForceField.h>
#include <sofa/core/topology/TopologyData.h>

#include <sofa/core/objectmodel/lifecycle/RenamedData.h>

namespace sofa::component::solidmechanics::fem::elastic
{


template<class DataTypes>
class TriangularAnisotropicFEMForceField : public TriangularFEMForceField<DataTypes>
{

public:
    SOFA_CLASS(SOFA_TEMPLATE(TriangularAnisotropicFEMForceField, DataTypes), SOFA_TEMPLATE(TriangularFEMForceField, DataTypes));

    typedef TriangularFEMForceField<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecReal VecReal;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;
    typedef typename Inherited::TriangleInformation   TriangleInformation  ;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    typedef sofa::core::topology::BaseMeshTopology::Index Index;
    typedef sofa::core::topology::BaseMeshTopology::Triangle Element;
    typedef sofa::core::topology::BaseMeshTopology::SeqTriangles VecElement;

    void init() override;
    void reinit() override;
    void draw(const core::visual::VisualParams* vparams) override;
protected:
    TriangularAnisotropicFEMForceField();
    ~TriangularAnisotropicFEMForceField();
public:
    void computeMaterialStiffness(int i, Index& a, Index& b, Index& c) override;
    void getFiberDir(int element, Deriv& dir);

    //Data<Real> f_poisson2;
    //Data<Real> d_young2; ///< Young modulus along transverse direction
    typedef typename TriangularAnisotropicFEMForceField::Deriv TriangleFiberDirection;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<type::vector<Real>> f_young2;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<Real> f_theta;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<VecCoord> f_fiberCenter;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<bool> showFiber;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_SOLIDMECHANICS_FEM_ELASTIC()
    sofa::core::objectmodel::lifecycle::RenamedData<sofa::type::vector< TriangleFiberDirection > > localFiberDirection;


    Data<type::vector<Real> > f_poisson2;
    Data<type::vector<Real> > d_young2; ///< Young modulus along transverse direction
    Data<Real> d_theta; ///< Fiber angle in global reference frame (in degrees)
    Data<VecCoord> d_fiberCenter; ///< Concentric fiber center in global reference frame
    Data<bool> d_showFiber; ///< Flag activating rendering of fiber directions within each triangle
    core::topology::TriangleData < sofa::type::vector< TriangleFiberDirection > > d_localFiberDirection; ///< Computed fibers direction within each triangle

    /// Link to be set to the topology container in the component graph.
    using Inherit1::l_topology;

    /** Method to initialize @sa TriangleFiberDirection when a new Triangle is created.
    * Will be set as creation callback in the TriangleData @sa d_localFiberDirection
    */
    void createTriangleInfo(Index triangleIndex,
        TriangleFiberDirection&,
        const core::topology::BaseMeshTopology::Triangle& t,
        const sofa::type::vector< unsigned int >&,
        const sofa::type::vector< SReal >&);

};

#if !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGULARANISOTROPICFEMFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API TriangularAnisotropicFEMForceField<defaulttype::Vec3Types>;

#endif

} // namespace sofa::component::solidmechanics::fem::elastic
