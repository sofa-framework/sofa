/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_InvertibleFVMForceField_H
#define SOFA_COMPONENT_FORCEFIELD_InvertibleFVMForceField_H

#include <InvertibleFVM/config.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Mat.h>
//#include <sofa/helper/OptionsGroup.h>



namespace sofa
{

namespace component
{

namespace forcefield
{

template<class DataTypes>
class InvertibleFVMForceField;

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class DataTypes>
class InvertibleFVMForceFieldInternalData
{
public:
};


/** Compute Finite Volume forces based on tetrahedral and hexahedral elements.
 * implementation of "invertible FEM..."
*/
template<class DataTypes>
class InvertibleFVMForceField : virtual public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(InvertibleFVMForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecReal VecReal;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

    typedef core::topology::BaseMeshTopology::index_type Index;
    typedef core::topology::BaseMeshTopology::Tetra Tetra;
    typedef core::topology::BaseMeshTopology::SeqTetrahedra VecTetra;



protected:

    /// @name Per tetrahedron data
    /// @{

    /// Displacement vector (deformation of the 4 corners of a tetrahedron)
    typedef defaulttype::VecNoInit<12, Real> Displacement;

    /// Rigid transformation (rotation) matrix
    typedef defaulttype::MatNoInit<3, 3, Real> Transformation;

    /// @}

    helper::vector<Transformation> _rotationsU;
    helper::vector<Transformation> _rotationsV;



    /* typedef std::pair<int,Real> Col_Value;
     typedef vector< Col_Value > CompressedValue;
     typedef vector< CompressedValue > CompressedMatrix;
     CompressedMatrix _stiffnesses;
     SReal m_potentialEnergy;*/



    core::topology::BaseMeshTopology* _mesh;
    const VecTetra *_indexedTetra;

    helper::vector<Transformation> _initialTransformation;
    helper::vector<Transformation> _initialRotation;

    helper::vector<Transformation> _U;
    helper::vector<Transformation> _V;
    helper::vector<defaulttype::Vec<3,Coord> > _b;

    InvertibleFVMForceFieldInternalData<DataTypes> data;
    friend class InvertibleFVMForceFieldInternalData<DataTypes>;

public:


    Data< VecCoord > _initialPoints; ///< the intial positions of the points

    Data<Real> _poissonRatio;
    Data<VecReal > _youngModulus;
    Data<VecReal> _localStiffnessFactor;


    Data< bool > drawHeterogeneousTetra;
    Data< bool > drawAsEdges;

    Data< bool > _verbose;

    Real minYoung;
    Real maxYoung;
protected:
    InvertibleFVMForceField() ;
    virtual ~InvertibleFVMForceField() ;

public:
    void setPoissonRatio(Real val) ;
    void setYoungModulus(Real val) ;

    virtual void reset() override ;
    virtual void init() override ;
    virtual void reinit()override ;

    virtual void addForce(const core::MechanicalParams* mparams,
                          DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override ;

    virtual void addDForce(const core::MechanicalParams* mparams,
                           DataVecDeriv& , const DataVecDeriv& ) override ;

    virtual void addKToMatrix(sofa::defaulttype::BaseMatrix *m, SReal kFactor, unsigned int &offset) override ;

    virtual SReal getPotentialEnergy(const core::MechanicalParams* mparams,
                                     const DataVecCoord&  x) const override ;

    void draw(const core::visual::VisualParams* vparams);
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_InvertibleFVMForceField_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_InvertibleFVM_API InvertibleFVMForceField<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_InvertibleFVM_API InvertibleFVMForceField<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
