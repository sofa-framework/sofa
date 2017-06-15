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
#ifndef SOFA_COMPONENT_MASS_MATRIXMASS_H
#define SOFA_COMPONENT_MASS_MATRIXMASS_H
#include "config.h"

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/Mass.h>
#include <SofaBaseTopology/TopologyData.h>
#include <sofa/helper/vector.h>

namespace sofa
{



namespace component
{

namespace mass
{


/**
Mass Matrices. By default a diagonal matrix is created with diagonal to _defaultValue. Else matrices have to be given in the .scn file.
It is possible to use lumped matrices.
*/

template <class DataTypes, class MassType>
class MatrixMass : public core::behavior::Mass<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(MatrixMass,DataTypes,MassType), SOFA_TEMPLATE(core::behavior::Mass,DataTypes));

    typedef core::behavior::Mass<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    typedef helper::vector<MassType> VecMass;


    Data< VecMass > f_mass;
    Data< bool >    _lumped;
    Data< Real >    _defaultValue;

    VecMass _lumpedMasses; ///< lumped mass matrices

    const VecMass* _usedMassMatrices; ///< what VecMass is used to represent matrices ? f_mass.getValue() or _lumpedMasses ?

protected:
    MatrixMass()
        :  f_mass( initData(&f_mass, "massMatrices", "values of the particles masses") )
        , _lumped(initData( &_lumped, false, "lumped", ""))
        , _defaultValue( initData(&_defaultValue, (Real)1.0,"defaultValue", "real default value") )
        , _usingDefaultDiagonalMatrices(false)
    {
    };

    ~MatrixMass();

public:

    void clear();
    //void addMass(const MassType& mass);
    void resize(int vsize);


    virtual void init();
    virtual void reinit();


    // -- Mass interface
    void addMDx(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecDeriv& dx, SReal factor);

    void accFromF(const core::MechanicalParams* mparams, DataVecDeriv& a, const DataVecDeriv& f);

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);

    void addGravityToV(const core::MechanicalParams* mparams, DataVecDeriv& d_v);

    SReal getKineticEnergy(const core::MechanicalParams* mparams, const DataVecDeriv& v) const;  ///< vMv/2 using dof->getV()

    SReal getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& x) const;   ///< Mgx potential in a uniform gravity field, null at origin

    defaulttype::Vector6 getMomentum(const core::MechanicalParams* mparams, const DataVecCoord& x, const DataVecDeriv& v) const;  ///< (Mv,cross(x,Mv)+Iw)

    /// Add Mass contribution to global Matrix assembling
    //void addMToMatrix(defaulttype::BaseMatrix * mat, SReal mFact, unsigned int &offset);
    void addMToMatrix(const core::MechanicalParams *mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix);

    SReal getElementMass(unsigned int index) const;
    void getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const;

protected:
    MassType diagonalMass( const Real& m ); ///< return a diagonal matrix mass with value m on all the diagonal
    MassType lump( const MassType& m ); ///< lump the matrix m, ie sum line on diagonal
    void lumpMatrices( ); ///< lump all mass matrices
    void defaultDiagonalMatrices( ); ///< compute default diagonal matrices
    bool _usingDefaultDiagonalMatrices; ///< default diagonal matrices are used

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MASS_MATRIXMASS_CPP)
#ifndef SOFA_FLOAT
extern template class MatrixMass<defaulttype::Vec3dTypes, defaulttype::Mat3x3d>;
extern template class MatrixMass<defaulttype::Vec2dTypes, defaulttype::Mat2x2d>;
extern template class MatrixMass<defaulttype::Vec1dTypes, defaulttype::Mat1x1d>;
#endif
#ifndef SOFA_DOUBLE
extern template class MatrixMass<defaulttype::Vec3fTypes, defaulttype::Mat3x3f>;
extern template class MatrixMass<defaulttype::Vec2fTypes, defaulttype::Mat2x2f>;
extern template class MatrixMass<defaulttype::Vec1fTypes, defaulttype::Mat1x1f>;
#endif
#endif

} // namespace mass

} // namespace component

} // namespace sofa

#endif
