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
#ifndef SOFA_COMPONENT_MASS_MATRIXMASS_H
#define SOFA_COMPONENT_MASS_MATRIXMASS_H



#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/componentmodel/behavior/Mass.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/component/topology/PointData.h>
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
class MatrixMass : public core::componentmodel::behavior::Mass<DataTypes>
{
public:
    typedef core::componentmodel::behavior::Mass<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;

    using Inherited::sout;
    using Inherited::serr;
    using Inherited::sendl;
    typedef helper::vector<MassType> VecMass;


    Data< VecMass > f_mass;
    Data< bool >    _lumped;
    Data< Real >    _defaultValue;

    VecMass _lumpedMasses; ///< lumped mass matrices

    const VecMass* _usedMassMatrices; ///< what VecMass is used to represent matrices ? f_mass.getValue() or _lumpedMasses ?



    MatrixMass()
        :  f_mass( initData(&f_mass, "massMatrices", "values of the particles masses") )
        , _defaultValue( initData(&_defaultValue, (Real)1.0,"defaultValue", "real default value") )
        ,_usingDefaultDiagonalMatrices(false)
    {
        _lumped = initData( &this->_lumped, false, "lumped", "");
    };

    ~MatrixMass();


    void clear();
    //void addMass(const MassType& mass);
    void resize(int vsize);


    virtual void init();
    virtual void reinit();


    // -- Mass interface
    void addMDx(VecDeriv& f, const VecDeriv& dx, double factor = 1.0);

    void accFromF(VecDeriv& a, const VecDeriv& f);

    void addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    void addGravityToV(double dt/*, defaulttype::BaseVector& v*/);

    double getKineticEnergy(const VecDeriv& v);  ///< vMv/2 using dof->getV()

    double getPotentialEnergy(const VecCoord& x);   ///< Mgx potential in a uniform gravity field, null at origin

    /// Add Mass contribution to global Matrix assembling
    void addMToMatrix(defaulttype::BaseMatrix * mat, double mFact, unsigned int &offset);

    double getElementMass(unsigned int index) const;
    void getElementMass(unsigned int index, defaulttype::BaseMatrix *m) const;

protected:
    MassType diagonalMass( const Real& m ); ///< return a diagonal matrix mass with value m on all the diagonal
    MassType lump( const MassType& m ); ///< lump the matrix m, ie sum line on diagonal
    void lumpMatrices( ); ///< lump all mass matrices
    void defaultDiagonalMatrices( ); ///< compute default diagonal matrices
    bool _usingDefaultDiagonalMatrices; ///< default diagonal matrices are used

};

} // namespace mass

} // namespace component

} // namespace sofa

#endif
