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
#ifndef SOFA_COMPONENT_FORCEFIELD_FASTTETRAHEDRALCOROTATIONALFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_FASTTETRAHEDRALCOROTATIONALFORCEFIELD_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/component/topology/TetrahedronData.h>
#include <sofa/component/topology/EdgeData.h>


namespace sofa
{

namespace component
{


namespace forcefield
{

using namespace sofa::defaulttype;
using namespace sofa::component::topology;


template<class DataTypes>
class FastTetrahedralCorotationalForceField : public core::componentmodel::behavior::ForceField<DataTypes>, public virtual core::objectmodel::BaseObject
{
public:
    typedef core::componentmodel::behavior::ForceField<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;
    typedef Mat<3,3,Real>       Mat3x3  ;

    using Inherited::sout;
    using Inherited::serr;
    using Inherited::sendl;

    typedef enum
    {
        POLAR_DECOMPOSITION,
        QR_DECOMPOSITION
    } RotationDecompositionMethod;
protected:


    class EdgeRestInformation
    {
    public:
        Mat3x3 DfDx; /// the edge stiffness matrix
        Mat3x3 reverseDfDx; /// the edge stiffness matrix
        unsigned int v[2];
        Coord restDp;

        EdgeRestInformation()
        {
        }
        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const EdgeRestInformation& /*eri*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, EdgeRestInformation& /*eri*/ )
        {
            return in;
        }
    };
    /// data structure stored for each tetrahedron
    class TetrahedronRestInformation
    {
    public:
        /// shape vector at the rest configuration
        Coord shapeVector[4];
        /// rest volume
        Real restVolume;
        Coord restEdgeVector[6];
        Mat3x3 linearDfDx[6];  // the off-diagonal 3x3 block matrices that makes the 12x12 linear elastic matrix
        Mat3x3 transposedLinearDfDx[6]; // the transposed of those matrices
        Mat3x3 rotation; // rotation from deformed to rest configuration
        Mat3x3 restRotation; // used for QR decomposition
        unsigned int v[4]; // the indices of the 4 vertices

        EdgeRestInformation *edgeInfo[6];  // shortcut to the 6 edge information
        Real edgeOrientation[6];


        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const TetrahedronRestInformation& /*eri*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, TetrahedronRestInformation& /*eri*/ )
        {
            return in;
        }

        TetrahedronRestInformation()
        {
        }
    };
    EdgeData<EdgeRestInformation> edgeInfo;
    TetrahedronData<TetrahedronRestInformation> tetrahedronInfo;


    sofa::core::componentmodel::topology::BaseMeshTopology* _topology;
    VecCoord  _initialPoints;///< the intial positions of the points

    bool updateMatrix;
    bool updateTopologyInfo;

    Data<std::string> f_method; ///< the computation method of the displacements
    RotationDecompositionMethod decompositionMethod;

    Data<Real> f_poissonRatio;
    Data<Real> f_youngModulus;

    Real lambda;  /// first Lame coefficient
    Real mu;    /// second Lame coefficient
public:

    FastTetrahedralCorotationalForceField();

    virtual ~FastTetrahedralCorotationalForceField();

    virtual double  getPotentialEnergy(const VecCoord& x);

    virtual void init();
    virtual void addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v);
    virtual void addDForce(VecDeriv& df, const VecDeriv& dx);

    void updateTopologyInformation();

    virtual Real getLambda() const { return lambda;}
    virtual Real getMu() const { return mu;}

    // handle topological changes
    virtual void handleTopologyChange();

    void setYoungModulus(const double modulus)
    {
        f_youngModulus.setValue((Real)modulus);
    }
    void setPoissonRatio(const double ratio)
    {
        f_poissonRatio.setValue((Real)ratio);
    }
    void setRotationDecompositionMethod( const RotationDecompositionMethod m)
    {
        decompositionMethod=m;
    }
    void draw();
    /// compute lambda and mu based on the Young modulus and Poisson ratio
    void updateLameCoefficients();



protected :

    static void computeQRRotation( Mat3x3 &r, const Coord *dp);

    EdgeData<EdgeRestInformation> &getEdgeInfo() {return edgeInfo;}

    static void CorotationalTetrahedronCreationFunction (int , void* ,
            TetrahedronRestInformation &,
            const Tetrahedron& , const helper::vector< unsigned int > &, const helper::vector< double >&);

};
} //namespace forcefield

} // namespace Components


} // namespace Sofa



#endif /* _TetrahedralTensorMassForceField_H_ */
