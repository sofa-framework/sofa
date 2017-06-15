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
#ifndef SOFA_COMPONENT_FORCEFIELD_FASTTETRAHEDRALCOROTATIONALFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_FASTTETRAHEDRALCOROTATIONALFORCEFIELD_H
#include "config.h"

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/behavior/ForceField.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <SofaBaseTopology/TopologyData.h>


namespace sofa
{

namespace component
{

namespace forcefield
{


template<class DataTypes>
class FastTetrahedralCorotationalForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(FastTetrahedralCorotationalForceField,DataTypes), SOFA_TEMPLATE(core::behavior::ForceField,DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherited;
    typedef typename DataTypes::Real        Real        ;
    typedef typename DataTypes::Coord       Coord       ;
    typedef typename DataTypes::Deriv       Deriv       ;
    typedef typename DataTypes::VecCoord    VecCoord    ;
    typedef typename DataTypes::VecDeriv    VecDeriv    ;
    typedef typename DataTypes::VecReal     VecReal     ;
    typedef Data<VecCoord>                  DataVecCoord;
    typedef Data<VecDeriv>                  DataVecDeriv;

    typedef defaulttype::Mat<3,3,Real>       Mat3x3  ;

    typedef enum
    {
        POLAR_DECOMPOSITION,
        QR_DECOMPOSITION,
	    POLAR_DECOMPOSITION_MODIFIED,
		LINEAR_ELASTIC
    } RotationDecompositionMethod;

    typedef core::topology::BaseMeshTopology::Tetra Tetra;
    typedef core::topology::BaseMeshTopology::EdgesInTetrahedron EdgesInTetrahedron;
    typedef core::topology::BaseMeshTopology::Tetra Tetrahedron;


protected:

    class PointRestInformation
    {
    public:
        Mat3x3 DfDx;  /// the vertex stiffness matrix
        unsigned int v;

        PointRestInformation() {}

        inline friend std::ostream& operator<< ( std::ostream& os, const PointRestInformation& /*eri*/ ) {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, PointRestInformation& /*eri*/ ) {
            return in;
        }
    };

    class EdgeRestInformation
    {
    public:
        Mat3x3 DfDx; /// the edge stiffness matrix
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
        Mat3x3 linearDfDxDiag[4];  // the diagonal 3x3 block matrices that makes the 12x12 linear elastic matrix
        Mat3x3 linearDfDx[6];  // the off-diagonal 3x3 block matrices that makes the 12x12 linear elastic matrix
        Mat3x3 rotation; // rotation from deformed to rest configuration
        Mat3x3 restRotation; // used for QR decomposition
        //unsigned int v[4]; // the indices of the 4 vertices

        PointRestInformation *pointInfo[6]; // shortcut to the 4 vertex information
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

    class FTCFTetrahedronHandler : public topology::TopologyDataHandler<core::topology::BaseMeshTopology::Tetrahedron, sofa::helper::vector<TetrahedronRestInformation> >
    {
    public:
        typedef typename FastTetrahedralCorotationalForceField<DataTypes>::TetrahedronRestInformation TetrahedronRestInformation;

        FTCFTetrahedronHandler(FastTetrahedralCorotationalForceField<DataTypes>* ff,
                topology::TetrahedronData<sofa::helper::vector<TetrahedronRestInformation> >* data )
            :topology::TopologyDataHandler<core::topology::BaseMeshTopology::Tetrahedron, sofa::helper::vector<TetrahedronRestInformation> >(data)
            ,ff(ff)
        {

        }

        void applyCreateFunction(unsigned int, TetrahedronRestInformation &t,
                                 const core::topology::BaseMeshTopology::Tetrahedron&,
                                 const sofa::helper::vector<unsigned int> &,
                                 const sofa::helper::vector<double> &);

    protected:
        FastTetrahedralCorotationalForceField<DataTypes>* ff;

    };

    topology::PointData<sofa::helper::vector<PointRestInformation> > pointInfo;
    topology::EdgeData<sofa::helper::vector<EdgeRestInformation> > edgeInfo;
    topology::TetrahedronData<sofa::helper::vector<TetrahedronRestInformation> > tetrahedronInfo;


    sofa::core::topology::BaseMeshTopology* _topology;
    VecCoord  _initialPoints;///< the intial positions of the points

    bool updateMatrix;
    bool updateTopologyInfo;

    Data<std::string> f_method; ///< the computation method of the displacements
    RotationDecompositionMethod decompositionMethod;

    Data<Real> f_poissonRatio;
    Data<Real> f_youngModulus;

    Real lambda;  /// first Lame coefficient
    Real mu;    /// second Lame coefficient

    FastTetrahedralCorotationalForceField();

    virtual ~FastTetrahedralCorotationalForceField();

public:

    virtual void init();


    virtual void addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv &  dataF, const DataVecCoord &  dataX , const DataVecDeriv & dataV ) ;
    virtual void addDForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv&   datadF , const DataVecDeriv&   datadX ) ;
    virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const
    {
        serr << "Get potentialEnergy not implemented" << sendl;
        return 0.0;
    }

    virtual void addKToMatrix(sofa::defaulttype::BaseMatrix *m, SReal kFactor, unsigned int &offset);
    virtual void addKToMatrix(const core::MechanicalParams* /*mparams*/, const sofa::core::behavior::MultiMatrixAccessor* /*matrix*/ );

    void updateTopologyInformation();

    virtual Real getLambda() const { return lambda;}
    virtual Real getMu() const { return mu;}

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
    void draw(const core::visual::VisualParams* vparams);
    /// compute lambda and mu based on the Young modulus and Poisson ratio
    void updateLameCoefficients();



protected :
    FTCFTetrahedronHandler* tetrahedronHandler;

    static void computeQRRotation( Mat3x3 &r, const Coord *dp);

    topology::EdgeData<sofa::helper::vector<EdgeRestInformation> > &getEdgeInfo() {return edgeInfo;}
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_INTERACTIONFORCEFIELD_FASTTETRAHEDRALCOROTATIONALFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_MISC_FEM_API FastTetrahedralCorotationalForceField<sofa::defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_MISC_FEM_API FastTetrahedralCorotationalForceField<sofa::defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_FASTTETRAHEDRALCOROTATIONALFORCEFIELD_H
