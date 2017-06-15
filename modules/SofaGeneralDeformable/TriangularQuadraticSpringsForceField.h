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
#ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGULARQUADRATICSPRINGSFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_TRIANGULARQUADRATICSPRINGSFORCEFIELD_H
#include "config.h"

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>
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
class TriangularQuadraticSpringsForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TriangularQuadraticSpringsForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    class Mat3 : public sofa::helper::fixed_array<Deriv,3>
    {
    public:
        Deriv operator*(const Deriv& v)
        {
            return Deriv((*this)[0]*v,(*this)[1]*v,(*this)[2]*v);
        }
        Deriv transposeMultiply(const Deriv& v)
        {
            return Deriv(v[0]*((*this)[0])[0]+v[1]*((*this)[1])[0]+v[2]*((*this)[2][0]),
                    v[0]*((*this)[0][1])+v[1]*((*this)[1][1])+v[2]*((*this)[2][1]),
                    v[0]*((*this)[0][2])+v[1]*((*this)[1][2])+v[2]*((*this)[2][2]));
        }
    };

protected:


    class EdgeRestInformation
    {
    public:
        Real  restLength;	// the rest length
        Real  currentLength; 	// the current edge length
        Real  dl;  // the current unit direction
        Real stiffness;

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

    class TriangleRestInformation
    {
    public:
        Real  gamma[3];	// the angular stiffness
        Real stiffness[3]; // the elongation stiffness
        Mat3 DfDx[3]; /// the edge stiffness matrix

        TriangleRestInformation()
        {
        }
        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const TriangleRestInformation& /*tri*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, TriangleRestInformation& /*tri*/ )
        {
            return in;
        }
    };


    sofa::core::topology::BaseMeshTopology* _topology;
    Data< VecCoord > _initialPoints;										///< the intial positions of the points

    bool updateMatrix;

    Data<Real> f_poissonRatio;
    Data<Real> f_youngModulus;
    Data<Real> f_dampingRatio;
    Data<bool> f_useAngularSprings; // whether angular springs should be included

    Real lambda;  /// first Lam� coefficient
    Real mu;    /// second Lam� coefficient


    TriangularQuadraticSpringsForceField();

    virtual ~TriangularQuadraticSpringsForceField();
public:
    virtual void init();

    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);
    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx);
    virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const
    {
        serr << "Get potentialEnergy not implemented" << sendl;
        return 0.0;
    }

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
    void includeAngularSprings(const bool useAngularSprings)
    {
        f_useAngularSprings.setValue(useAngularSprings);
    }

    void draw(const core::visual::VisualParams* vparams);
    /// compute lambda and mu based on the Young modulus and Poisson ratio
    void updateLameCoefficients();


    class TRQSTriangleHandler : public sofa::component::topology::TopologyDataHandler<core::topology::BaseMeshTopology::Triangle,helper::vector<TriangleRestInformation> >
    {
    public:
        typedef typename TriangularQuadraticSpringsForceField<DataTypes>::TriangleRestInformation TriangleRestInformation;
        TRQSTriangleHandler(TriangularQuadraticSpringsForceField<DataTypes>* _ff, sofa::component::topology::TriangleData<sofa::helper::vector<TriangleRestInformation> >* _data) : sofa::component::topology::TopologyDataHandler<core::topology::BaseMeshTopology::Triangle, sofa::helper::vector<TriangleRestInformation> >(_data), ff(_ff) {}

        void applyCreateFunction(unsigned int triangleIndex, TriangleRestInformation& ,
                const core::topology::BaseMeshTopology::Triangle & t,
                const sofa::helper::vector< unsigned int > &,
                const sofa::helper::vector< double > &);

        void applyDestroyFunction(unsigned int, TriangleRestInformation &);

    protected:
        TriangularQuadraticSpringsForceField<DataTypes>* ff;
    };

    class TRQSEdgeHandler : public sofa::component::topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge,helper::vector<EdgeRestInformation> >
    {
    public:
        typedef typename TriangularQuadraticSpringsForceField<DataTypes>::EdgeRestInformation EdgeRestInformation;
        TRQSEdgeHandler(TriangularQuadraticSpringsForceField<DataTypes>* _ff, sofa::component::topology::EdgeData<sofa::helper::vector<EdgeRestInformation> >* _data) : sofa::component::topology::TopologyDataHandler<core::topology::BaseMeshTopology::Edge, sofa::helper::vector<EdgeRestInformation> >(_data), ff(_ff) {}

        void applyCreateFunction(unsigned int edgeIndex, EdgeRestInformation& ,
                const core::topology::BaseMeshTopology::Edge & t,
                const sofa::helper::vector< unsigned int > &,
                const sofa::helper::vector< double > &);

    protected:
        TriangularQuadraticSpringsForceField<DataTypes>* ff;
    };


protected :
    sofa::component::topology::TriangleData<sofa::helper::vector<TriangleRestInformation> > triangleInfo;
    sofa::component::topology::EdgeData<sofa::helper::vector<EdgeRestInformation> > edgeInfo;

    sofa::component::topology::EdgeData<sofa::helper::vector<EdgeRestInformation> > &getEdgeInfo() {return edgeInfo;}

    TRQSTriangleHandler* triangleHandler;
    TRQSEdgeHandler* edgeHandler;

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGULARQUADRATICSPRINGSFORCEFIELD_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_DEFORMABLE_API TriangularQuadraticSpringsForceField<sofa::defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_DEFORMABLE_API TriangularQuadraticSpringsForceField<sofa::defaulttype::Vec3fTypes>;
#endif


#endif // defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGULARQUADRATICSPRINGSFORCEFIELD_CPP)


} //namespace forcefield

} // namespace component

} // namespace sofa

#endif /* SOFA_COMPONENT_FORCEFIELD_TRIANGULARQUADRATICSPRINGSFORCEFIELD_H */
