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
#ifndef SOFA_COMPONENT_FORCEFIELD_TETRAHEDRALTENSORMASSFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_TETRAHEDRALTENSORMASSFORCEFIELD_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/SofaMisc.h>
#include <sofa/core/behavior/ForceField.h>
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
class TetrahedralTensorMassForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TetrahedralTensorMassForceField,DataTypes), SOFA_TEMPLATE(core::behavior::ForceField,DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;
    typedef typename VecCoord::template rebind<VecCoord>::other VecType;


    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

    class Mat3 : public helper::fixed_array<Deriv,3>
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
        Mat3 DfDx; /// the edge stiffness matrix
        float vertices[2]; // the vertices of this edge

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
    typedef typename VecCoord::template rebind<EdgeRestInformation>::other edgeRestInfoVector;


    sofa::core::topology::BaseMeshTopology* _topology;
    VecCoord  _initialPoints;///< the intial positions of the points

    bool updateMatrix;

    Data<Real> f_poissonRatio;
    Data<Real> f_youngModulus;

    Real lambda;  /// first Lame coefficient
    Real mu;    /// second Lame coefficient

    TetrahedralTensorMassForceField();

    virtual ~TetrahedralTensorMassForceField();

public:

    virtual void init();
    void initNeighbourhoodPoints();

    virtual void addForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);
    virtual void addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_df, const DataVecDeriv& d_dx);
    virtual double getPotentialEnergy(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, const DataVecCoord&  /* x */) const
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
    void draw(const core::visual::VisualParams* vparams);
    /// compute lambda and mu based on the Young modulus and Poisson ratio
    void updateLameCoefficients();


    class TetrahedralTMEdgeHandler : public topology::TopologyDataHandler<topology::Edge,edgeRestInfoVector >
    {
    public:
        typedef typename TetrahedralTensorMassForceField<DataTypes>::EdgeRestInformation EdgeRestInformation;
        TetrahedralTMEdgeHandler(TetrahedralTensorMassForceField<DataTypes>* _ff, topology::EdgeData<edgeRestInfoVector >* _data) : topology::TopologyDataHandler<topology::Edge, edgeRestInfoVector >(_data), ff(_ff) {}

        void applyCreateFunction(unsigned int edgeIndex, EdgeRestInformation& ei,
                const topology::Edge &,
                const sofa::helper::vector< unsigned int > &,
                const sofa::helper::vector< double > &);

        void applyTetrahedronCreation(const sofa::helper::vector<unsigned int> &edgeAdded,
                const sofa::helper::vector<topology::Tetrahedron> &,
                const sofa::helper::vector<sofa::helper::vector<unsigned int> > &,
                const sofa::helper::vector<sofa::helper::vector<double> > &);

        void applyTetrahedronDestruction(const sofa::helper::vector<unsigned int> &edgeRemoved);

        /// Callback to add tetrahedron elements.
        void ApplyTopologyChange(const core::topology::TetrahedraAdded* /*event*/);
        /// Callback to remove tetrahedron elements.
        void ApplyTopologyChange(const core::topology::TetrahedraRemoved* /*event*/);

    protected:
        TetrahedralTensorMassForceField<DataTypes>* ff;
    };

protected:

//    EdgeData < typename VecType < EdgeRestInformation > > edgeInfo;
    topology::EdgeData < edgeRestInfoVector > edgeInfo;

//    EdgeData < typename VecType < EdgeRestInformation > > &getEdgeInfo() {return edgeInfo;}
    topology::EdgeData < edgeRestInfoVector > &getEdgeInfo() {return edgeInfo;}


    TetrahedralTMEdgeHandler* edgeHandler;

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_TETRAHEDRALTENSORMASSFORCEFIELD_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_MISC_FEM_API TetrahedralTensorMassForceField<sofa::defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_MISC_FEM_API TetrahedralTensorMassForceField<sofa::defaulttype::Vec3fTypes>;
#endif

#endif // defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_TETRAHEDRALTENSORMASSFORCEFIELD_CPP)


} //namespace forcefield

} // namespace component

} // namespace sofa



#endif /* _TetrahedralTensorMassForceField_H_ */
