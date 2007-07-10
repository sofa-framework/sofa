#ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGULARQUADRATICSPRINGSFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_TRIANGULARQUADRATICSPRINGSFORCEFIELD_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/core/VisualModel.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/component/topology/TriangleData.h>
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
class TriangularQuadraticSpringsForceField : public core::componentmodel::behavior::ForceField<DataTypes>, public core::VisualModel
{
public:
    typedef core::componentmodel::behavior::ForceField<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;


    class Mat3 : public fixed_array<Deriv,3>
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
    };

    TriangleData<TriangleRestInformation> triangleInfo;
    EdgeData<EdgeRestInformation> edgeInfo;

    TriangleSetTopology<DataTypes> * _mesh;
    VecCoord _initialPoints;										///< the intial positions of the points

    bool updateMatrix;

    DataField<Real> f_poissonRatio;
    DataField<Real> f_youngModulus;
    DataField<Real> f_dampingRatio;
    DataField<bool> f_useAngularSprings; // whether angular springs should be included

    Real lambda;  /// first Lamé coefficient
    Real mu;    /// second Lamé coefficient
public:

    TriangularQuadraticSpringsForceField();

    TriangleSetTopology<DataTypes> *getTriangularTopology() const {return _mesh;}

    virtual ~TriangularQuadraticSpringsForceField();

    virtual double getPotentialEnergy(const VecCoord& x);

    virtual void init();
    virtual void addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v);
    virtual void addDForce(VecDeriv& df, const VecDeriv& dx);

    virtual Real getLambda() const { return lambda;}
    virtual Real getMu() const { return mu;}

    // handle topological changes
    virtual void handleTopologyChange();

    // -- VisualModel interface
    void draw();
    void initTextures() { };
    void update() { };
    /// compute lambda and mu based on the Young modulus and Poisson ratio
    void updateLameCoefficients();



protected :

    EdgeData<EdgeRestInformation> &getEdgeInfo() {return edgeInfo;}

    template <typename DataTypes>
    friend void TRQSTriangleCreationFunction (int , void* ,
            typename TriangularQuadraticSpringsForceField<DataTypes>::TriangleRestInformation &,
            const Triangle& , const std::vector< unsigned int > &, const std::vector< double >&);


    template <typename DataTypes>
    friend void TRQSTriangleDestroyFunction( int , void* , typename TriangularQuadraticSpringsForceField<DataTypes>::TriangleRestInformation &);

};
} //namespace forcefield

} // namespace Components


} // namespace Sofa



#endif /* _TriangularQuadraticSpringsForceField_H_ */
