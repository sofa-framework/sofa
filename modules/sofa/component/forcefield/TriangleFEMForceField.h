#ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGLEFEMFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_TRIANGLEFEMFORCEFIELD_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/core/VisualModel.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/SofaBaseMatrix.h>
#include <sofa/defaulttype/SofaBaseVector.h>




namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;


template<class DataTypes>
class TriangleFEMForceField : public core::componentmodel::behavior::ForceField<DataTypes>, public core::VisualModel
{
public:
    typedef core::componentmodel::behavior::ForceField<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;

    typedef topology::MeshTopology::index_type Index;
    typedef topology::MeshTopology::Triangle Element;
    typedef topology::MeshTopology::SeqTriangles VecElement;

    static const int SMALL = 1;										///< Symbol of small displacements triangle solver
    static const int LARGE = 0;										///< Symbol of large displacements triangle solver

protected:
//    component::MechanicalObject<DataTypes>* _object;

    typedef Vec<6, Real> Displacement;								///< the displacement vector

    typedef Mat<3, 3, Real> MaterialStiffness;						///< the matrix of material stiffness
    typedef std::vector<MaterialStiffness> VecMaterialStiffness;    ///< a vector of material stiffness matrices
    VecMaterialStiffness _materialsStiffnesses;						///< the material stiffness matrices vector

    typedef Mat<6, 3, Real> StrainDisplacement;						///< the strain-displacement matrix
    typedef std::vector<StrainDisplacement> VecStrainDisplacement;	///< a vector of strain-displacement matrices
    VecStrainDisplacement _strainDisplacements;						///< the strain-displacement matrices vector

    typedef Mat<3, 3, Real > Transformation;						///< matrix for rigid transformations like rotations


    topology::MeshTopology* _mesh;
    const VecElement *_indexedElements;
    VecCoord _initialPoints; ///< the intial positions of the points
//     int _method; ///< the computation method of the displacements
//     Real _poissonRatio;
//     Real _youngModulus;
//     Real _dampingRatio;

public:

    TriangleFEMForceField();

    //virtual const char* getTypeName() const { return "TriangleFEMForceField"; }

    virtual ~TriangleFEMForceField();


    virtual void init();
    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    virtual void addDForce (VecDeriv& df, const VecDeriv& dx);

    virtual double getPotentialEnergy(const VecCoord& x);


    // -- Temporary added here for matrix ForceField
    void contributeToMatrixDimension(unsigned int * const, unsigned int * const)
    {}
    ;
    void computeMatrix(sofa::defaulttype::SofaBaseMatrix *, double , double , double, unsigned int &)
    {}
    ;
    void computeVector(sofa::defaulttype::SofaBaseVector *, unsigned int &)
    {}
    ;
    void matResUpdatePosition(sofa::defaulttype::SofaBaseVector *, unsigned int & )
    {}
    ;


    // -- VisualModel interface
    void draw();
    void initTextures()
    { }
    ;
    void update()
    { }
    ;

    DataField<int> f_method;
    DataField<Real> f_poisson;
    DataField<Real> f_young;
    DataField<Real> f_damping;

    Real getPoisson() { return f_poisson.getValue(); }
    void setPoisson(Real val) { f_poisson.setValue(val); }
    Real getYoung() { return f_young.getValue(); }
    void setYoung(Real val) { f_young.setValue(val); }
    Real getDamping() { return f_damping.getValue(); }
    void setDamping(Real val) { f_damping.setValue(val); }
    int getMethod() { return f_method.getValue(); }
    void setMethod(int val) { f_method.setValue(val); }

//     component::MechanicalObject<DataTypes>* getObject()
//     {
//         return _object;
//     }

protected :

    /// f += Kx where K is the stiffness matrix and x a displacement
    virtual void applyStiffness( VecCoord& f, Real h, const VecCoord& x );
    void computeStrainDisplacement( StrainDisplacement &J, Coord a, Coord b, Coord c);
    void computeMaterialStiffnesses();
    void computeForce( Displacement &F, const Displacement &Depl, const MaterialStiffness &K, const StrainDisplacement &J );

    ////////////// small displacements method
    void initSmall();
    void accumulateForceSmall( VecCoord& f, const VecCoord & p, Index elementIndex, bool implicit = false );
    void accumulateDampingSmall( VecCoord& f, Index elementIndex );
    void applyStiffnessSmall( VecCoord& f, Real h, const VecCoord& x );

    ////////////// large displacements method
    std::vector< fixed_array <Coord, 3> > _rotatedInitialElements;   ///< The initials positions in its frame
    std::vector< Transformation > _rotations;
    void initLarge();
    void computeRotationLarge( Transformation &r, const VecCoord &p, const Index &a, const Index &b, const Index &c);
    void accumulateForceLarge( VecCoord& f, const VecCoord & p, Index elementIndex, bool implicit=false );
    void accumulateDampingLarge( VecCoord& f, Index elementIndex );
    void applyStiffnessLarge( VecCoord& f, Real h, const VecCoord& x );
};


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
