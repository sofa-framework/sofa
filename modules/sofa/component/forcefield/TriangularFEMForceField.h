/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELD_H

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


// corotational triangle from
// @InProceedings{NPF05,
//   author       = "Nesme, Matthieu and Payan, Yohan and Faure, Fran\c{c}ois",
//   title        = "Efficient, Physically Plausible Finite Elements",
//   booktitle    = "Eurographics (short papers)",
//   month        = "august",
//   year         = "2005",
//   editor       = "J. Dingliana and F. Ganovelli",
//   keywords     = "animation, physical model, elasticity, finite elements",
//   url          = "http://www-evasion.imag.fr/Publications/2005/NPF05"
// }



namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;
using sofa::helper::vector;
using namespace sofa::component::topology;


template<class DataTypes>
class TriangularFEMForceField : public core::componentmodel::behavior::ForceField<DataTypes>, public core::VisualModel
{
public:
    typedef core::componentmodel::behavior::ForceField<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecReal VecReal;
    typedef VecCoord Vector;
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
    typedef sofa::helper::vector<MaterialStiffness> VecMaterialStiffness;    ///< a vector of material stiffness matrices
    //VecMaterialStiffness _materialsStiffnesses;						///< the material stiffness matrices vector

    typedef Mat<6, 3, Real> StrainDisplacement;						///< the strain-displacement matrix
    typedef sofa::helper::vector<StrainDisplacement> VecStrainDisplacement;	///< a vector of strain-displacement matrices
    //VecStrainDisplacement _strainDisplacements;						///< the strain-displacement matrices vector

    typedef Mat<3, 3, Real > Transformation;						///< matrix for rigid transformations like rotations

    class TriangleInformation
    {
    public:
        /// material stiffness matrices of each tetrahedron
        MaterialStiffness materialMatrix;
        ///< the strain-displacement matrices vector
        StrainDisplacement strainDisplacementMatrix;
        // large displacement method
        helper::fixed_array<Coord,3> rotatedInitialElements;
        Transformation rotation;
        /// polar method
        Transformation initialTransformation;

        TriangleInformation()
        {
        }
    };

    TriangleData<TriangleInformation> triangleInfo;

    //EdgeData<EdgeInformation> edgeInfo;

    TriangleSetTopology<DataTypes> * _mesh;
    //const VecElement *_indexedElements;
    Data< VecCoord > _initialPoints; ///< the intial positions of the points
//     int _method; ///< the computation method of the displacements
//     Real _poissonRatio;
//     Real _youngModulus;
//     Real _dampingRatio;

    bool updateMatrix;

public:

    TriangularFEMForceField();

    TriangleSetTopology<DataTypes> *getTriangularTopology() const {return _mesh;}

    //virtual const char* getTypeName() const { return "TriangularFEMForceField"; }

    virtual ~TriangularFEMForceField();


    virtual void init();
    virtual void reinit();
    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    virtual void addDForce (VecDeriv& df, const VecDeriv& dx);

    virtual double getPotentialEnergy(const VecCoord& x);

    // handle topological changes
    virtual void handleTopologyChange();

    // -- VisualModel interface
    void draw();
    void initTextures()
    { }
    ;
    void update()
    { }
    ;

    int method;
    Data<std::string> f_method;
    Data<Real> f_poisson;
    Data<Real> f_young;
    Data<Real> f_damping;

    Real getPoisson() { return f_poisson.getValue(); }
    void setPoisson(Real val) { f_poisson.setValue(val); }
    Real getYoung() { return f_young.getValue(); }
    void setYoung(Real val) { f_young.setValue(val); }
    Real getDamping() { return f_damping.getValue(); }
    void setDamping(Real val) { f_damping.setValue(val); }
    int  getMethod() { return method; }
    void setMethod(int val) { method = val; }

//     component::MechanicalObject<DataTypes>* getObject()
//     {
//         return _object;
//     }

protected :

    //EdgeData<EdgeInformation> &getEdgeInfo() {return edgeInfo;}

    /*
    static void TRQSEdgeCreationFunction(int edgeIndex, void* param, EdgeInformation &ei,
                                         const Edge& ,  const sofa::helper::vector< unsigned int > &,
                                         const sofa::helper::vector< double >&);
    */
    static void TRQSTriangleCreationFunction (int , void* ,
            TriangleInformation &,
            const Triangle& , const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >&);


    //static void TRQSTriangleDestroyFunction ( int , void* , TriangleInformation &);

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
    //sofa::helper::vector< helper::fixed_array <Coord, 3> > _rotatedInitialElements;   ///< The initials positions in its frame
    //sofa::helper::vector< Transformation > _rotations;
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
