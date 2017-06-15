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
#ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELD_H
#include "config.h"

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Mat.h>
#include <SofaBaseTopology/TopologyData.h>

#include <map>
#include <sofa/helper/map.h>

namespace sofa
{

namespace component
{

namespace forcefield
{


//#define PLOT_CURVE //lose some FPS


//using sofa::helper::vector;

/** corotational triangle from
* @InProceedings{NPF05,
*   author       = "Nesme, Matthieu and Payan, Yohan and Faure, Fran\c{c}ois",
*   title        = "Efficient, Physically Plausible Finite Elements",
*   booktitle    = "Eurographics (short papers)",
*   month        = "august",
*   year         = "2005",
*   editor       = "J. Dingliana and F. Ganovelli",
*   keywords     = "animation, physical model, elasticity, finite elements",
*   url          = "http://www-evasion.imag.fr/Publications/2005/NPF05"
* }
*/
template<class DataTypes>
class TriangularFEMForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TriangularFEMForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecReal VecReal;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord    Coord   ;
    typedef typename DataTypes::Deriv    Deriv   ;
    typedef typename Coord::value_type   Real    ;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    typedef sofa::core::topology::BaseMeshTopology::index_type Index;
    typedef sofa::core::topology::BaseMeshTopology::Triangle Element;
    typedef sofa::core::topology::BaseMeshTopology::SeqTriangles VecElement;
    typedef sofa::core::topology::BaseMeshTopology::TrianglesAroundVertex TrianglesAroundVertex;

    typedef sofa::helper::Quater<Real> Quat;

    enum {
        LARGE = 0,   ///< Symbol of small displacements triangle solver
        SMALL = 1,   ///< Symbol of large displacements triangle solver
    };

protected:

    bool _anisotropicMaterial;			                 	    /// used to turn on / off optimizations
    typedef defaulttype::Vec<6, Real> Displacement;					    ///< the displacement vector
    typedef defaulttype::Mat<3, 3, Real> MaterialStiffness;				    ///< the matrix of material stiffness
    typedef sofa::helper::vector<MaterialStiffness> VecMaterialStiffness;   ///< a vector of material stiffness matrices
    typedef defaulttype::Mat<6, 3, Real> StrainDisplacement;				    ///< the strain-displacement matrix
    typedef defaulttype::Mat<6, 6, Real> Stiffness;					    ///< the stiffness matrix
    typedef sofa::helper::vector<StrainDisplacement> VecStrainDisplacement; ///< a vector of strain-displacement matrices
    typedef defaulttype::Mat<3, 3, Real > Transformation;				    ///< matrix for rigid transformations like rotations


protected:
    /// ForceField API
    //{
    TriangularFEMForceField();

    virtual ~TriangularFEMForceField();
public:
    virtual void init();
    virtual void reinit();
    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);
    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx);
    virtual SReal getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& x) const;

    void draw(const core::visual::VisualParams* vparams);
    //}

    /// Class to store FEM information on each triangle, for topology modification handling
    class TriangleInformation
    {
    public:
        /// material stiffness matrices of each tetrahedron
        MaterialStiffness materialMatrix;
        ///< the strain-displacement matrices vector
        StrainDisplacement strainDisplacementMatrix;
        ///< the stiffness matrix
        Stiffness stiffness;
        Real area;
        // large displacement method
        helper::fixed_array<Coord,3> rotatedInitialElements;
        Transformation rotation;
        // strain vector
        defaulttype::Vec<3,Real> strain;
        // stress vector
        defaulttype::Vec<3,Real> stress;
        Transformation initialTransformation;
        Coord principalStressDirection;
        Real maxStress;
        Coord principalStrainDirection;
        Real maxStrain;

        helper::vector<Coord> lastNStressDirection;

        TriangleInformation() { }

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const TriangleInformation& /*ti*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, TriangleInformation& /*ti*/ )
        {
            return in;
        }

        Real differenceToCriteria;
    };

    /// Class to store FEM information on each edge, for topology modification handling
    class EdgeInformation
    {
    public:
        EdgeInformation()
            :fracturable(false) {}

        bool fracturable;

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const EdgeInformation& /*ei*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, EdgeInformation& /*ei*/ )
        {
            return in;
        }
    };

    /// Class to store FEM information on each vertex, for topology modification handling
    class VertexInformation
    {
    public:
        VertexInformation()
            :sumEigenValues(0.0), stress(0.0) {}

        Coord meanStrainDirection;
        double sumEigenValues;
        Transformation rotation;

        double stress; //average stress of triangles around (used only for drawing)

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const VertexInformation& /*vi*/)
        {
            return os;
        }
        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, VertexInformation& /*vi*/)
        {
            return in;
        }
    };

    /// Topology Data
    topology::TriangleData<sofa::helper::vector<TriangleInformation> > triangleInfo;
    topology::PointData<sofa::helper::vector<VertexInformation> > vertexInfo;
    topology::EdgeData<sofa::helper::vector<EdgeInformation> > edgeInfo;


    class TRQSTriangleHandler : public topology::TopologyDataHandler<core::topology::BaseMeshTopology::Triangle,helper::vector<TriangleInformation> >
    {
    public:
        TRQSTriangleHandler(TriangularFEMForceField<DataTypes>* _ff, topology::TriangleData<sofa::helper::vector<TriangleInformation> >* _data) : topology::TopologyDataHandler<core::topology::BaseMeshTopology::Triangle, sofa::helper::vector<TriangleInformation> >(_data), ff(_ff) {}

        void applyCreateFunction(unsigned int triangleIndex, TriangleInformation& ,
                const core::topology::BaseMeshTopology::Triangle & t,
                const sofa::helper::vector< unsigned int > &,
                const sofa::helper::vector< double > &);

    protected:
        TriangularFEMForceField<DataTypes>* ff;
    };


    sofa::core::topology::BaseMeshTopology* _topology;

    //const VecElement *_indexedElements;
    //Data< VecCoord > _initialPoints; ///< the intial positions of the points
    //const VecCoord* _initialPoints; //previously stored the mechanical state initial points but use it directly now
    //     int _method; ///< the computation method of the displacements


    /// Get/Set methods
    Real getPoisson() { return (f_poisson.getValue())[0]; }
    void setPoisson(Real val)
    {
        helper::vector<Real> newP(1, val);
        f_poisson.setValue(newP);
    }
    Real getYoung() { return (f_young.getValue())[0]; }
    void setYoung(Real val)
    {
        helper::vector<Real> newY(1, val);
        f_young.setValue(newY);
    }
    Real getDamping() { return f_damping.getValue(); }
    void setDamping(Real val) { f_damping.setValue(val); }
    int  getMethod() { return method; }
    void setMethod(int val) { method = val; }
    void setMethod(std::string methodName) {
        if (methodName == "small")
            this->setMethod(SMALL);
        else
        {
            if (methodName != "large")
                serr << "unknown method: large method will be used. Remark: Available method are \"small\", \"large\" "<<sendl;
            this->setMethod(LARGE);
        }
    }


public:

    int  getFracturedEdge();
    void getFractureCriteria(int element, Deriv& direction, Real& value);
    /// Compute value of stress along a given direction (typically the fiber direction and transverse direction in anisotropic materials)
    void computeStressAlongDirection(Real &stress_along_dir, Index elementIndex, const Coord &direction, const defaulttype::Vec<3,Real> &stress);
    /// Compute value of stress along a given direction (typically the fiber direction and transverse direction in anisotropic materials)
    void computeStressAlongDirection(Real &stress_along_dir, Index elementIndex, const Coord &direction);
    /// Compute value of stress across a given direction (typically the fracture direction)
    void computeStressAcrossDirection(Real &stress_across_dir, Index elementIndex, const Coord &direction, const defaulttype::Vec<3,Real> &stress);
    /// Compute value of stress across a given direction (typically the fracture direction)
    void computeStressAcrossDirection(Real &stress_across_dir, Index elementIndex, const Coord &direction);
    /// Compute current stress
    void computeStress(defaulttype::Vec<3,Real> &stress, Index elementIndex);

    // Getting the rotation of the vertex by averaing the rotation of neighboring elements
    void getRotation(Transformation& R, unsigned int nodeIdx);
    void getRotations();

protected :
    /// Forcefield computations
    void computeDisplacementSmall(Displacement &D, Index elementIndex, const VecCoord &p);
    void computeDisplacementLarge(Displacement &D, Index elementIndex, const Transformation &R_2_0, const VecCoord &p);
    void computeStrainDisplacement(StrainDisplacement &J, Index elementIndex, Coord a, Coord b, Coord c );
    void computeStiffness(StrainDisplacement &J, Stiffness &K, MaterialStiffness &D);
    void computeStrain(defaulttype::Vec<3,Real> &strain, const StrainDisplacement &J, const Displacement &D);
    void computeStress(defaulttype::Vec<3,Real> &stress, MaterialStiffness &K, defaulttype::Vec<3,Real> &strain);
    void computeForce(Displacement &F, Index elementIndex, const VecCoord &p);
    void computePrincipalStrain(Index elementIndex, defaulttype::Vec<3,Real> &strain);
    void computePrincipalStress(Index elementIndex, defaulttype::Vec<3,Real> &stress);


    /// f += Kx where K is the stiffness matrix and x a displacement
    virtual void applyStiffness( VecCoord& f, Real h, const VecCoord& x, const SReal &kFactor );
    virtual void computeMaterialStiffness(int i, Index& a, Index& b, Index& c);

    ////////////// small displacements method
    void initSmall(int i, Index&a, Index&b, Index&c);
    void accumulateForceSmall( VecCoord& f, const VecCoord & p, Index elementIndex);
    void accumulateDampingSmall( VecCoord& f, Index elementIndex );
    void applyStiffnessSmall( VecCoord& f, Real h, const VecCoord& x, const SReal &kFactor );

    ////////////// large displacements method
    //sofa::helper::vector< helper::fixed_array <Coord, 3> > _rotatedInitialElements;   ///< The initials positions in its frame
    //sofa::helper::vector< Transformation > _rotations;
    void initLarge(int i, Index&a, Index&b, Index&c);
    void computeRotationLarge( Transformation &r, const VecCoord &p, const Index &a, const Index &b, const Index &c);
    void accumulateForceLarge( VecCoord& f, const VecCoord & p, Index elementIndex);
    void accumulateDampingLarge( VecCoord& f, Index elementIndex );
    void applyStiffnessLarge( VecCoord& f, Real h, const VecCoord& x, const SReal &kFactor );


    bool updateMatrix;
    int lastFracturedEdgeIndex;

public:

    /// Forcefield intern paramaters
    int method;
    Data<std::string> f_method;
    Data<helper::vector<Real> > f_poisson;
    Data<helper::vector<Real> > f_young;
    Data<Real> f_damping;

    /// Initial strain parameters (if FEM is initialised with predefine values)
    Data< sofa::helper::vector <helper::fixed_array<Coord,3> > > m_rotatedInitialElements;
    Data< sofa::helper::vector <Transformation> > m_initialTransformation;

    /// Fracture parameters
    Data<bool> f_fracturable;
    Data<Real> hosfordExponant;
    Data<Real> criteriaValue;

    /// Display parameters
    Data<bool> showStressValue;
    Data<bool> showStressVector;
    Data<bool> showFracturableTriangles;

    Data<bool> f_computePrincipalStress;

    TRQSTriangleHandler* triangleHandler;

#ifdef PLOT_CURVE
    //structures to save values for each element along time
    sofa::helper::vector<std::map<std::string, sofa::helper::vector<double> > > allGraphStress;
    sofa::helper::vector<std::map<std::string, sofa::helper::vector<double> > > allGraphCriteria;
    sofa::helper::vector<std::map<std::string, sofa::helper::vector<double> > > allGraphOrientation;

    //the index of element we want to display the graphs
    Data<Real>  elementID;

    //data storing the values along time for the element with index elementID
    Data<std::map < std::string, sofa::helper::vector<double> > > f_graphStress;
    Data<std::map < std::string, sofa::helper::vector<double> > > f_graphCriteria;
    Data<std::map < std::string, sofa::helper::vector<double> > > f_graphOrientation;
#endif

};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELD_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_MISC_FEM_API TriangularFEMForceField<defaulttype::Vec3dTypes>;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_MISC_FEM_API TriangularFEMForceField<defaulttype::Vec3fTypes>;
#endif

#endif // defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELD_CPP)

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELD_H
