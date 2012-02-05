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
#ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELDOPTIM_H
#define SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELDOPTIM_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/component/component.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/component/topology/TopologyData.h>
#include <sofa/component/linearsolver/CompressedRowSparseMatrix.h>

#include <map>
#include <sofa/helper/map.h>

namespace sofa
{

namespace component
{

namespace forcefield
{


//#define PLOT_CURVE //lose some FPS


using namespace sofa::defaulttype;
using sofa::helper::vector;
using namespace sofa::component::topology;

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
class SOFA_SIMPLE_FEM_API TriangularFEMForceFieldOptim : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TriangularFEMForceFieldOptim, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

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

protected:

//    typedef Vec<6, Real> Displacement;					    ///< the displacement vector
//    typedef Mat<3, 3, Real> MaterialStiffness;				    ///< the matrix of material stiffness
//    typedef sofa::helper::vector<MaterialStiffness> VecMaterialStiffness;   ///< a vector of material stiffness matrices
//    typedef Mat<6, 3, Real> StrainDisplacement;				    ///< the strain-displacement matrix
//    typedef Mat<6, 6, Real> Stiffness;					    ///< the stiffness matrix
//    typedef sofa::helper::vector<StrainDisplacement> VecStrainDisplacement; ///< a vector of strain-displacement matrices
//    typedef Mat<3, 3, Real > Transformation;				    ///< matrix for rigid transformations like rotations
    typedef Mat<2, 3, Real > Transformation;				    ///< matrix for rigid transformations like rotations
    enum { DerivSize = DataTypes::deriv_total_size };
    typedef Mat<DerivSize, DerivSize, Real> MatBloc;

protected:
    /// ForceField API
    //{
    TriangularFEMForceFieldOptim();

    virtual ~TriangularFEMForceFieldOptim();
public:
    virtual void init();
    virtual void reinit();
    virtual void addForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v);
    virtual void addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& df, const DataVecDeriv& dx);
    virtual void addKToMatrix(sofa::defaulttype::BaseMatrix *m, SReal kFactor, unsigned int &offset);
    virtual double getPotentialEnergy(const core::MechanicalParams* mparams /* PARAMS FIRST */, const DataVecCoord& x) const;

    void draw(const core::visual::VisualParams* vparams);
    //}

    // parse method attribute (for compatibility with non-optimized version)
    void parse ( sofa::core::objectmodel::BaseObjectDescription* arg )
    {
        const char* method = arg->getAttribute("method");
        if (method && *method && std::string(method) != std::string("large"))
        {
            serr << "Attribute method was specified as \""<<method<<"\" while this version only implements the \"large\" method. Ignoring..." << sendl;
        }
        Inherited::parse(arg);
    }

    /// Class to store FEM information on each triangle, for topology modification handling
    class TriangleInfo
    {
    public:
        //Index ia, ib, ic;
        Real bx, cx, cy, ss_factor;
        Transformation init_frame; // Mat<2,3,Real>

        TriangleInfo() { }

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const TriangleInfo& ti )
        {
            return os << "bx= " << ti.bx << " cx= " << ti.cx << " cy= " << ti.cy << " ss_factor= " << ti.ss_factor << " init_frame= " << ti.init_frame << " END";
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, TriangleInfo& ti )
        {
            std::string str;
            while (in >> str)
            {
                if (str == "END") break;
                else if (str == "bx=") in >> ti.bx;
                else if (str == "cx=") in >> ti.cx;
                else if (str == "cy=") in >> ti.cy;
                else if (str == "ss_factor=") in >> ti.ss_factor;
                else if (str == "init_frame=") in >> ti.init_frame;
                else if (!str.empty() && str[str.length()-1]=='=') in >> str; // unknown value
            }
            return in;
        }
    };
    Real gamma, mu;

    class TriangleState
    {
    public:
        Transformation frame; // Mat<2,3,Real>

        TriangleState() { }

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const TriangleState& ti )
        {
            return os << "frame= " << ti.frame << " END";
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, TriangleState& ti )
        {
            std::string str;
            while (in >> str)
            {
                if (str == "END") break;
                else if (str == "frame=") in >> ti.frame;
                else if (!str.empty() && str[str.length()-1]=='=') in >> str; // unknown value
            }
            return in;
        }
    };

    /// Class to store FEM information on each edge, for topology modification handling
    class EdgeInfo
    {
    public:
        bool fracturable;

        EdgeInfo()
            : fracturable(false) { }

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const EdgeInfo& /*ei*/ )
        {
            return os;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, EdgeInfo& /*ei*/ )
        {
            return in;
        }
    };

    /// Class to store FEM information on each vertex, for topology modification handling
    class VertexInfo
    {
    public:
        VertexInfo()
        /*:sumEigenValues(0.0)*/ {}

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const VertexInfo& /*vi*/)
        {
            return os;
        }
        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, VertexInfo& /*vi*/)
        {
            return in;
        }
    };

    /// Topology Data
    TriangleData<sofa::helper::vector<TriangleInfo> > triangleInfo;
    TriangleData<sofa::helper::vector<TriangleState> > triangleState;
    PointData<sofa::helper::vector<VertexInfo> > vertexInfo;
    EdgeData<sofa::helper::vector<EdgeInfo> > edgeInfo;


    class TFEMFFOTriangleInfoHandler : public TopologyDataHandler<Triangle,vector<TriangleInfo> >
    {
    public:
        TFEMFFOTriangleInfoHandler(TriangularFEMForceFieldOptim<DataTypes>* _ff, TriangleData<sofa::helper::vector<TriangleInfo> >* _data) : TopologyDataHandler<Triangle, sofa::helper::vector<TriangleInfo> >(_data), ff(_ff) {}

        void applyCreateFunction(unsigned int triangleIndex, TriangleInfo& ,
                const Triangle & t,
                const sofa::helper::vector< unsigned int > &,
                const sofa::helper::vector< double > &);

    protected:
        TriangularFEMForceFieldOptim<DataTypes>* ff;
    };
    void initTriangleInfo(unsigned int triangleIndex, TriangleInfo& ti, const Triangle t, const VecCoord& x0);
    void initTriangleState(unsigned int triangleIndex, TriangleState& ti, const Triangle t, const VecCoord& x);

    void computeTriangleRotation(Transformation& result, Coord eab, Coord eac);
    void computeTriangleRotation(Transformation& result, Coord a, Coord b, Coord c)
    {
        computeTriangleRotation(result,b-a,c-a);
    }
    void computeTriangleRotation(Transformation& result, VecCoord& x0, Triangle t)
    {
        computeTriangleRotation(result,x0[t[0]], x0[t[1]], x0[t[2]]);
    }

    class TFEMFFOTriangleStateHandler : public TopologyDataHandler<Triangle,vector<TriangleState> >
    {
    public:
        TFEMFFOTriangleStateHandler(TriangularFEMForceFieldOptim<DataTypes>* _ff, TriangleData<sofa::helper::vector<TriangleState> >* _data) : TopologyDataHandler<Triangle, sofa::helper::vector<TriangleState> >(_data), ff(_ff) {}

        void applyCreateFunction(unsigned int triangleIndex, TriangleState& ,
                const Triangle & t,
                const sofa::helper::vector< unsigned int > &,
                const sofa::helper::vector< double > &);

    protected:
        TriangularFEMForceFieldOptim<DataTypes>* ff;
    };

    void initTriangleState(unsigned int triangleIndex, TriangleState& ti, const Triangle t);

    sofa::core::topology::BaseMeshTopology* _topology;

    template<class MatrixWriter>
    void addKToMatrixT(MatrixWriter m, Real kFactor);

    class BaseMatrixWriter
    {
        BaseMatrix* m;
        unsigned int offset;
    public:
        BaseMatrixWriter(BaseMatrix* m, unsigned int offset) : m(m), offset(offset) {}
        void add(unsigned int bi, unsigned int bj, const MatBloc& b)
        {
            unsigned int i0 = offset + bi*DerivSize;
            unsigned int j0 = offset + bj*DerivSize;
            for (unsigned int i=0; i<DerivSize; ++i)
                for (unsigned int j=0; j<DerivSize; ++j)
                    m->add(i0+i,j0+j,b[i][j]);
        }
    };

    class BlocBaseMatrixWriter
    {
        BaseMatrix* m;
        unsigned int boffset;
    public:
        BlocBaseMatrixWriter(BaseMatrix* m, unsigned int boffset) : m(m), boffset(boffset) {}
        void add(unsigned int bi, unsigned int bj, const MatBloc& b)
        {
            unsigned int i0 = boffset + bi;
            unsigned int j0 = boffset + bj;
            m->blocAdd(i0,j0,b.ptr());
        }
    };

    template<class MReal>
    class BlocCRSMatrixWriter
    {
        sofa::component::linearsolver::CompressedRowSparseMatrix<defaulttype::Mat<DerivSize,DerivSize,MReal> >* m;
        unsigned int boffset;
    public:
        BlocCRSMatrixWriter(sofa::component::linearsolver::CompressedRowSparseMatrix<defaulttype::Mat<DerivSize,DerivSize,MReal> >* m, unsigned int boffset) : m(m), boffset(boffset) {}
        void add(unsigned int bi, unsigned int bj, const MatBloc& b)
        {
            unsigned int i0 = boffset + bi;
            unsigned int j0 = boffset + bj;
            //defaulttype::Mat<DerivSize,DerivSize,MReal> bconv = b;
            *m->wbloc(i0,j0,true) += b;
        }
    };

    template<class MReal>
    class CRSMatrixWriter
    {
        sofa::component::linearsolver::CompressedRowSparseMatrix<MReal>* m;
        unsigned int offset;
    public:
        CRSMatrixWriter(sofa::component::linearsolver::CompressedRowSparseMatrix<MReal>* m, unsigned int offset) : m(m), offset(offset) {}
        void add(unsigned int bi, unsigned int bj, const MatBloc& b)
        {
            unsigned int i0 = offset + bi*DerivSize;
            unsigned int j0 = offset + bj*DerivSize;
            for (unsigned int i=0; i<DerivSize; ++i)
                for (unsigned int j=0; j<DerivSize; ++j)
                    *m->wbloc(i0+i,j0+j,true) += (MReal)b[i][j];
        }
    };

public:

    /// Forcefield intern paramaters
    Data<Real> f_poisson;
    Data<Real> f_young;
    Data<Real> f_damping;

    /// Display parameters
    Data<bool> showStressValue;
    Data<bool> showStressVector;

    TFEMFFOTriangleInfoHandler* triangleInfoHandler;
    TFEMFFOTriangleStateHandler* triangleStateHandler;

};


#if defined(WIN32) && !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELDOPTIM_CPP)
#pragma warning(disable : 4231)

#ifndef SOFA_FLOAT
extern template class SOFA_SIMPLE_FEM_API TriangularFEMForceFieldOptim<defaulttype::Vec3dTypes>;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_SIMPLE_FEM_API TriangularFEMForceFieldOptim<defaulttype::Vec3fTypes>;
#endif

#endif // defined(WIN32) && !defined(SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELDOPTIM_CPP)

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_TRIANGULARFEMFORCEFIELDOPTIM_H
