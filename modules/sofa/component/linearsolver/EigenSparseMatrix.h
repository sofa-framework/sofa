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
#ifndef SOFA_COMPONENT_LINEARSOLVER_EigenSparseMatrix_H
#define SOFA_COMPONENT_LINEARSOLVER_EigenSparseMatrix_H

#include <sofa/component/linearsolver/EigenBaseSparseMatrix.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/SortedPermutation.h>
#include <sofa/helper/vector.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
using std::cerr;
using std::endl;

namespace sofa
{

namespace component
{

namespace linearsolver
{
using helper::vector;

//#define EigenSparseMatrix_CHECK
//#define EigenSparseMatrix_VERBOSE


/** Container of an Eigen::SparseMatrix<Real, RowMajor> matrix, able to perform computations on InDataTypes::VecDeriv and OutDataTypes::VecDeriv vectors.
  The vectors are converted to/from Eigen format during the computations.

  WARNING: Random write is not possible. For efficiency, the filling must be performed per row, column in increasing order.
Method beginRow(int index) must be called before any entry can be appended to row i.
Then set(i,j,value) must be used in for increasing j. There is no need to explicitly end a row.
When all the entries are written, method endEdit() must be applied to finalize the matrix.
  */
template<class InDataTypes, class OutDataTypes>
class EigenSparseMatrix : public EigenBaseSparseMatrix<typename InDataTypes::Real>
{
public:
    typedef EigenBaseSparseMatrix<typename InDataTypes::Real> Inherit;
    typedef typename InDataTypes::Real Real;
    typedef Eigen::SparseMatrix<Real,Eigen::RowMajor> Matrix;
    typedef Eigen::Matrix<Real,Eigen::Dynamic,1>  VectorEigen;

    typedef typename InDataTypes::Deriv InDeriv;
    typedef typename InDataTypes::VecDeriv InVecDeriv;
    typedef typename OutDataTypes::Deriv OutDeriv;
    typedef typename OutDataTypes::VecDeriv OutVecDeriv;
    enum { Nin=InDataTypes::deriv_total_size, Nout=OutDataTypes::deriv_total_size };
    typedef defaulttype::Mat<Nout,Nin,Real> Block;  ///< block relating an OutDeriv to an InDeriv. This is used for input only, not for internal storage.


    EigenSparseMatrix(int nbRow=0, int nbCol=0):Inherit(nbRow,nbCol) {}

    /// Resize the matrix without preserving the data (the matrix is set to zero), with the size given in number of blocks
    void resizeBlocks(int nbBlockRows, int nbBlockCols)
    {
        this->eigenMatrix.resize(nbBlockRows * Nout, nbBlockCols * Nin);
    }


    bool canCast( const InVecDeriv& v ) const
    {
        //        cerr<<"canCast, size = " << v.size() << endl;
        //        cerr<<"canCast, length = " << &v[v.size()-1][0] - &v[0][0] << endl;
        //        cerr<<"canCast, (v.size()-1)*sizeof(InDeriv) = " << (v.size()-1)*sizeof(InDeriv) << endl;
        //        int diff = (&v[v.size()-1][0]-&v[0][0]) * sizeof(Real);
        //        cerr<<"canCast,  diff = " << diff << endl;
        if(  (v.size()-1)*sizeof(InDeriv) ==  (&v[v.size()-1][0]-&v[0][0]) * sizeof(Real)) // contiguous
            return true;
        else return false;

    }

//    bool canCast( const OutVecDeriv& v ) const
//    {
//        if(  (v.size()-1)*sizeof(OutDeriv) ==  (&v[v.size()-1][0]-&v[0][0]) * sizeof(Real)) // contiguous
//            return true;
//        else return false;
//    }

    /** Insert a new row of blocks in the matrix. The rows must be inserted in increasing order. bRow is the row number. brow and bcolumns are block indices.
      Insert one row of scalars after another
    */
    void appendBlockRow(  unsigned bRow, const vector<unsigned>& bColumns, const vector<Block>& blocks )
    {
        vector<unsigned> p = helper::sortedPermutation(bColumns); // indices in ascending column order

        for( unsigned r=0; r<Nout; r++ )   // process one scalar row after another
        {
            beginRow(r+ bRow*Nout);
            for(unsigned i=0; i<p.size(); i++ )  // process the blocks in ascending order
            {
                const Block& b = blocks[p[i]];
                for( unsigned c=0; c<Nin; c++ )
                {
                    this->set( r + bRow*Nout, c + bColumns[p[i]] * Nin, b[r][c]);
                }
            }
        }
    }



    /// compute result = A * data
    void mult( OutVecDeriv& result, const InVecDeriv& data ) const
    {
        // use optimized product if possible
        if(canCast(data))
        {
            const Eigen::Map<VectorEigen> d(const_cast<Real*>(&data[0][0]),data.size()*Nin);
            Eigen::Map<VectorEigen> r(&result[0][0],result.size()*Nout);
            r = this->eigenMatrix * d;
        }
        else
        {

            // convert the data to Eigen type
            VectorEigen aux1(this->colSize(),1), aux2(this->rowSize(),1);
            for(unsigned i=0; i<data.size(); i++)
            {
                for(unsigned j=0; j<Nin; j++)
                    aux1[Nin* i+j] = data[i][j];
            }
            // compute the product
            aux2 = this->eigenMatrix * aux1;
            // convert the result back to the Sofa type
            for(unsigned i=0; i<result.size(); i++)
            {
                for(unsigned j=0; j<Nout; j++)
                    result[i][j] = aux2[Nout* i+j];
            }
        }
    }

    /// compute result = A * data
    void mult( Data<OutVecDeriv>& _result, const Data<InVecDeriv>& _data ) const
    {
        helper::WriteAccessor<Data<OutVecDeriv> > result (_result);
        helper::ReadAccessor<Data<InVecDeriv> > data (_data);

        // use optimized product if possible
        if(canCast(data.ref()))
        {
            const Eigen::Map<VectorEigen> d(const_cast<Real*>(&data[0][0]),data.size()*Nin);
            Eigen::Map<VectorEigen> r(&result[0][0],result.size()*Nout);
            r = this->eigenMatrix * d;
            //            cerr<<"EigenSparseMatrix::mult using maps, in = "<< data << endl;
            //            cerr<<"EigenSparseMatrix::mult using maps, map<in> = "<< d.transpose() << endl;
            //            cerr<<"EigenSparseMatrix::mult using maps, out = "<< result << endl;
            //            cerr<<"EigenSparseMatrix::mult using maps, map<out> = "<< r.transpose() << endl;
            return;
        }

        // convert the data to Eigen type
        VectorEigen aux1(this->colSize(),1), aux2(this->rowSize(),1);
        for(unsigned i=0; i<data.size(); i++)
        {
            for(unsigned j=0; j<Nin; j++)
                aux1[Nin* i+j] = data[i][j];
        }
        // compute the product
        aux2 = this->eigenMatrix * aux1;
        // convert the result back to the Sofa type
        for(unsigned i=0; i<result.size(); i++)
        {
            for(unsigned j=0; j<Nout; j++)
                result[i][j] = aux2[Nout* i+j];
        }
    }

    /// compute result += A * data
    void addMult( OutVecDeriv& result, const InVecDeriv& data ) const
    {
        // use optimized product if possible
        if(canCast(data))
        {
            const Eigen::Map<VectorEigen> d(const_cast<Real*>(&data[0][0]),data.size()*Nin);
            Eigen::Map<VectorEigen> r(&result[0][0],result.size()*Nout);
            r += this->eigenMatrix * d;
            return;
        }

        // convert the data to Eigen type
        VectorEigen aux1(this->colSize()),aux2(this->rowSize());
        for(unsigned i=0; i<data.size(); i++)
        {
            for(unsigned j=0; j<Nin; j++)
                aux1[Nin* i+j] = data[i][j];
        }
        // compute the product
        aux2 = this->eigenMatrix * aux1;
        // convert the result back to the Sofa type
        for(unsigned i=0; i<result.size(); i++)
        {
            for(unsigned j=0; j<Nout; j++)
                result[i][j] += aux2[Nout* i+j];
        }
    }

    /// compute result += A * data
    void addMultTranspose( InVecDeriv& result, const OutVecDeriv& data ) const
    {
        // use optimized product if possible
        if(canCast(result))
        {
            const Eigen::Map<VectorEigen> d(const_cast<Real*>(&data[0][0]),data.size()*Nout);
            Eigen::Map<VectorEigen> r(&result[0][0],result.size()*Nin);
            r += this->eigenMatrix.transpose() * d;
            return;
        }

        // convert the data to Eigen type
        VectorEigen aux1(this->rowSize()),aux2(this->colSize());
        for(unsigned i=0; i<data.size(); i++)
        {
            for(unsigned j=0; j<Nout; j++)
                aux1[Nout* i+j] = data[i][j];
        }
        // compute the product
        aux2 = this->eigenMatrix.transpose() * aux1;
        // convert the result back to the Sofa type
        for(unsigned i=0; i<result.size(); i++)
        {
            for(unsigned j=0; j<Nin; j++)
                result[i][j] += aux2[Nin* i+j];
        }
    }
    /// compute result += A * data
    void addMultTranspose( Data<InVecDeriv>& result, const Data<OutVecDeriv>& data ) const
    {
        helper::WriteAccessor<Data<InVecDeriv> > res (result);
        helper::ReadAccessor<Data<OutVecDeriv> > dat (data);


        // use optimized product if possible
        if(canCast(res.wref()))
        {
            const Eigen::Map<VectorEigen> d(const_cast<Real*>(&dat[0][0]),dat.size()*Nout);
            Eigen::Map<VectorEigen> r(&res[0][0],res.size()*Nin);
            r += this->eigenMatrix.transpose() * d;
            return;
        }


        // convert the data to Eigen type
        VectorEigen aux1(this->rowSize()),aux2(this->colSize());
        for(unsigned i=0; i<dat.size(); i++)
        {
            for(unsigned j=0; j<Nout; j++)
                aux1[Nout* i+j] = dat[i][j];
        }
        // compute the product
        aux2 = this->eigenMatrix.transpose() * aux1;
        // convert the result back to the Sofa type
        for(unsigned i=0; i<res.size(); i++)
        {
            for(unsigned j=0; j<Nin; j++)
                res[i][j] += aux2[Nin* i+j];
        }
    }


    static const char* Name();


};

template<> inline const char* EigenSparseMatrix<defaulttype::Vec3dTypes, defaulttype::Vec1dTypes >::Name() { return "EigenSparseMatrix3d1d"; }
template<> inline const char* EigenSparseMatrix<defaulttype::Vec3fTypes, defaulttype::Vec1fTypes >::Name() { return "EigenSparseMatrix3f1f"; }







/** Specialization for the case where In and Out have different Real types. Just a quick fix, to be improved.
  */
template<class InDataTypes>
class EigenSparseMatrix<InDataTypes,defaulttype::ExtVec3fTypes> : public EigenBaseSparseMatrix<typename InDataTypes::Real>
{
public:
    typedef EigenBaseSparseMatrix<typename InDataTypes::Real> Inherit;
    typedef typename InDataTypes::Real InReal;
    typedef defaulttype::ExtVec3fTypes OutDataTypes;
    typedef typename OutDataTypes::Real OutReal;
    typedef Eigen::DynamicSparseMatrix<InReal> Matrix;
    typedef Eigen::Matrix<InReal,Eigen::Dynamic,1>  VectorEigen;
    typedef Eigen::Matrix<InReal,Eigen::Dynamic,1>  InVectorEigen;
    typedef Eigen::Matrix<OutReal,Eigen::Dynamic,1>  OutVectorEigen;


    typedef typename InDataTypes::Deriv InDeriv;
    typedef typename InDataTypes::VecDeriv InVecDeriv;
    typedef typename OutDataTypes::Deriv OutDeriv;
    typedef typename OutDataTypes::VecDeriv OutVecDeriv;
    enum { Nin=InDataTypes::deriv_total_size, Nout=OutDataTypes::deriv_total_size };
    typedef defaulttype::Mat<Nout,Nin,InReal> Block;  ///< block relating an OutDeriv to an InDeriv. This is used for input only, not for internal storage.


    EigenSparseMatrix(int nRow=0, int nCol=0):Inherit(nRow,nCol) {}


    /// Resize the matrix without preserving the data (the matrix is set to zero), with the size given in number of blocks
    void resizeBlocks(int nbBlockRows, int nbBlockCols)
    {
        this->eigenMatrix.resize(nbBlockRows * Nout, nbBlockCols * Nin);
    }

    bool canCast( const InVecDeriv& v ) const
    {
        //        cerr<<"canCast, size = " << v.size() << endl;
        //        cerr<<"canCast, length = " << &v[v.size()-1][0] - &v[0][0] << endl;
        //        cerr<<"canCast, (v.size()-1)*sizeof(InDeriv) = " << (v.size()-1)*sizeof(InDeriv) << endl;
        //        int diff = (&v[v.size()-1][0]-&v[0][0]) * sizeof(Real);
        //        cerr<<"canCast,  diff = " << diff << endl;
        if(  (v.size()-1)*sizeof(InDeriv) ==  (&v[v.size()-1][0]-&v[0][0]) * sizeof(InReal)) // contiguous
            return true;
        else return false;

    }

    /** Insert a new row of blocks in the matrix. The rows must be inserted in increasing order. bRow is the row number. brow and bcolumns are block indices.
      Insert one row of scalars after another
    */
    void appendBlockRow(  unsigned bRow, const vector<unsigned>& bColumns, const vector<Block>& blocks )
    {
        vector<unsigned> p = helper::sortedPermutation(bColumns); // indices in ascending column order

        for( unsigned r=0; r<Nout; r++ )   // process one scalar row after another
        {
            beginRow(r+ bRow*Nout);
            for(unsigned i=0; i<p.size(); i++ )  // process the blocks in ascending order
            {
                const Block& b = blocks[p[i]];
                for( unsigned c=0; c<Nin; c++ )
                {
                    this->set( r + bRow*Nout, c + bColumns[p[i]] * Nin, b[r][c]);
                }
            }
        }
    }



    /// compute result = A * data
    void mult( OutVecDeriv& result, const InVecDeriv& data ) const
    {
//        // use optimized product if possible
//        if(canCast(data)){
//            const Eigen::Map<VectorEigen> d(const_cast<Real*>(&data[0][0]),data.size()*Nin);
//            Eigen::Map<VectorEigen> r(&result[0][0],result.size()*Nout);
//            r = eigenMatrix * d;
//        }
//        else {

        // convert the data to Eigen type
        VectorEigen aux1(this->colSize(),1), aux2(this->rowSize(),1);
        for(unsigned i=0; i<data.size(); i++)
        {
            for(unsigned j=0; j<Nin; j++)
                aux1[Nin* i+j] = data[i][j];
        }
        // compute the product
        aux2 = this->eigenMatrix * aux1;
        // convert the result back to the Sofa type
        for(unsigned i=0; i<result.size(); i++)
        {
            for(unsigned j=0; j<Nout; j++)
                result[i][j] = aux2[Nout* i+j];
        }
//        }
    }

    /// compute result = A * data
    void mult( Data<OutVecDeriv>& _result, const Data<InVecDeriv>& _data ) const
    {
        helper::WriteAccessor<Data<OutVecDeriv> > result (_result);
        helper::ReadAccessor<Data<InVecDeriv> > data (_data);

//        // use optimized product if possible
//        if(canCast(data.ref())){
//            const Eigen::Map<VectorEigen> d(const_cast<Real*>(&data[0][0]),data.size()*Nin);
//            Eigen::Map<VectorEigen> r(&result[0][0],result.size()*Nout);
//            r = eigenMatrix * d;
//            //            cerr<<"EigenSparseMatrix::mult using maps, in = "<< data << endl;
//            //            cerr<<"EigenSparseMatrix::mult using maps, map<in> = "<< d.transpose() << endl;
//            //            cerr<<"EigenSparseMatrix::mult using maps, out = "<< result << endl;
//            //            cerr<<"EigenSparseMatrix::mult using maps, map<out> = "<< r.transpose() << endl;
//            return;
//        }

        // convert the data to Eigen type
        VectorEigen aux1(this->colSize(),1), aux2(this->rowSize(),1);
        for(unsigned i=0; i<data.size(); i++)
        {
            for(unsigned j=0; j<Nin; j++)
                aux1[Nin* i+j] = data[i][j];
        }
        // compute the product
        aux2 = this->eigenMatrix * aux1;
        // convert the result back to the Sofa type
        for(unsigned i=0; i<result.size(); i++)
        {
            for(unsigned j=0; j<Nout; j++)
                result[i][j] = aux2[Nout* i+j];
        }
    }

    /// compute result += A * data
    void addMult( OutVecDeriv& result, const InVecDeriv& data ) const
    {
//        // use optimized product if possible
//        if(canCast(data)){
//            const Eigen::Map<VectorEigen> d(const_cast<Real*>(&data[0][0]),data.size()*Nin);
//            Eigen::Map<VectorEigen> r(&result[0][0],result.size()*Nout);
//            r += eigenMatrix * d;
//            return;
//        }

        // convert the data to Eigen type
        VectorEigen aux1(this->colSize()),aux2(this->rowSize());
        for(unsigned i=0; i<data.size(); i++)
        {
            for(unsigned j=0; j<Nin; j++)
                aux1[Nin* i+j] = data[i][j];
        }
        // compute the product
        aux2 = this->eigenMatrix * aux1;
        // convert the result back to the Sofa type
        for(unsigned i=0; i<result.size(); i++)
        {
            for(unsigned j=0; j<Nout; j++)
                result[i][j] += aux2[Nout* i+j];
        }
    }

    /// compute result += A * data
    void addMultTranspose( InVecDeriv& result, const OutVecDeriv& data ) const
    {
//        // use optimized product if possible
//        if(canCast(result)){
//            const Eigen::Map<VectorEigen> d(const_cast<Real*>(&data[0][0]),data.size()*Nout);
//            Eigen::Map<VectorEigen> r(&result[0][0],result.size()*Nin);
//            r += eigenMatrix.transpose() * d;
//            return;
//        }

        // convert the data to Eigen type
        VectorEigen aux1(this->rowSize()),aux2(this->colSize());
        for(unsigned i=0; i<data.size(); i++)
        {
            for(unsigned j=0; j<Nout; j++)
                aux1[Nout* i+j] = data[i][j];
        }
        // compute the product
        aux2 = this->eigenMatrix.transpose() * aux1;
        // convert the result back to the Sofa type
        for(unsigned i=0; i<result.size(); i++)
        {
            for(unsigned j=0; j<Nin; j++)
                result[i][j] += aux2[Nin* i+j];
        }
    }
    /// compute result += A * data
    void addMultTranspose( Data<InVecDeriv>& result, const Data<OutVecDeriv>& data ) const
    {
        helper::WriteAccessor<Data<InVecDeriv> > res (result);
        helper::ReadAccessor<Data<OutVecDeriv> > dat (data);


//        // use optimized product if possible
//        if(canCast(res.wref())){
//            const Eigen::Map<VectorEigen> d(const_cast<Real*>(&dat[0][0]),dat.size()*Nout);
//            Eigen::Map<VectorEigen> r(&res[0][0],res.size()*Nin);
//            r += eigenMatrix.transpose() * d;
//            return;
//        }


        // convert the data to Eigen type
        VectorEigen aux1(this->rowSize()),aux2(this->colSize());
        for(unsigned i=0; i<dat.size(); i++)
        {
            for(unsigned j=0; j<Nout; j++)
                aux1[Nout* i+j] = dat[i][j];
        }
        // compute the product
        aux2 = this->eigenMatrix.transpose() * aux1;
        // convert the result back to the Sofa type
        for(unsigned i=0; i<res.size(); i++)
        {
            for(unsigned j=0; j<Nin; j++)
                res[i][j] += aux2[Nin* i+j];
        }
    }


//    static const char* Name();


};



} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
