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

#include "EigenBaseSparseMatrix.h"
#include <sofa/defaulttype/Mat.h>
#include <sofa/component/linearsolver/CompressedRowSparseMatrix.h>
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


/** Variant of EigenBaseSparseMatrix, capable of block-view access.
  The blocks correspond to matrix blocks of the size of the DataTypes Deriv.

  There are two ways of filling the matrix:
  - Random block access is provided by method wBlock. Use compress() after the last insertion.
  - Block rows can be efficiently appended using methods beginBlockRow, createBlock, endBlockRow. Use compress() after the last insertion. The rows must be created in increasing index order.

  The two ways of filling the matrix can not be used at the same time.
  */
template<class InDataTypes, class OutDataTypes>
class EigenSparseMatrix : public EigenBaseSparseMatrix<typename OutDataTypes::Real>
{
public:
    typedef EigenBaseSparseMatrix<typename OutDataTypes::Real> Inherit;
    typedef typename OutDataTypes::Real Real;
    typedef Eigen::SparseMatrix<Real,Eigen::RowMajor> CompressedMatrix;
    typedef Eigen::Matrix<Real,Eigen::Dynamic,1>  VectorEigen;

    typedef typename InDataTypes::Deriv InDeriv;
    typedef typename InDataTypes::VecDeriv InVecDeriv;
    typedef typename InDataTypes::Real InReal;
    typedef typename OutDataTypes::Deriv OutDeriv;
    typedef typename OutDataTypes::VecDeriv OutVecDeriv;
    enum { Nin=InDataTypes::deriv_total_size, Nout=OutDataTypes::deriv_total_size };
    typedef defaulttype::Mat<Nout,Nin,Real> Block;  ///< block relating an OutDeriv to an InDeriv. This is used for input only, not for internal storage.

protected:
    typedef std::map<int,Block> BlockRowMap;        ///< Map which represents one block-view row of the matrix. The index represents the block-view column index of an entry.
    typedef std::map<int,BlockRowMap> BlockMatMap;  ///< Map which represents a block-view matrix. The index represents the block-view index of a block-view row.
    BlockMatMap incomingBlocks;                     ///< To store block-view data before it is compressed in optimized format.
    typedef Eigen::Matrix<InReal,Eigen::Dynamic,1>  VectorEigenIn;

public:

    EigenSparseMatrix(int nbRow=0, int nbCol=0):Inherit(nbRow,nbCol) {}

    /// Resize the matrix without preserving the data (the matrix is set to zero), with the size given in number of blocks
    void resizeBlocks(int nbBlockRows, int nbBlockCols)
    {
        this->compressedMatrix.resize(nbBlockRows * Nout, nbBlockCols * Nin);
    }


    /// Finalize the matrix after a series of insertions. Add the values from the temporary list to the compressed matrix, and clears the list.
    virtual void compress()
    {
        Inherit::compress();

        if( incomingBlocks.empty() ) return;
        compress_incomingBlocks();
        //        cerr<<"compress, before incoming blocks " << this->eigenMatrix << endl;
        //        cerr<<"compress, incoming blocks " << this->compressedIncoming << endl;
        this->compressedMatrix += this->compressedIncoming;
        //        cerr<<"compress, final value " << this->eigenMatrix << endl;
        this->compressedMatrix.finalize();
    }

    /** Return write access to an incoming block.
    Note that this does not give access to the compressed matrix.
    The block belongs to a temporary list which will be added to the compressed matrix using method compress().
    */
    Block& wBlock( int i, int j )
    {
        return incomingBlocks[i][j];
    }


    /** Prepare the insertion of a new row of blocks in the matrix.
       Then create blocks using createBlock( unsigned column,  const Block& b ).
        Then finally use endBlockRow() to validate the row insertion.
        @sa createBlock( unsigned column,  const Block& b )
        @sa endBlockRow()
        */
    void beginBlockRow(unsigned row)
    {
        bRow = row;
        bColumns.clear();
        blocks.clear();
    }

    /** Create a block in the current row, previously initialized using beginBlockRow(unsigned row).
        The blocks need not be created in column order. The blocks are not actually created in the matrix until method endBlockRow() is called.
        */
    void createBlock( unsigned column,  const Block& b )
    {
        blocks.push_back(b);
        bColumns.push_back(column);
    }

    /** Finalize the creation of the current block row.
      @sa beginBlockRow(unsigned row)
      @sa createBlock( unsigned column,  const Block& b )
      */
    void endBlockRow()
    {
        vector<unsigned> p = helper::sortedPermutation(bColumns); // indices in ascending column order

        for( unsigned r=0; r<Nout; r++ )   // process one scalar row after another
        {
            this->beginRow(r+ bRow*Nout);
            for(unsigned i=0; i<p.size(); i++ )  // process the blocks in ascending order
            {
                const Block& b = blocks[p[i]];
                for( unsigned c=0; c<Nin; c++ )
                {
                    if( b[r][c]!=0.0 )
                        this->insertBack( r + bRow*Nout, c + bColumns[p[i]] * Nin, b[r][c]);
                }
            }
        }
    }

    /** Set from a CompressedRowSparseMatrix. @pre crs must be compressed
      */
    void copyFrom( const CompressedRowSparseMatrix<Block>& crs )
    {
        this->resize( crs.rowSize(), crs.colSize() );
//        cerr<<"copyFrom, size " << crs.rowSize() << ", " << crs.colSize()<< ", block rows: " << crs.rowIndex.size() << endl;
//        cerr<<"copyFrom, crs = " << crs << endl;

        int rowStarted = 0;
        for (unsigned int xi = 0; xi < crs.rowIndex.size(); ++xi)  // for each non-null block row
        {
            int blRow = crs.rowIndex[xi];      // block row

            while( rowStarted<blRow*Nout )   // make sure all the rows are started, even the empty ones
            {
                this->compressedMatrix.startVec(rowStarted);
                rowStarted++;
            }

            typename CompressedRowSparseMatrix<Block>::Range rowRange(crs.rowBegin[xi], crs.rowBegin[xi+1]);

            for( unsigned r=0; r<Nout; r++ )   // process one scalar row after another
            {
                if(r+ blRow*Nout >= this->rowSize() ) break;
//                cerr<<"copyFrom,  startVec " << rowStarted << endl;
                this->compressedMatrix.startVec(rowStarted++);


                for (int xj = rowRange.begin(); xj < rowRange.end(); ++xj)  // for each non-null block
                {
                    int blCol = crs.colsIndex[xj];     // block column
                    const Block& b = crs.colsValue[xj]; // block value
                    for( unsigned c=0; c<Nin; c++ ) if( c+ blCol*Nin < this->colSize() )
                        {
                            this->compressedMatrix.insertBack(r + blRow*Nout, c + blCol*Nin) = b[r][c];
//                        cerr<<"copyFrom,  insert at " << r + blRow*Nout << ", " << c + blCol*Nin << endl;
                        }

                }
            }
        }
        this->compress();

    }


    /// compute result = A * data
    void mult( OutVecDeriv& result, const InVecDeriv& data ) const
    {
        // use optimized product if possible
        if(canCast(data))
        {
            const Eigen::Map<VectorEigen> d(const_cast<Real*>(&data[0][0]),data.size()*Nin);
            Eigen::Map<VectorEigen> r(&result[0][0],result.size()*Nout);
            r = this->compressedMatrix * d;
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
            aux2 = this->compressedMatrix * aux1;
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
            typedef Eigen::Map<VectorEigenIn> InVectorMap;
            const InVectorMap d(const_cast<InReal*>(&data[0][0]),data.size()*Nin);
            typedef Eigen::Map<VectorEigen> OutVectorMap;
            OutVectorMap r(&result[0][0],result.size()*Nout);
            r = this->compressedMatrix * d;
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
        aux2 = this->compressedMatrix * aux1;
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
            r += this->compressedMatrix * d;
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
        aux2 = this->compressedMatrix * aux1;
        // convert the result back to the Sofa type
        for(unsigned i=0; i<result.size(); i++)
        {
            for(unsigned j=0; j<Nout; j++)
                result[i][j] += aux2[Nout* i+j];
        }
    }

    /// compute result += A * data
    void addMult( Data<OutVecDeriv>& result, const Data<InVecDeriv>& data) const
    {
        helper::WriteAccessor<Data<OutVecDeriv> > res (result);
        helper::ReadAccessor<Data<InVecDeriv> > dat (data);

        // use optimized product if possible
        if(canCast(dat.ref()))
        {
            const Eigen::Map<VectorEigen> d(const_cast<Real*>(&dat[0][0]),dat.size()*Nin);
            Eigen::Map<VectorEigen> r(&res[0][0],res.size()*Nout);
            r += this->compressedMatrix * d;
            return;
        }

        // convert the data to Eigen type
        VectorEigen aux1(this->colSize()),aux2(this->rowSize());
        for(unsigned i=0; i<dat.size(); i++)
        {
            for(unsigned j=0; j<Nin; j++)
                aux1[Nin* i+j] = dat[i][j];
        }
        // compute the product
        aux2 = this->compressedMatrix * aux1;
        // convert the result back to the Sofa type
        for(unsigned i=0; i<res.size(); i++)
        {
            for(unsigned j=0; j<Nout; j++)
                res[i][j] += aux2[Nout* i+j];
        }
    }

    /// compute result += A * data * fact
    void addMult( Data<OutVecDeriv>& result, const Data<InVecDeriv>& data, const Real fact) const
    {
        helper::WriteAccessor<Data<OutVecDeriv> > res (result);
        helper::ReadAccessor<Data<InVecDeriv> > dat (data);

        // use optimized product if possible
        if(canCast(dat.ref()))
        {
            const Eigen::Map<VectorEigen> d(const_cast<Real*>(&dat[0][0]),dat.size()*Nin);
            Eigen::Map<VectorEigen> r(&res[0][0],res.size()*Nout);
            r += this->compressedMatrix * d * fact;
            return;
        }

        // convert the data to Eigen type
        VectorEigen aux1(this->colSize()),aux2(this->rowSize());
        for(unsigned i=0; i<dat.size(); i++)
        {
            for(unsigned j=0; j<Nin; j++)
                aux1[Nin* i+j] = dat[i][j];
        }
        // compute the product
        aux2 = this->compressedMatrix * aux1;
        // convert the result back to the Sofa type
        for(unsigned i=0; i<res.size(); i++)
        {
            for(unsigned j=0; j<Nout; j++)
                res[i][j] += aux2[Nout* i+j]*fact;
        }
    }

    /// compute result += A^T * data
    void addMultTranspose( InVecDeriv& result, const OutVecDeriv& data ) const
    {
        // use optimized product if possible
        if(canCast(result))
        {
            const Eigen::Map<VectorEigen> d(const_cast<Real*>(&data[0][0]),data.size()*Nout);
            Eigen::Map<VectorEigen> r(&result[0][0],result.size()*Nin);
            r += this->compressedMatrix.transpose() * d;
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
        aux2 = this->compressedMatrix.transpose() * aux1;
        // convert the result back to the Sofa type
        for(unsigned i=0; i<result.size(); i++)
        {
            for(unsigned j=0; j<Nin; j++)
                result[i][j] += aux2[Nin* i+j];
        }
    }
    /// compute result += A^T * data
    void addMultTranspose( Data<InVecDeriv>& result, const Data<OutVecDeriv>& data ) const
    {
        helper::WriteAccessor<Data<InVecDeriv> > res (result);
        helper::ReadAccessor<Data<OutVecDeriv> > dat (data);


        // use optimized product if possible
        if(canCast(res.wref()))
        {
            const Eigen::Map<VectorEigen> d(const_cast<Real*>(&dat[0][0]),dat.size()*Nout);
            Eigen::Map<VectorEigen> r(&res[0][0],res.size()*Nin);
            r += this->compressedMatrix.transpose() * d;
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
        aux2 = this->compressedMatrix.transpose() * aux1;
        // convert the result back to the Sofa type
        for(unsigned i=0; i<res.size(); i++)
        {
            for(unsigned j=0; j<Nin; j++)
                res[i][j] += aux2[Nin* i+j];
        }
    }


    static const char* Name();

private:
    //@{
    /** Auxiliary variables for methods beginBlockRow(unsigned row), createBlock( unsigned column,  const Block& b ) and endBlockRow() */
    unsigned bRow;
    vector<unsigned> bColumns;
    vector<Block> blocks;
    //@}

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

    /// Converts the incoming matrix to compreddedIncoming and clears the incoming matrix.
    void compress_incomingBlocks()
    {
        this->compressedIncoming.setZero();
        this->compressedIncoming.resize( this->compressedMatrix.rows(), this->compressedMatrix.cols() );
        if( incomingBlocks.empty() ) return;

        int rowStarted = 0;
        for( typename BlockMatMap::const_iterator blockRow=incomingBlocks.begin(),rend=incomingBlocks.end(); blockRow!=rend; blockRow++ )
        {
            int blRow = (*blockRow).first;

            while( rowStarted<blRow*Nout )   // make sure all the rows are started, even the empty ones
            {
                this->compressedIncoming.startVec(rowStarted);
                rowStarted++;
            }

            for( unsigned r=0; r<Nout; r++ )   // process one scalar row after another
            {
                if(r+ blRow*Nout >= this->rowSize() ) break;
//                cerr<<"compress_incomingBlock():: startVec " << rowStarted << endl;
                this->compressedIncoming.startVec(rowStarted++);
                for( typename BlockRowMap::const_iterator c=(*blockRow).second.begin(),cend=(*blockRow).second.end(); c!=cend; c++ )
                {
                    int blCol = (*c).first;
                    const Block& b = (*c).second;
                    for( unsigned c=0; c<Nin; c++ ) if( c+ blCol*Nin < this->colSize() )
                        {
                            this->compressedIncoming.insertBack(r + blRow*Nout, c + blCol*Nin) = b[r][c];
                        }
                }
            }
        }
        this->compressedIncoming.finalize();
        incomingBlocks.clear();
    }


};

template<> inline const char* EigenSparseMatrix<defaulttype::Vec3dTypes, defaulttype::Vec1dTypes >::Name() { return "EigenSparseMatrix3d1d"; }
template<> inline const char* EigenSparseMatrix<defaulttype::Vec3fTypes, defaulttype::Vec1fTypes >::Name() { return "EigenSparseMatrix3f1f"; }







///** Partia specialization for the case where In and Out have different Real types. Just a quick fix, to be improved.
//  */
template<class InDataTypes>
class EigenSparseMatrix<InDataTypes,defaulttype::ExtVec3fTypes> : public EigenBaseSparseMatrix<float>
//class EigenSparseMatrix : public EigenBaseSparseMatrix<typename OutDataTypes::Real>
{
public:
    typedef defaulttype::ExtVec3fTypes OutDataTypes;
    typedef EigenBaseSparseMatrix<typename OutDataTypes::Real> Inherit;
    typedef typename OutDataTypes::Real Real;
    typedef Eigen::SparseMatrix<Real,Eigen::RowMajor> CompressedMatrix;
    typedef Eigen::Matrix<Real,Eigen::Dynamic,1>  VectorEigen;

    typedef typename InDataTypes::Deriv InDeriv;
    typedef typename InDataTypes::VecDeriv InVecDeriv;
    typedef typename InDataTypes::Real InReal;
    typedef typename OutDataTypes::Deriv OutDeriv;
    typedef typename OutDataTypes::VecDeriv OutVecDeriv;
    enum { Nin=InDataTypes::deriv_total_size, Nout=OutDataTypes::deriv_total_size };
    typedef defaulttype::Mat<Nout,Nin,Real> Block;  ///< block relating an OutDeriv to an InDeriv. This is used for input only, not for internal storage.

protected:
    typedef std::map<int,Block> BlockRowMap;        ///< Map which represents one block-view row of the matrix. The index represents the block-view column index of an entry.
    typedef std::map<int,BlockRowMap> BlockMatMap;  ///< Map which represents a block-view matrix. The index represents the block-view index of a block-view row.
    BlockMatMap incomingBlocks;                     ///< To store block-view data before it is compressed in optimized format.
    typedef Eigen::Matrix<InReal,Eigen::Dynamic,1>  VectorEigenIn;

public:

    EigenSparseMatrix(int nbRow=0, int nbCol=0):Inherit(nbRow,nbCol) {}

    /// Resize the matrix without preserving the data (the matrix is set to zero), with the size given in number of blocks
    void resizeBlocks(int nbBlockRows, int nbBlockCols)
    {
        this->compressedMatrix.resize(nbBlockRows * Nout, nbBlockCols * Nin);
    }


    /// Finalize the matrix after a series of insertions. Add the values from the temporary list to the compressed matrix, and clears the list.
    virtual void compress()
    {
        Inherit::compress();

        if( incomingBlocks.empty() ) return;
        compress_incomingBlocks();
        //        cerr<<"compress, before incoming blocks " << this->eigenMatrix << endl;
        //        cerr<<"compress, incoming blocks " << this->compressedIncoming << endl;
        this->compressedMatrix += this->compressedIncoming;
        //        cerr<<"compress, final value " << this->eigenMatrix << endl;
        this->compressedMatrix.finalize();
    }

    /** Return write access to an incoming block.
    Note that this does not give access to the compressed matrix.
    The block belongs to a temporary list which will be added to the compressed matrix using method compress().
    */
    Block& wBlock( int i, int j )
    {
        return incomingBlocks[i][j];
    }


    /** Prepare the insertion of a new row of blocks in the matrix.
       Then create blocks using createBlock( unsigned column,  const Block& b ).
        Then finally use endBlockRow() to validate the row insertion.
        @sa createBlock( unsigned column,  const Block& b )
        @sa endBlockRow()
        */
    void beginBlockRow(unsigned row)
    {
        bRow = row;
        bColumns.clear();
        blocks.clear();
    }

    /** Create a block in the current row, previously initialized using beginBlockRow(unsigned row).
        The blocks need not be created in column order. The blocks are not actually created in the matrix until method endBlockRow() is called.
        */
    void createBlock( unsigned column,  const Block& b )
    {
        blocks.push_back(b);
        bColumns.push_back(column);
    }

    /** Finalize the creation of the current block row.
      @sa beginBlockRow(unsigned row)
      @sa createBlock( unsigned column,  const Block& b )
      */
    void endBlockRow()
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
                    if( b[r][c]!=0.0 )
                        this->insertBack( r + bRow*Nout, c + bColumns[p[i]] * Nin, b[r][c]);
                }
            }
        }
    }

    /** Set from a CompressedRowSparseMatrix. @pre crs must be compressed
      */
    void copyFrom( const CompressedRowSparseMatrix<Block>& crs )
    {
        resize( crs.rowSize(), crs.colSize() );

        int rowStarted = 0;
        for (unsigned int xi = 0; xi < crs.rowIndex.size(); ++xi)  // for each non-null block row
        {
            int blRow = crs.rowIndex[xi];      // block row

            while( rowStarted<blRow*Nout )   // make sure all the rows are started, even the empty ones
            {
                this->compressedMatrix.startVec(rowStarted);
                rowStarted++;
            }

            typename CompressedRowSparseMatrix<Block>::Range rowRange(crs.rowBegin[xi], crs.rowBegin[xi+1]);

            for( unsigned r=0; r<Nout; r++ )   // process one scalar row after another
            {
                if(r+ blRow*Nout >= this->rowSize() ) break;
                //                cerr<<"compress_incomingBlock():: startVec " << rowStarted << endl;
                this->compressedMatrix.startVec(rowStarted++);


                for (int xj = rowRange.begin(); xj < rowRange.end(); ++xj)  // for each non-null block
                {
                    int blCol = crs.colsIndex[xj];     // block column
                    const Block& b = crs.colsValue[xj]; // block value
                    for( unsigned c=0; c<Nin; c++ ) if( c+ blCol*Nin < this->colSize() )
                        {
                            this->compressedMatrix.insertBack(r + blRow*Nout, c + blCol*Nin) = b[r][c];
                        }

                }
            }
        }
        this->compress();

    }


//    /// compute result = A * data
//    void mult( OutVecDeriv& result, const InVecDeriv& data ) const
//    {
//        // use optimized product if possible
//        if(canCast(data)){
//            const Eigen::Map<VectorEigen> d(const_cast<Real*>(&data[0][0]),data.size()*Nin);
//            Eigen::Map<VectorEigen> r(&result[0][0],result.size()*Nout);
//            r = this->compressedMatrix * d;
//        }
//        else {

//            // convert the data to Eigen type
//            VectorEigen aux1(this->colSize(),1), aux2(this->rowSize(),1);
//            for(unsigned i=0; i<data.size();i++){
//                for(unsigned j=0; j<Nin; j++)
//                    aux1[Nin* i+j] = data[i][j];
//            }
//            // compute the product
//            aux2 = this->compressedMatrix * aux1;
//            // convert the result back to the Sofa type
//            for(unsigned i=0; i<result.size();i++){
//                for(unsigned j=0; j<Nout; j++)
//                    result[i][j] = aux2[Nout* i+j];
//            }
//        }
//    }



    /// compute result = A * data
    void mult( Data<OutVecDeriv>& _result, const Data<InVecDeriv>& _data ) const
    {
        helper::WriteAccessor<Data<OutVecDeriv> > result (_result);
        helper::ReadAccessor<Data<InVecDeriv> > data (_data);

//        // use optimized product if possible
//        if(canCast(data.ref())){
//            typedef Eigen::Map<VectorEigenIn> InVectorMap;
//            const InVectorMap d(const_cast<InReal*>(&data[0][0]),data.size()*Nin);
//            VectorEigenIn din = d;
//            VectorEigen dout = din.cast<float>();
//            typedef Eigen::Map<VectorEigen> OutVectorMap;
//            OutVectorMap r(&result[0][0],result.size()*Nout);
//            r = this->compressedMatrix * dout;
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
        aux2 = this->compressedMatrix * aux1;
        // convert the result back to the Sofa type
        for(unsigned i=0; i<result.size(); i++)
        {
            for(unsigned j=0; j<Nout; j++)
                result[i][j] = aux2[Nout* i+j];
        }
    }

//    /// compute result += A * data
//    void addMult( OutVecDeriv& result, const InVecDeriv& data ) const
//    {
//        // use optimized product if possible
//        if(canCast(data)){
//            const Eigen::Map<VectorEigen> d(const_cast<Real*>(&data[0][0]),data.size()*Nin);
//            Eigen::Map<VectorEigen> r(&result[0][0],result.size()*Nout);
//            r += this->compressedMatrix * d;
//            return;
//        }

//        // convert the data to Eigen type
//        VectorEigen aux1(this->colSize()),aux2(this->rowSize());
//        for(unsigned i=0; i<data.size();i++){
//            for(unsigned j=0; j<Nin; j++)
//                aux1[Nin* i+j] = data[i][j];
//        }
//        // compute the product
//        aux2 = this->compressedMatrix * aux1;
//        // convert the result back to the Sofa type
//        for(unsigned i=0; i<result.size();i++){
//            for(unsigned j=0; j<Nout; j++)
//                result[i][j] += aux2[Nout* i+j];
//        }
//    }

//    /// compute result += A * data
//    void addMult( Data<OutVecDeriv>& result, const Data<InVecDeriv>& data) const
//    {
//        helper::WriteAccessor<Data<OutVecDeriv> > res (result);
//        helper::ReadAccessor<Data<InVecDeriv> > dat (data);

//        // use optimized product if possible
//        if(canCast(dat.ref())){
//            const Eigen::Map<VectorEigen> d(const_cast<Real*>(&dat[0][0]),dat.size()*Nin);
//            Eigen::Map<VectorEigen> r(&res[0][0],res.size()*Nout);
//            r += this->compressedMatrix * d;
//            return;
//        }

//        // convert the data to Eigen type
//        VectorEigen aux1(this->colSize()),aux2(this->rowSize());
//        for(unsigned i=0; i<dat.size();i++){
//            for(unsigned j=0; j<Nin; j++)
//                aux1[Nin* i+j] = dat[i][j];
//        }
//        // compute the product
//        aux2 = this->compressedMatrix * aux1;
//        // convert the result back to the Sofa type
//        for(unsigned i=0; i<res.size();i++){
//            for(unsigned j=0; j<Nout; j++)
//                res[i][j] += aux2[Nout* i+j];
//        }
//    }

//    /// compute result += A * data * fact
//    void addMult( Data<OutVecDeriv>& result, const Data<InVecDeriv>& data, const Real fact) const
//    {
//        helper::WriteAccessor<Data<OutVecDeriv> > res (result);
//        helper::ReadAccessor<Data<InVecDeriv> > dat (data);

//        // use optimized product if possible
//        if(canCast(dat.ref())){
//            const Eigen::Map<VectorEigen> d(const_cast<Real*>(&dat[0][0]),dat.size()*Nin);
//            Eigen::Map<VectorEigen> r(&res[0][0],res.size()*Nout);
//            r += this->compressedMatrix * d * fact;
//            return;
//        }

//        // convert the data to Eigen type
//        VectorEigen aux1(this->colSize()),aux2(this->rowSize());
//        for(unsigned i=0; i<dat.size();i++){
//            for(unsigned j=0; j<Nin; j++)
//                aux1[Nin* i+j] = dat[i][j];
//        }
//        // compute the product
//        aux2 = this->compressedMatrix * aux1;
//        // convert the result back to the Sofa type
//        for(unsigned i=0; i<res.size();i++){
//            for(unsigned j=0; j<Nout; j++)
//                res[i][j] += aux2[Nout* i+j]*fact;
//        }
//    }

//    /// compute result += A^T * data
//    void addMultTranspose( InVecDeriv& result, const OutVecDeriv& data ) const
//    {
//        // use optimized product if possible
//        if(canCast(result)){
//            const Eigen::Map<VectorEigen> d(const_cast<Real*>(&data[0][0]),data.size()*Nout);
//            Eigen::Map<VectorEigen> r(&result[0][0],result.size()*Nin);
//            r += this->compressedMatrix.transpose() * d;
//            return;
//        }

//        // convert the data to Eigen type
//        VectorEigen aux1(this->rowSize()),aux2(this->colSize());
//        for(unsigned i=0; i<data.size();i++){
//            for(unsigned j=0; j<Nout; j++)
//                aux1[Nout* i+j] = data[i][j];
//        }
//        // compute the product
//        aux2 = this->compressedMatrix.transpose() * aux1;
//        // convert the result back to the Sofa type
//        for(unsigned i=0; i<result.size();i++){
//            for(unsigned j=0; j<Nin; j++)
//                result[i][j] += aux2[Nin* i+j];
//        }
//    }

    /// compute result += A^T * data
    void addMultTranspose( Data<InVecDeriv>& result, const Data<OutVecDeriv>& data ) const
    {
        helper::WriteAccessor<Data<InVecDeriv> > res (result);
        helper::ReadAccessor<Data<OutVecDeriv> > dat (data);


//        // use optimized product if possible
//        if(canCast(res.wref())){
//            const Eigen::Map<VectorEigen> d(const_cast<Real*>(&dat[0][0]),dat.size()*Nout);
//            VectorEigen rr = this->compressedMatrix.transpose() * d;
//            Eigen::Map<VectorEigenIn> r(&res[0][0],res.size()*Nin);
//            r += rr;
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
        aux2 = this->compressedMatrix.transpose() * aux1;
        // convert the result back to the Sofa type
        for(unsigned i=0; i<res.size(); i++)
        {
            for(unsigned j=0; j<Nin; j++)
                res[i][j] += aux2[Nin* i+j];
        }
    }


    static const char* Name();

private:
    //@{
    /** Auxiliary variables for methods beginBlockRow(unsigned row), createBlock( unsigned column,  const Block& b ) and endBlockRow() */
    unsigned bRow;
    vector<unsigned> bColumns;
    vector<Block> blocks;
    //@}

//    bool canCast( const InVecDeriv& v ) const
//    {
//        //        cerr<<"canCast, size = " << v.size() << endl;
//        //        cerr<<"canCast, length = " << &v[v.size()-1][0] - &v[0][0] << endl;
//        //        cerr<<"canCast, (v.size()-1)*sizeof(InDeriv) = " << (v.size()-1)*sizeof(InDeriv) << endl;
//        //        int diff = (&v[v.size()-1][0]-&v[0][0]) * sizeof(Real);
//        //        cerr<<"canCast,  diff = " << diff << endl;
//        if(  (v.size()-1)*sizeof(InDeriv) ==  (&v[v.size()-1][0]-&v[0][0]) * sizeof(Real)) // contiguous
//            return true;
//        else return false;

//    }

    /// Converts the incoming matrix to compreddedIncoming and clears the incoming matrix.
    void compress_incomingBlocks()
    {
        this->compressedIncoming.setZero();
        this->compressedIncoming.resize( this->compressedMatrix.rows(), this->compressedMatrix.cols() );
        if( incomingBlocks.empty() ) return;

        int rowStarted = 0;
        for( typename BlockMatMap::const_iterator blockRow=incomingBlocks.begin(),rend=incomingBlocks.end(); blockRow!=rend; blockRow++ )
        {
            int blRow = (*blockRow).first;

            while( rowStarted<blRow*Nout )   // make sure all the rows are started, even the empty ones
            {
                this->compressedIncoming.startVec(rowStarted);
                rowStarted++;
            }

            for( unsigned r=0; r<Nout; r++ )   // process one scalar row after another
            {
                if(r+ blRow*Nout >= this->rowSize() ) break;
//                cerr<<"compress_incomingBlock():: startVec " << rowStarted << endl;
                this->compressedIncoming.startVec(rowStarted++);
                for( typename BlockRowMap::const_iterator c=(*blockRow).second.begin(),cend=(*blockRow).second.end(); c!=cend; c++ )
                {
                    int blCol = (*c).first;
                    const Block& b = (*c).second;
                    for( unsigned c=0; c<Nin; c++ ) if( c+ blCol*Nin < this->colSize() )
                        {
                            this->compressedIncoming.insertBack(r + blRow*Nout, c + blCol*Nin) = b[r][c];
                        }
                }
            }
        }
        this->compressedIncoming.finalize();
        incomingBlocks.clear();
    }


};
//{
//public:
//    typedef EigenBaseSparseMatrix<float> Inherit;
////    typedef typename Inherit::Real InReal;
//    typedef defaulttype::ExtVec3fTypes OutDataTypes;
//    typedef typename OutDataTypes::Real OutReal;
//    typedef Eigen::DynamicSparseMatrix<InReal> CompressedMatrix;
//    typedef Eigen::Matrix<InReal,Eigen::Dynamic,1>  VectorEigen;
//    typedef Eigen::Matrix<InReal,Eigen::Dynamic,1>  InVectorEigen;
//    typedef Eigen::Matrix<OutReal,Eigen::Dynamic,1>  OutVectorEigen;


//    typedef typename InDataTypes::Deriv InDeriv;
//    typedef typename InDataTypes::VecDeriv InVecDeriv;
//    typedef typename OutDataTypes::Deriv OutDeriv;
//    typedef typename OutDataTypes::VecDeriv OutVecDeriv;
//    enum { Nin=InDataTypes::deriv_total_size, Nout=OutDataTypes::deriv_total_size };
//    typedef defaulttype::Mat<Nout,Nin,InReal> Block;  ///< block relating an OutDeriv to an InDeriv. This is used for input only, not for internal storage.

//protected:
//    typedef std::map<int,Block> BlockRowMap;        ///< Map which represents one block-view row of the matrix. The index represents the block-view column index of an entry.
//    typedef std::map<int,BlockRowMap> BlockMatMap;  ///< Map which represents a block-view matrix. The index represents the block-view index of a block-view row.
//    BlockMatMap incomingBlocks;                     ///< To store block-view data before it is compressed in optimized format.

//public:

//    EigenSparseMatrix(int nRow=0, int nCol=0):Inherit(nRow,nCol){}


//    /// Resize the matrix without preserving the data (the matrix is set to zero), with the size given in number of blocks
//    void resizeBlocks(int nbBlockRows, int nbBlockCols)
//    {
//        this->compressedMatrix.resize(nbBlockRows * Nout, nbBlockCols * Nin);
//    }


//    /// Finalize the matrix after a series of insertions. Add the values from the temporary list to the compressed matrix, and clears the list.
//    virtual void compress()
//    {
//        Inherit::compress();

//        if( incomingBlocks.empty() ) return;
//        compress_incomingBlocks();
//        //        cerr<<"compress, before incoming blocks " << this->eigenMatrix << endl;
//        //        cerr<<"compress, incoming blocks " << this->compressedIncoming << endl;
//        this->compressedMatrix += this->compressedIncoming;
//        //        cerr<<"compress, final value " << this->eigenMatrix << endl;
//        this->compressedMatrix.finalize();
//    }

//    /** Return write access to an incoming block.
//    Note that this does not give access to the compressed matrix.
//    The block belongs to a temporary list which will be added to the compressed matrix using method compress().
//    */
//    Block& wBlock( int i, int j )
//    {
//        return incomingBlocks[i][j];
//    }


//    /** Prepare the insertion of a new row of blocks in the matrix.
//       Then create blocks using createBlock( unsigned column,  const Block& b ).
//        Then finally use endBlockRow() to validate the row insertion.
//        @sa createBlock( unsigned column,  const Block& b )
//        @sa endBlockRow()
//        */
//    void beginBlockRow(unsigned row)
//    {
//        bRow = row;
//        bColumns.clear();
//        blocks.clear();
//    }

//    /** Create a block in the current row, previously initialized using beginBlockRow(unsigned row).
//        The blocks need not be created in column order. The blocks are not actually created in the matrix until method endBlockRow() is called.
//        */
//    void createBlock( unsigned column,  const Block& b )
//    {
//        blocks.push_back(b);
//        bColumns.push_back(column);
//    }

//    /** Finalize the creation of the current block row.
//      @sa beginBlockRow(unsigned row)
//      @sa createBlock( unsigned column,  const Block& b )
//      */
//    void endBlockRow()
//    {
//        vector<unsigned> p = helper::sortedPermutation(bColumns); // indices in ascending column order

//        for( unsigned r=0; r<Nout; r++ ){  // process one scalar row after another
//            beginRow(r+ bRow*Nout);
//            for(unsigned i=0; i<p.size(); i++ ){ // process the blocks in ascending order
//                const Block& b = blocks[p[i]];
//                for( unsigned c=0; c<Nin; c++ ){
//                    if( b[r][c]!=0.0 )
//                        this->insertBack( r + bRow*Nout, c + bColumns[p[i]] * Nin, b[r][c]);
//                }
//            }
//        }
//    }

//    /** Set from a CompressedRowSparseMatrix. @pre crs must be compressed
//      */
//    void copyFrom( const CompressedRowSparseMatrix<Block>& crs )
//    {
//        resize( crs.rowSize(), crs.colSize() );

//        int rowStarted = 0;
//        for (unsigned int xi = 0; xi < crs.rowIndex.size(); ++xi)  // for each non-null block row
//        {
//            int blRow = crs.rowIndex[xi];      // block row

//            while( rowStarted<blRow*Nout ){  // make sure all the rows are started, even the empty ones
//                this->compressedMatrix.startVec(rowStarted);
//                rowStarted++;
//            }

//            typename CompressedRowSparseMatrix<Block>::Range rowRange(crs.rowBegin[xi], crs.rowBegin[xi+1]);

//            for( unsigned r=0; r<Nout; r++ ){  // process one scalar row after another
//                if(r+ blRow*Nout >= this->rowSize() ) break;
//                //                cerr<<"compress_incomingBlock():: startVec " << rowStarted << endl;
//                this->compressedMatrix.startVec(rowStarted++);


//                for (int xj = rowRange.begin(); xj < rowRange.end(); ++xj)  // for each non-null block
//                {
//                    int blCol = crs.colsIndex[xj];     // block column
//                    const Block& b = crs.colsValue[xj]; // block value
//                    for( unsigned c=0; c<Nin; c++ ) if( c+ blCol*Nin < this->colSize() )
//                    {
//                        this->compressedMatrix.insertBack(r + blRow*Nout, c + blCol*Nin) = b[r][c];
//                    }

//                }
//            }
//        }
//        this->compress();

//    }




//    /// compute result = A * data
//    void mult( OutVecDeriv& result, const InVecDeriv& data ) const
//    {
//        // convert the data to Eigen type
//        VectorEigen aux1(this->colSize(),1), aux2(this->rowSize(),1);
//        for(unsigned i=0; i<data.size();i++){
//            for(unsigned j=0; j<Nin; j++)
//                aux1[Nin* i+j] = data[i][j];
//        }
//        // compute the product
//        aux2 = this->compressedMatrix * aux1;
//        // convert the result back to the Sofa type
//        for(unsigned i=0; i<result.size();i++){
//            for(unsigned j=0; j<Nout; j++)
//                result[i][j] = aux2[Nout* i+j];
//        }
//    }

//    /// compute result = A * data
//    void mult( Data<OutVecDeriv>& _result, const Data<InVecDeriv>& _data ) const
//    {
//        helper::WriteAccessor<Data<OutVecDeriv> > result (_result);
//        helper::ReadAccessor<Data<InVecDeriv> > data (_data);

//        //        // use optimized product if possible
//        //        if(canCast(data.ref())){
//        //            const Eigen::Map<VectorEigen> d(const_cast<Real*>(&data[0][0]),data.size()*Nin);
//        //            Eigen::Map<VectorEigen> r(&result[0][0],result.size()*Nout);
//        //            r = eigenMatrix * d;
//        //            //            cerr<<"EigenSparseMatrix::mult using maps, in = "<< data << endl;
//        //            //            cerr<<"EigenSparseMatrix::mult using maps, map<in> = "<< d.transpose() << endl;
//        //            //            cerr<<"EigenSparseMatrix::mult using maps, out = "<< result << endl;
//        //            //            cerr<<"EigenSparseMatrix::mult using maps, map<out> = "<< r.transpose() << endl;
//        //            return;
//        //        }

//        // convert the data to Eigen type
//        VectorEigen aux1(this->colSize(),1), aux2(this->rowSize(),1);
//        for(unsigned i=0; i<data.size();i++){
//            for(unsigned j=0; j<Nin; j++)
//                aux1[Nin* i+j] = data[i][j];
//        }
//        // compute the product
//        aux2 = this->compressedMatrix * aux1;
//        // convert the result back to the Sofa type
//        for(unsigned i=0; i<result.size();i++){
//            for(unsigned j=0; j<Nout; j++)
//                result[i][j] = aux2[Nout* i+j];
//        }
//    }

//    /// compute result += A * data
//    void addMult( OutVecDeriv& result, const InVecDeriv& data ) const
//    {
//        // convert the data to Eigen type
//        VectorEigen aux1(this->colSize()),aux2(this->rowSize());
//        for(unsigned i=0; i<data.size();i++){
//            for(unsigned j=0; j<Nin; j++)
//                aux1[Nin* i+j] = data[i][j];
//        }
//        // compute the product
//        aux2 = this->compressedMatrix * aux1;
//        // convert the result back to the Sofa type
//        for(unsigned i=0; i<result.size();i++){
//            for(unsigned j=0; j<Nout; j++)
//                result[i][j] += aux2[Nout* i+j];
//        }
//    }


//    /// compute result += A * data
//    void addMultTranspose( InVecDeriv& result, const OutVecDeriv& data ) const
//    {
//        //        // use optimized product if possible
//        //        if(canCast(result)){
//        //            const Eigen::Map<VectorEigen> d(const_cast<Real*>(&data[0][0]),data.size()*Nout);
//        //            Eigen::Map<VectorEigen> r(&result[0][0],result.size()*Nin);
//        //            r += eigenMatrix.transpose() * d;
//        //            return;
//        //        }

//        // convert the data to Eigen type
//        VectorEigen aux1(this->rowSize()),aux2(this->colSize());
//        for(unsigned i=0; i<data.size();i++){
//            for(unsigned j=0; j<Nout; j++)
//                aux1[Nout* i+j] = data[i][j];
//        }
//        // compute the product
//        aux2 = this->compressedMatrix.transpose() * aux1;
//        // convert the result back to the Sofa type
//        for(unsigned i=0; i<result.size();i++){
//            for(unsigned j=0; j<Nin; j++)
//                result[i][j] += aux2[Nin* i+j];
//        }
//    }
//    /// compute result += A * data
//    void addMultTranspose( Data<InVecDeriv>& result, const Data<OutVecDeriv>& data ) const
//    {
//        helper::WriteAccessor<Data<InVecDeriv> > res (result);
//        helper::ReadAccessor<Data<OutVecDeriv> > dat (data);


//        //        // use optimized product if possible
//        //        if(canCast(res.wref())){
//        //            const Eigen::Map<VectorEigen> d(const_cast<Real*>(&dat[0][0]),dat.size()*Nout);
//        //            Eigen::Map<VectorEigen> r(&res[0][0],res.size()*Nin);
//        //            r += eigenMatrix.transpose() * d;
//        //            return;
//        //        }


//        // convert the data to Eigen type
//        VectorEigen aux1(this->rowSize()),aux2(this->colSize());
//        for(unsigned i=0; i<dat.size();i++){
//            for(unsigned j=0; j<Nout; j++)
//                aux1[Nout* i+j] = dat[i][j];
//        }
//        // compute the product
//        aux2 = this->compressedMatrix.transpose() * aux1;
//        // convert the result back to the Sofa type
//        for(unsigned i=0; i<res.size();i++){
//            for(unsigned j=0; j<Nin; j++)
//                res[i][j] += aux2[Nin* i+j];
//        }
//    }


//    //    static const char* Name();
//private:
//    //@{
//    /** Auxiliary variables for methods beginBlockRow(unsigned row), createBlock( unsigned column,  const Block& b ) and endBlockRow() */
//    unsigned bRow;
//    vector<unsigned> bColumns;
//    vector<Block> blocks;
//    //@}

//    bool canCast( const InVecDeriv& v ) const
//    {
//        //        cerr<<"canCast, size = " << v.size() << endl;
//        //        cerr<<"canCast, length = " << &v[v.size()-1][0] - &v[0][0] << endl;
//        //        cerr<<"canCast, (v.size()-1)*sizeof(InDeriv) = " << (v.size()-1)*sizeof(InDeriv) << endl;
//        //        int diff = (&v[v.size()-1][0]-&v[0][0]) * sizeof(Real);
//        //        cerr<<"canCast,  diff = " << diff << endl;
//        if(  (v.size()-1)*sizeof(InDeriv) ==  (&v[v.size()-1][0]-&v[0][0]) * sizeof(InReal)) // contiguous
//            return true;
//        else return false;

//    }

//    /// Converts the incoming matrix to compreddedIncoming and clears the incoming matrix.
//    void compress_incomingBlocks()
//    {
//        this->compressedIncoming.setZero();
//        this->compressedIncoming.resize( this->compressedMatrix.rows(), this->compressedMatrix.cols() );
//        if( incomingBlocks.empty() ) return;

//        int rowStarted = 0;
//        for( typename BlockMatMap::const_iterator blockRow=incomingBlocks.begin(),rend=incomingBlocks.end(); blockRow!=rend; blockRow++ )
//        {
//            int blRow = (*blockRow).first;

//            while( rowStarted<blRow*Nout ){  // make sure all the rows are started, even the empty ones
//                this->compressedIncoming.startVec(rowStarted);
//                rowStarted++;
//            }

//            for( unsigned r=0; r<Nout; r++ ){  // process one scalar row after another
//                if(r+ blRow*Nout >= this->rowSize() ) break;
////                cerr<<"compress_incomingBlock():: startVec " << rowStarted << endl;
//                this->compressedIncoming.startVec(rowStarted++);
//                for( typename BlockRowMap::const_iterator c=(*blockRow).second.begin(),cend=(*blockRow).second.end(); c!=cend; c++ )
//                {
//                    int blCol = (*c).first;
//                    const Block& b = (*c).second;
//                    for( unsigned c=0; c<Nin; c++ ) if( c+ blCol*Nin < this->colSize() )
//                    {
//                        this->compressedIncoming.insertBack(r + blRow*Nout, c + blCol*Nin) = b[r][c];
//                    }
//                }
//            }
//        }
//        this->compressedIncoming.finalize();
//        incomingBlocks.clear();
//    }


//};



} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
