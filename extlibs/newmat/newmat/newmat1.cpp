//$$ newmat1.cpp   Matrix type functions

// Copyright (C) 1991,2,3,4: R B Davies

//#define WANT_STREAM

#include "newmat.h"

#ifdef use_namespace
namespace NEWMAT {
#endif

#ifdef DO_REPORT
#define REPORT { static ExeCounter ExeCount(__LINE__,1); ++ExeCount; }
#else
#define REPORT {}
#endif


/************************* MatrixType functions *****************************/


// all operations to return MatrixTypes which correspond to valid types
// of matrices.
// Eg: if it has the Diagonal attribute, then it must also have
// Upper, Lower, Band and Symmetric.


MatrixType MatrixType::operator*(const MatrixType& mt) const
{
   REPORT
   int a = attribute & mt.attribute & ~Symmetric;
   a |= (a & Diagonal) * 31;                   // recognise diagonal
   return MatrixType(a);
}

MatrixType MatrixType::SP(const MatrixType& mt) const
// elementwise product
// Lower, Upper, Diag, Band if only one is
// Symmetric, Ones, Valid (and Real) if both are
// Need to include Lower & Upper => Diagonal
// Will need to include both Skew => Symmetric
{
   REPORT
   int a = ((attribute | mt.attribute) & ~(Symmetric + Valid + Ones))
      | (attribute & mt.attribute);
   if ((a & Lower) != 0  &&  (a & Upper) != 0) a |= Diagonal;
   a |= (a & Diagonal) * 31;                   // recognise diagonal
   return MatrixType(a);
}

MatrixType MatrixType::KP(const MatrixType& mt) const
// Kronecker product
// Lower, Upper, Diag, Symmetric, Band, Valid if both are
// Do not treat Band separately
// Ones is complicated so leave this out
{
   REPORT
   int a = (attribute & mt.attribute) & ~Ones;
   return MatrixType(a);
}

MatrixType MatrixType::i() const               // type of inverse
{
   REPORT
   int a = attribute & ~(Band+LUDeco);
   a |= (a & Diagonal) * 31;                   // recognise diagonal
   return MatrixType(a);
}

MatrixType MatrixType::t() const
// swap lower and upper attributes
// assume Upper is in bit above Lower
{
   REPORT
   int a = attribute;
   a ^= (((a >> 1) ^ a) & Lower) * 3;
   return MatrixType(a);
}

MatrixType MatrixType::MultRHS() const
{
   REPORT
   // remove symmetric attribute unless diagonal
   return (attribute >= Dg) ? attribute : (attribute & ~Symmetric);
}

bool Rectangular(MatrixType a, MatrixType b, MatrixType c)
{
   REPORT
   return
      ((a.attribute | b.attribute | c.attribute) & ~MatrixType::Valid) == 0;
}

const char* MatrixType::Value() const
{
// make a string with the name of matrix with the given attributes
   switch (attribute)
   {
   case Valid:                              REPORT return "Rect ";
   case Valid+Symmetric:                    REPORT return "Sym  ";
   case Valid+Band:                         REPORT return "Band ";
   case Valid+Symmetric+Band:               REPORT return "SmBnd";
   case Valid+Upper:                        REPORT return "UT   ";
   case Valid+Diagonal+Symmetric+Band+Upper+Lower:
                                            REPORT return "Diag ";
   case Valid+Diagonal+Symmetric+Band+Upper+Lower+Ones:
                                            REPORT return "Ident";
   case Valid+Band+Upper:                   REPORT return "UpBnd";
   case Valid+Lower:                        REPORT return "LT   ";
   case Valid+Band+Lower:                   REPORT return "LwBnd";
   default:
      REPORT
      if (!(attribute & Valid))             return "UnSp ";
      if (attribute & LUDeco)
         return (attribute & Band) ?     "BndLU" : "Crout";
                                            return "?????";
   }
}


GeneralMatrix* MatrixType::New(int nr, int nc, BaseMatrix* bm) const
{
// make a new matrix with the given attributes

   Tracer tr("New"); GeneralMatrix* gm=0;   // initialised to keep gnu happy
   switch (attribute)
   {
   case Valid:
      REPORT
      if (nc==1) { gm = new ColumnVector(nr); break; }
      if (nr==1) { gm = new RowVector(nc); break; }
      gm = new Matrix(nr, nc); break;

   case Valid+Symmetric:
      REPORT gm = new SymmetricMatrix(nr); break;

   case Valid+Band:
      {
         REPORT
         MatrixBandWidth bw = bm->BandWidth();
         gm = new BandMatrix(nr,bw.lower,bw.upper); break;
      }

   case Valid+Symmetric+Band:
      REPORT gm = new SymmetricBandMatrix(nr,bm->BandWidth().lower); break;

   case Valid+Upper:
      REPORT gm = new UpperTriangularMatrix(nr); break;

   case Valid+Diagonal+Symmetric+Band+Upper+Lower:
      REPORT gm = new DiagonalMatrix(nr); break;

   case Valid+Band+Upper:
      REPORT gm = new UpperBandMatrix(nr,bm->BandWidth().upper); break;

   case Valid+Lower:
      REPORT gm = new LowerTriangularMatrix(nr); break;

   case Valid+Band+Lower:
      REPORT gm = new LowerBandMatrix(nr,bm->BandWidth().lower); break;

   case Valid+Diagonal+Symmetric+Band+Upper+Lower+Ones:
      REPORT gm = new IdentityMatrix(nr); break;

   default:
      Throw(ProgramException("Invalid matrix type"));
   }
   
   MatrixErrorNoSpace(gm); gm->Protect(); return gm;
}


#ifdef use_namespace
}
#endif

