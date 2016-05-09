//$$ submat.cpp                         submatrices

// Copyright (C) 1991,2,3,4: R B Davies

#include "include.h"

#include "newmat.h"
#include "newmatrc.h"

#ifdef use_namespace
namespace NEWMAT {
#endif

#ifdef DO_REPORT
#define REPORT { static ExeCounter ExeCount(__LINE__,11); ++ExeCount; }
#else
#define REPORT {}
#endif


/****************************** submatrices *********************************/

#ifdef TEMPS_DESTROYED_QUICKLY
GetSubMatrix& BaseMatrix::SubMatrix(int first_row, int last_row, int first_col,
   int last_col) const
#else
GetSubMatrix BaseMatrix::SubMatrix(int first_row, int last_row, int first_col,
   int last_col) const
#endif
{
   REPORT
   Tracer tr("SubMatrix");
   int a = first_row - 1; int b = last_row - first_row + 1;
   int c = first_col - 1; int d = last_col - first_col + 1;
   if (a<0 || b<0 || c<0 || d<0) Throw(SubMatrixDimensionException());
                             // allow zero rows or columns
#ifdef TEMPS_DESTROYED_QUICKLY
   GetSubMatrix* x = new GetSubMatrix(this, a, b, c, d, false);
   MatrixErrorNoSpace(x);
   return *x;
#else
   return GetSubMatrix(this, a, b, c, d, false);
#endif
}

#ifdef TEMPS_DESTROYED_QUICKLY
GetSubMatrix& BaseMatrix::SymSubMatrix(int first_row, int last_row) const
#else
GetSubMatrix BaseMatrix::SymSubMatrix(int first_row, int last_row) const
#endif
{
   REPORT
   Tracer tr("SubMatrix(symmetric)");
   int a = first_row - 1; int b = last_row - first_row + 1;
   if (a<0 || b<0) Throw(SubMatrixDimensionException());
                             // allow zero rows or columns
#ifdef TEMPS_DESTROYED_QUICKLY
   GetSubMatrix* x = new GetSubMatrix(this, a, b, a, b, true);
   MatrixErrorNoSpace(x);
   return *x;
#else
   return GetSubMatrix( this, a, b, a, b, true);
#endif
}

#ifdef TEMPS_DESTROYED_QUICKLY
GetSubMatrix& BaseMatrix::Row(int first_row) const
#else
GetSubMatrix BaseMatrix::Row(int first_row) const
#endif
{
   REPORT
   Tracer tr("SubMatrix(row)");
   int a = first_row - 1;
   if (a<0) Throw(SubMatrixDimensionException());
#ifdef TEMPS_DESTROYED_QUICKLY
   GetSubMatrix* x = new GetSubMatrix(this, a, 1, 0, -1, false);
   MatrixErrorNoSpace(x);
   return *x;
#else
   return GetSubMatrix(this, a, 1, 0, -1, false);
#endif
}

#ifdef TEMPS_DESTROYED_QUICKLY
GetSubMatrix& BaseMatrix::Rows(int first_row, int last_row) const
#else
GetSubMatrix BaseMatrix::Rows(int first_row, int last_row) const
#endif
{
   REPORT
   Tracer tr("SubMatrix(rows)");
   int a = first_row - 1; int b = last_row - first_row + 1;
   if (a<0 || b<0) Throw(SubMatrixDimensionException());
                             // allow zero rows or columns
#ifdef TEMPS_DESTROYED_QUICKLY
   GetSubMatrix* x = new GetSubMatrix(this, a, b, 0, -1, false);
   MatrixErrorNoSpace(x);
   return *x;
#else
   return GetSubMatrix(this, a, b, 0, -1, false);
#endif
}

#ifdef TEMPS_DESTROYED_QUICKLY
GetSubMatrix& BaseMatrix::Column(int first_col) const
#else
GetSubMatrix BaseMatrix::Column(int first_col) const
#endif
{
   REPORT
   Tracer tr("SubMatrix(column)");
   int c = first_col - 1;
   if (c<0) Throw(SubMatrixDimensionException());
#ifdef TEMPS_DESTROYED_QUICKLY
   GetSubMatrix* x = new GetSubMatrix(this, 0, -1, c, 1, false);
   MatrixErrorNoSpace(x);
   return *x;
#else
   return GetSubMatrix(this, 0, -1, c, 1, false);
#endif
}

#ifdef TEMPS_DESTROYED_QUICKLY
GetSubMatrix& BaseMatrix::Columns(int first_col, int last_col) const
#else
GetSubMatrix BaseMatrix::Columns(int first_col, int last_col) const
#endif
{
   REPORT
   Tracer tr("SubMatrix(columns)");
   int c = first_col - 1; int d = last_col - first_col + 1;
   if (c<0 || d<0) Throw(SubMatrixDimensionException());
                             // allow zero rows or columns
#ifdef TEMPS_DESTROYED_QUICKLY
   GetSubMatrix* x = new GetSubMatrix(this, 0, -1, c, d, false);
   MatrixErrorNoSpace(x);
   return *x;
#else
   return GetSubMatrix(this, 0, -1, c, d, false);
#endif
}

void GetSubMatrix::SetUpLHS()
{
   REPORT
   Tracer tr("SubMatrix(LHS)");
   const BaseMatrix* bm1 = bm;
   GeneralMatrix* gm1 = ((BaseMatrix*&)bm)->Evaluate();
   if ((BaseMatrix*)gm1!=bm1)
      Throw(ProgramException("Invalid LHS"));
   if (row_number < 0) row_number = gm1->Nrows();
   if (col_number < 0) col_number = gm1->Ncols();
   if (row_skip+row_number > gm1->Nrows()
      || col_skip+col_number > gm1->Ncols())
         Throw(SubMatrixDimensionException());
}

void GetSubMatrix::operator<<(const BaseMatrix& bmx)
{
   REPORT
   Tracer tr("SubMatrix(<<)"); GeneralMatrix* gmx = 0;
   Try
   {
      SetUpLHS(); gmx = ((BaseMatrix&)bmx).Evaluate();
      if (row_number != gmx->Nrows() || col_number != gmx->Ncols())
         Throw(IncompatibleDimensionsException());
      MatrixRow mrx(gmx, LoadOnEntry);
      MatrixRow mr(gm, LoadOnEntry+StoreOnExit+DirectPart, row_skip);
                                     // do need LoadOnEntry
      MatrixRowCol sub; int i = row_number;
      while (i--)
      {
         mr.SubRowCol(sub, col_skip, col_number);   // put values in sub
         sub.Copy(mrx); mr.Next(); mrx.Next();
      }
      gmx->tDelete();
#ifdef TEMPS_DESTROYED_QUICKLY
      delete this;
#endif
   }

   CatchAll
   {
      if (gmx) gmx->tDelete();
#ifdef TEMPS_DESTROYED_QUICKLY
      delete this;
#endif
      ReThrow;
   }
}

void GetSubMatrix::operator=(const BaseMatrix& bmx)
{
   REPORT
   Tracer tr("SubMatrix(=)"); GeneralMatrix* gmx = 0;
   // MatrixConversionCheck mcc;         // Check for loss of info
   Try
   {
      SetUpLHS(); gmx = ((BaseMatrix&)bmx).Evaluate();
      if (row_number != gmx->Nrows() || col_number != gmx->Ncols())
         Throw(IncompatibleDimensionsException());
      LoadAndStoreFlag lasf =
         (  row_skip == col_skip
            && gm->Type().IsSymmetric()
            && gmx->Type().IsSymmetric() )
        ? LoadOnEntry+DirectPart
        : LoadOnEntry;
      MatrixRow mrx(gmx, lasf);
      MatrixRow mr(gm, LoadOnEntry+StoreOnExit+DirectPart, row_skip);
                                     // do need LoadOnEntry
      MatrixRowCol sub; int i = row_number;
      while (i--)
      {
         mr.SubRowCol(sub, col_skip, col_number);   // put values in sub
         sub.CopyCheck(mrx); mr.Next(); mrx.Next();
      }
      gmx->tDelete();
#ifdef TEMPS_DESTROYED_QUICKLY
      delete this;
#endif
   }

   CatchAll
   {
      if (gmx) gmx->tDelete();
#ifdef TEMPS_DESTROYED_QUICKLY
      delete this;
#endif
      ReThrow;
   }
}

void GetSubMatrix::operator<<(const Real* r)
{
   REPORT
   Tracer tr("SubMatrix(<<Real*)");
   SetUpLHS();
   if (row_skip+row_number > gm->Nrows() || col_skip+col_number > gm->Ncols())
      Throw(SubMatrixDimensionException());
   MatrixRow mr(gm, LoadOnEntry+StoreOnExit+DirectPart, row_skip);
                                  // do need LoadOnEntry
   MatrixRowCol sub; int i = row_number;
   while (i--)
   {
      mr.SubRowCol(sub, col_skip, col_number);   // put values in sub
      sub.Copy(r); mr.Next();
   }
#ifdef TEMPS_DESTROYED_QUICKLY
   delete this;
#endif
}

void GetSubMatrix::operator=(Real r)
{
   REPORT
   Tracer tr("SubMatrix(=Real)");
   SetUpLHS();
   MatrixRow mr(gm, LoadOnEntry+StoreOnExit+DirectPart, row_skip);
                                  // do need LoadOnEntry
   MatrixRowCol sub; int i = row_number;
   while (i--)
   {
      mr.SubRowCol(sub, col_skip, col_number);   // put values in sub
      sub.Copy(r); mr.Next();
   }
#ifdef TEMPS_DESTROYED_QUICKLY
   delete this;
#endif
}

void GetSubMatrix::Inject(const GeneralMatrix& gmx)
{
   REPORT
   Tracer tr("SubMatrix(inject)");
   SetUpLHS();
   if (row_number != gmx.Nrows() || col_number != gmx.Ncols())
      Throw(IncompatibleDimensionsException());
   MatrixRow mrx((GeneralMatrix*)(&gmx), LoadOnEntry);
   MatrixRow mr(gm, LoadOnEntry+StoreOnExit+DirectPart, row_skip);
                                  // do need LoadOnEntry
   MatrixRowCol sub; int i = row_number;
   while (i--)
   {
      mr.SubRowCol(sub, col_skip, col_number);   // put values in sub
      sub.Inject(mrx); mr.Next(); mrx.Next();
   }
#ifdef TEMPS_DESTROYED_QUICKLY
   delete this;
#endif
}

void GetSubMatrix::operator+=(const BaseMatrix& bmx)
{
   REPORT
   Tracer tr("SubMatrix(+=)"); GeneralMatrix* gmx = 0;
   // MatrixConversionCheck mcc;         // Check for loss of info
   Try
   {
      SetUpLHS(); gmx = ((BaseMatrix&)bmx).Evaluate();
      if (row_number != gmx->Nrows() || col_number != gmx->Ncols())
         Throw(IncompatibleDimensionsException());
      MatrixRow mrx(gmx, LoadOnEntry);
      MatrixRow mr(gm, LoadOnEntry+StoreOnExit+DirectPart, row_skip);
                                     // do need LoadOnEntry
      MatrixRowCol sub; int i = row_number;
      while (i--)
      {
         mr.SubRowCol(sub, col_skip, col_number);   // put values in sub
         sub.Check(mrx);                            // check for loss of info
         sub.Add(mrx); mr.Next(); mrx.Next();
      }
      gmx->tDelete();
#ifdef TEMPS_DESTROYED_QUICKLY
      delete this;
#endif
   }

   CatchAll
   {
      if (gmx) gmx->tDelete();
#ifdef TEMPS_DESTROYED_QUICKLY
      delete this;
#endif
      ReThrow;
   }
}

void GetSubMatrix::operator-=(const BaseMatrix& bmx)
{
   REPORT
   Tracer tr("SubMatrix(-=)"); GeneralMatrix* gmx = 0;
   // MatrixConversionCheck mcc;         // Check for loss of info
   Try
   {
      SetUpLHS(); gmx = ((BaseMatrix&)bmx).Evaluate();
      if (row_number != gmx->Nrows() || col_number != gmx->Ncols())
         Throw(IncompatibleDimensionsException());
      MatrixRow mrx(gmx, LoadOnEntry);
      MatrixRow mr(gm, LoadOnEntry+StoreOnExit+DirectPart, row_skip);
                                     // do need LoadOnEntry
      MatrixRowCol sub; int i = row_number;
      while (i--)
      {
         mr.SubRowCol(sub, col_skip, col_number);   // put values in sub
         sub.Check(mrx);                            // check for loss of info
         sub.Sub(mrx); mr.Next(); mrx.Next();
      }
      gmx->tDelete();
#ifdef TEMPS_DESTROYED_QUICKLY
      delete this;
#endif
   }

   CatchAll
   {
      if (gmx) gmx->tDelete();
#ifdef TEMPS_DESTROYED_QUICKLY
      delete this;
#endif
      ReThrow;
   }
}

void GetSubMatrix::operator+=(Real r)
{
   REPORT
   Tracer tr("SubMatrix(+= or -= Real)");
   // MatrixConversionCheck mcc;         // Check for loss of info
   Try
   {
      SetUpLHS();
      MatrixRow mr(gm, LoadOnEntry+StoreOnExit+DirectPart, row_skip);
                                     // do need LoadOnEntry
      MatrixRowCol sub; int i = row_number;
      while (i--)
      {
         mr.SubRowCol(sub, col_skip, col_number);   // put values in sub
         sub.Check();                               // check for loss of info
         sub.Add(r); mr.Next();
      }
#ifdef TEMPS_DESTROYED_QUICKLY
      delete this;
#endif
   }

   CatchAll
   {
#ifdef TEMPS_DESTROYED_QUICKLY
      delete this;
#endif
      ReThrow;
   }
}

void GetSubMatrix::operator*=(Real r)
{
   REPORT
   Tracer tr("SubMatrix(*= or /= Real)");
   // MatrixConversionCheck mcc;         // Check for loss of info
   Try
   {
      SetUpLHS();
      MatrixRow mr(gm, LoadOnEntry+StoreOnExit+DirectPart, row_skip);
                                     // do need LoadOnEntry
      MatrixRowCol sub; int i = row_number;
      while (i--)
      {
         mr.SubRowCol(sub, col_skip, col_number);   // put values in sub
         sub.Multiply(r); mr.Next();
      }
#ifdef TEMPS_DESTROYED_QUICKLY
      delete this;
#endif
   }

   CatchAll
   {
#ifdef TEMPS_DESTROYED_QUICKLY
      delete this;
#endif
      ReThrow;
   }
}

#ifdef use_namespace
}
#endif

