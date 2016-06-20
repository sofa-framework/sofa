//$$ newmat8.cpp         Advanced LU transform, scalar functions

// Copyright (C) 1991,2,3,4,8: R B Davies

#define WANT_MATH

#include "include.h"

#include "newmat.h"
#include "newmatrc.h"
#include "precisio.h"

#ifdef use_namespace
namespace NEWMAT {
#endif


#ifdef DO_REPORT
#define REPORT { static ExeCounter ExeCount(__LINE__,8); ++ExeCount; }
#else
#define REPORT {}
#endif


/************************** LU transformation ****************************/

void CroutMatrix::ludcmp()
// LU decomposition from Golub & Van Loan, algorithm 3.4.1, (the "outer
// product" version).
// This replaces the code derived from Numerical Recipes in C in previous
// versions of newmat and being row oriented runs much faster with large
// matrices.
{
   REPORT
   Tracer trace( "Crout(ludcmp)" ); sing = false;
   Real* akk = store;                    // runs down diagonal

   Real big = fabs(*akk); int mu = 0; Real* ai = akk; int k;

   for (k = 1; k < nrows; k++)
   {
      ai += nrows; const Real trybig = fabs(*ai);
      if (big < trybig) { big = trybig; mu = k; }
   }


   if (nrows) for (k = 0;;)
   {
      /*
      int mu1;
      {
         Real big = fabs(*akk); mu1 = k; Real* ai = akk; int i;

         for (i = k+1; i < nrows; i++)
         {
            ai += nrows; const Real trybig = fabs(*ai);
            if (big < trybig) { big = trybig; mu1 = i; }
         }
      }
      if (mu1 != mu) cout << k << " " << mu << " " << mu1 << endl;
      */

      indx[k] = mu;

      if (mu != k)                       //row swap
      {
         Real* a1 = store + nrows * k; Real* a2 = store + nrows * mu; d = !d;
         int j = nrows;
         while (j--) { const Real temp = *a1; *a1++ = *a2; *a2++ = temp; }
      }

      Real diag = *akk; big = 0; mu = k + 1;
      if (diag != 0)
      {
         ai = akk; int i = nrows - k - 1;
         while (i--)
         {
            ai += nrows; Real* al = ai; Real mult = *al / diag; *al = mult;
            int l = nrows - k - 1; Real* aj = akk;
            // work out the next pivot as part of this loop
            // this saves a column operation
            if (l-- != 0)
            {
               *(++al) -= (mult * *(++aj));
               const Real trybig = fabs(*al);
               if (big < trybig) { big = trybig; mu = nrows - i - 1; }
               while (l--) *(++al) -= (mult * *(++aj));
            }
         }
      }
      else sing = true;
      if (++k == nrows) break;          // so next line won't overflow
      akk += nrows + 1;
   }
}

void CroutMatrix::lubksb(Real* B, int mini)
{
   REPORT
   // this has been adapted from Numerical Recipes in C. The code has been
   // substantially streamlined, so I do not think much of the original
   // copyright remains. However there is not much opportunity for
   // variation in the code, so it is still similar to the NR code.
   // I follow the NR code in skipping over initial zeros in the B vector.

   Tracer trace("Crout(lubksb)");
   if (sing) Throw(SingularException(*this));
   int i, j, ii = nrows;            // ii initialised : B might be all zeros


   // scan for first non-zero in B
   for (i = 0; i < nrows; i++)
   {
      int ip = indx[i]; Real temp = B[ip]; B[ip] = B[i]; B[i] = temp;
      if (temp != 0.0) { ii = i; break; }
   }

   Real* bi; Real* ai;
   i = ii + 1;

   if (i < nrows)
   {
      bi = B + ii; ai = store + ii + i * nrows;
      for (;;)
      {
         int ip = indx[i]; Real sum = B[ip]; B[ip] = B[i];
         Real* aij = ai; Real* bj = bi; j = i - ii;
         while (j--) sum -= *aij++ * *bj++;
         B[i] = sum;
         if (++i == nrows) break;
         ai += nrows;
      }
   }

   ai = store + nrows * nrows;

   for (i = nrows - 1; i >= mini; i--)
   {
      Real* bj = B+i; ai -= nrows; Real* ajx = ai+i;
      Real sum = *bj; Real diag = *ajx;
      j = nrows - i; while(--j) sum -= *(++ajx) * *(++bj);
      B[i] = sum / diag;
   }
}

/****************************** scalar functions ****************************/

inline Real square(Real x) { return x*x; }

Real GeneralMatrix::SumSquare() const
{
   REPORT
   Real sum = 0.0; int i = storage; Real* s = store;
   while (i--) sum += square(*s++);
   ((GeneralMatrix&)*this).tDelete(); return sum;
}

Real GeneralMatrix::SumAbsoluteValue() const
{
   REPORT
   Real sum = 0.0; int i = storage; Real* s = store;
   while (i--) sum += fabs(*s++);
   ((GeneralMatrix&)*this).tDelete(); return sum;
}

Real GeneralMatrix::Sum() const
{
   REPORT
   Real sum = 0.0; int i = storage; Real* s = store;
   while (i--) sum += *s++;
   ((GeneralMatrix&)*this).tDelete(); return sum;
}

// maxima and minima

// There are three sets of routines
// MaximumAbsoluteValue, MinimumAbsoluteValue, Maximum, Minimum
// ... these find just the maxima and minima
// MaximumAbsoluteValue1, MinimumAbsoluteValue1, Maximum1, Minimum1
// ... these find the maxima and minima and their locations in a
//     one dimensional object
// MaximumAbsoluteValue2, MinimumAbsoluteValue2, Maximum2, Minimum2
// ... these find the maxima and minima and their locations in a
//     two dimensional object

// If the matrix has no values throw an exception

// If we do not want the location find the maximum or minimum on the
// array stored by GeneralMatrix
// This won't work for BandMatrices. We call ClearCorner for
// MaximumAbsoluteValue but for the others use the AbsoluteMinimumValue2
// version and discard the location.

// For one dimensional objects, when we want the location of the
// maximum or minimum, work with the array stored by GeneralMatrix

// For two dimensional objects where we want the location of the maximum or
// minimum proceed as follows:

// For rectangular matrices use the array stored by GeneralMatrix and
// deduce the location from the location in the GeneralMatrix

// For other two dimensional matrices use the Matrix Row routine to find the
// maximum or minimum for each row.

static void NullMatrixError(const GeneralMatrix* gm)
{
   ((GeneralMatrix&)*gm).tDelete();
   Throw(ProgramException("Maximum or minimum of null matrix"));
}

Real GeneralMatrix::MaximumAbsoluteValue() const
{
   REPORT
   if (storage == 0) NullMatrixError(this);
   Real maxval = 0.0; int l = storage; Real* s = store;
   while (l--) { Real a = fabs(*s++); if (maxval < a) maxval = a; }
   ((GeneralMatrix&)*this).tDelete(); return maxval;
}

Real GeneralMatrix::MaximumAbsoluteValue1(int& i) const
{
   REPORT
   if (storage == 0) NullMatrixError(this);
   Real maxval = 0.0; int l = storage; Real* s = store; int li = storage;
   while (l--)
      { Real a = fabs(*s++); if (maxval <= a) { maxval = a; li = l; }  }
   i = storage - li;
   ((GeneralMatrix&)*this).tDelete(); return maxval;
}

Real GeneralMatrix::MinimumAbsoluteValue() const
{
   REPORT
   if (storage == 0) NullMatrixError(this);
   int l = storage - 1; Real* s = store; Real minval = fabs(*s++);
   while (l--) { Real a = fabs(*s++); if (minval > a) minval = a; }
   ((GeneralMatrix&)*this).tDelete(); return minval;
}

Real GeneralMatrix::MinimumAbsoluteValue1(int& i) const
{
   REPORT
   if (storage == 0) NullMatrixError(this);
   int l = storage - 1; Real* s = store; Real minval = fabs(*s++); int li = l;
   while (l--)
      { Real a = fabs(*s++); if (minval >= a) { minval = a; li = l; }  }
   i = storage - li;
   ((GeneralMatrix&)*this).tDelete(); return minval;
}

Real GeneralMatrix::Maximum() const
{
   REPORT
   if (storage == 0) NullMatrixError(this);
   int l = storage - 1; Real* s = store; Real maxval = *s++;
   while (l--) { Real a = *s++; if (maxval < a) maxval = a; }
   ((GeneralMatrix&)*this).tDelete(); return maxval;
}

Real GeneralMatrix::Maximum1(int& i) const
{
   REPORT
   if (storage == 0) NullMatrixError(this);
   int l = storage - 1; Real* s = store; Real maxval = *s++; int li = l;
   while (l--) { Real a = *s++; if (maxval <= a) { maxval = a; li = l; } }
   i = storage - li;
   ((GeneralMatrix&)*this).tDelete(); return maxval;
}

Real GeneralMatrix::Minimum() const
{
   REPORT
   if (storage == 0) NullMatrixError(this);
   int l = storage - 1; Real* s = store; Real minval = *s++;
   while (l--) { Real a = *s++; if (minval > a) minval = a; }
   ((GeneralMatrix&)*this).tDelete(); return minval;
}

Real GeneralMatrix::Minimum1(int& i) const
{
   REPORT
   if (storage == 0) NullMatrixError(this);
   int l = storage - 1; Real* s = store; Real minval = *s++; int li = l;
   while (l--) { Real a = *s++; if (minval >= a) { minval = a; li = l; } }
   i = storage - li;
   ((GeneralMatrix&)*this).tDelete(); return minval;
}

Real GeneralMatrix::MaximumAbsoluteValue2(int& i, int& j) const
{
   REPORT
   if (storage == 0) NullMatrixError(this);
   Real maxval = 0.0; int nr = Nrows();
   MatrixRow mr((GeneralMatrix*)this, LoadOnEntry+DirectPart);
   for (int r = 1; r <= nr; r++)
   {
      int c; maxval = mr.MaximumAbsoluteValue1(maxval, c);
      if (c > 0) { i = r; j = c; }
      mr.Next();
   }
   ((GeneralMatrix&)*this).tDelete(); return maxval;
}

Real GeneralMatrix::MinimumAbsoluteValue2(int& i, int& j) const
{
   REPORT
   if (storage == 0)  NullMatrixError(this);
   Real minval = FloatingPointPrecision::Maximum(); int nr = Nrows();
   MatrixRow mr((GeneralMatrix*)this, LoadOnEntry+DirectPart);
   for (int r = 1; r <= nr; r++)
   {
      int c; minval = mr.MinimumAbsoluteValue1(minval, c);
      if (c > 0) { i = r; j = c; }
      mr.Next();
   }
   ((GeneralMatrix&)*this).tDelete(); return minval;
}

Real GeneralMatrix::Maximum2(int& i, int& j) const
{
   REPORT
   if (storage == 0) NullMatrixError(this);
   Real maxval = -FloatingPointPrecision::Maximum(); int nr = Nrows();
   MatrixRow mr((GeneralMatrix*)this, LoadOnEntry+DirectPart);
   for (int r = 1; r <= nr; r++)
   {
      int c; maxval = mr.Maximum1(maxval, c);
      if (c > 0) { i = r; j = c; }
      mr.Next();
   }
   ((GeneralMatrix&)*this).tDelete(); return maxval;
}

Real GeneralMatrix::Minimum2(int& i, int& j) const
{
   REPORT
   if (storage == 0) NullMatrixError(this);
   Real minval = FloatingPointPrecision::Maximum(); int nr = Nrows();
   MatrixRow mr((GeneralMatrix*)this, LoadOnEntry+DirectPart);
   for (int r = 1; r <= nr; r++)
   {
      int c; minval = mr.Minimum1(minval, c);
      if (c > 0) { i = r; j = c; }
      mr.Next();
   }
   ((GeneralMatrix&)*this).tDelete(); return minval;
}

Real Matrix::MaximumAbsoluteValue2(int& i, int& j) const
{
   REPORT
   int k; Real m = GeneralMatrix::MaximumAbsoluteValue1(k); k--;
   i = k / Ncols(); j = k - i * Ncols(); i++; j++;
   return m;
}

Real Matrix::MinimumAbsoluteValue2(int& i, int& j) const
{
   REPORT
   int k; Real m = GeneralMatrix::MinimumAbsoluteValue1(k); k--;
   i = k / Ncols(); j = k - i * Ncols(); i++; j++;
   return m;
}

Real Matrix::Maximum2(int& i, int& j) const
{
   REPORT
   int k; Real m = GeneralMatrix::Maximum1(k); k--;
   i = k / Ncols(); j = k - i * Ncols(); i++; j++;
   return m;
}

Real Matrix::Minimum2(int& i, int& j) const
{
   REPORT
   int k; Real m = GeneralMatrix::Minimum1(k); k--;
   i = k / Ncols(); j = k - i * Ncols(); i++; j++;
   return m;
}

Real SymmetricMatrix::SumSquare() const
{
   REPORT
   Real sum1 = 0.0; Real sum2 = 0.0; Real* s = store; int nr = nrows;
   for (int i = 0; i<nr; i++)
   {
      int j = i;
      while (j--) sum2 += square(*s++);
      sum1 += square(*s++);
   }
   ((GeneralMatrix&)*this).tDelete(); return sum1 + 2.0 * sum2;
}

Real SymmetricMatrix::SumAbsoluteValue() const
{
   REPORT
   Real sum1 = 0.0; Real sum2 = 0.0; Real* s = store; int nr = nrows;
   for (int i = 0; i<nr; i++)
   {
      int j = i;
      while (j--) sum2 += fabs(*s++);
      sum1 += fabs(*s++);
   }
   ((GeneralMatrix&)*this).tDelete(); return sum1 + 2.0 * sum2;
}

Real IdentityMatrix::SumAbsoluteValue() const
   { REPORT  return fabs(Trace()); }    // no need to do tDelete?

Real SymmetricMatrix::Sum() const
{
   REPORT
   Real sum1 = 0.0; Real sum2 = 0.0; Real* s = store; int nr = nrows;
   for (int i = 0; i<nr; i++)
   {
      int j = i;
      while (j--) sum2 += *s++;
      sum1 += *s++;
   }
   ((GeneralMatrix&)*this).tDelete(); return sum1 + 2.0 * sum2;
}

Real IdentityMatrix::SumSquare() const
{
   Real sum = *store * *store * nrows;
   ((GeneralMatrix&)*this).tDelete(); return sum;
}


Real BaseMatrix::SumSquare() const
{
   REPORT GeneralMatrix* gm = ((BaseMatrix&)*this).Evaluate();
   Real s = gm->SumSquare(); return s;
}

Real BaseMatrix::NormFrobenius() const
   { REPORT  return sqrt(SumSquare()); }

Real BaseMatrix::SumAbsoluteValue() const
{
   REPORT GeneralMatrix* gm = ((BaseMatrix&)*this).Evaluate();
   Real s = gm->SumAbsoluteValue(); return s;
}

Real BaseMatrix::Sum() const
{
   REPORT GeneralMatrix* gm = ((BaseMatrix&)*this).Evaluate();
   Real s = gm->Sum(); return s;
}

Real BaseMatrix::MaximumAbsoluteValue() const
{
   REPORT GeneralMatrix* gm = ((BaseMatrix&)*this).Evaluate();
   Real s = gm->MaximumAbsoluteValue(); return s;
}

Real BaseMatrix::MaximumAbsoluteValue1(int& i) const
{
   REPORT GeneralMatrix* gm = ((BaseMatrix&)*this).Evaluate();
   Real s = gm->MaximumAbsoluteValue1(i); return s;
}

Real BaseMatrix::MaximumAbsoluteValue2(int& i, int& j) const
{
   REPORT GeneralMatrix* gm = ((BaseMatrix&)*this).Evaluate();
   Real s = gm->MaximumAbsoluteValue2(i, j); return s;
}

Real BaseMatrix::MinimumAbsoluteValue() const
{
   REPORT GeneralMatrix* gm = ((BaseMatrix&)*this).Evaluate();
   Real s = gm->MinimumAbsoluteValue(); return s;
}

Real BaseMatrix::MinimumAbsoluteValue1(int& i) const
{
   REPORT GeneralMatrix* gm = ((BaseMatrix&)*this).Evaluate();
   Real s = gm->MinimumAbsoluteValue1(i); return s;
}

Real BaseMatrix::MinimumAbsoluteValue2(int& i, int& j) const
{
   REPORT GeneralMatrix* gm = ((BaseMatrix&)*this).Evaluate();
   Real s = gm->MinimumAbsoluteValue2(i, j); return s;
}

Real BaseMatrix::Maximum() const
{
   REPORT GeneralMatrix* gm = ((BaseMatrix&)*this).Evaluate();
   Real s = gm->Maximum(); return s;
}

Real BaseMatrix::Maximum1(int& i) const
{
   REPORT GeneralMatrix* gm = ((BaseMatrix&)*this).Evaluate();
   Real s = gm->Maximum1(i); return s;
}

Real BaseMatrix::Maximum2(int& i, int& j) const
{
   REPORT GeneralMatrix* gm = ((BaseMatrix&)*this).Evaluate();
   Real s = gm->Maximum2(i, j); return s;
}

Real BaseMatrix::Minimum() const
{
   REPORT GeneralMatrix* gm = ((BaseMatrix&)*this).Evaluate();
   Real s = gm->Minimum(); return s;
}

Real BaseMatrix::Minimum1(int& i) const
{
   REPORT GeneralMatrix* gm = ((BaseMatrix&)*this).Evaluate();
   Real s = gm->Minimum1(i); return s;
}

Real BaseMatrix::Minimum2(int& i, int& j) const
{
   REPORT GeneralMatrix* gm = ((BaseMatrix&)*this).Evaluate();
   Real s = gm->Minimum2(i, j); return s;
}

Real DotProduct(const Matrix& A, const Matrix& B)
{
   REPORT
   int n = A.storage;
   if (n != B.storage) Throw(IncompatibleDimensionsException(A,B));
   Real sum = 0.0; Real* a = A.store; Real* b = B.store;
   while (n--) sum += *a++ * *b++;
   return sum;
}

Real Matrix::Trace() const
{
   REPORT
   Tracer trace("Trace");
   int i = nrows; int d = i+1;
   if (i != ncols) Throw(NotSquareException(*this));
   Real sum = 0.0; Real* s = store;
//   while (i--) { sum += *s; s += d; }
   if (i) for (;;) { sum += *s; if (!(--i)) break; s += d; }
   ((GeneralMatrix&)*this).tDelete(); return sum;
}

Real DiagonalMatrix::Trace() const
{
   REPORT
   int i = nrows; Real sum = 0.0; Real* s = store;
   while (i--) sum += *s++;
   ((GeneralMatrix&)*this).tDelete(); return sum;
}

Real SymmetricMatrix::Trace() const
{
   REPORT
   int i = nrows; Real sum = 0.0; Real* s = store; int j = 2;
   // while (i--) { sum += *s; s += j++; }
   if (i) for (;;) { sum += *s; if (!(--i)) break; s += j++; }
   ((GeneralMatrix&)*this).tDelete(); return sum;
}

Real LowerTriangularMatrix::Trace() const
{
   REPORT
   int i = nrows; Real sum = 0.0; Real* s = store; int j = 2;
   // while (i--) { sum += *s; s += j++; }
   if (i) for (;;) { sum += *s; if (!(--i)) break; s += j++; }
   ((GeneralMatrix&)*this).tDelete(); return sum;
}

Real UpperTriangularMatrix::Trace() const
{
   REPORT
   int i = nrows; Real sum = 0.0; Real* s = store;
   while (i) { sum += *s; s += i--; }             // won t cause a problem
   ((GeneralMatrix&)*this).tDelete(); return sum;
}

Real BandMatrix::Trace() const
{
   REPORT
   int i = nrows; int w = lower+upper+1;
   Real sum = 0.0; Real* s = store+lower;
   // while (i--) { sum += *s; s += w; }
   if (i) for (;;) { sum += *s; if (!(--i)) break; s += w; }
   ((GeneralMatrix&)*this).tDelete(); return sum;
}

Real SymmetricBandMatrix::Trace() const
{
   REPORT
   int i = nrows; int w = lower+1;
   Real sum = 0.0; Real* s = store+lower;
   // while (i--) { sum += *s; s += w; }
   if (i) for (;;) { sum += *s; if (!(--i)) break; s += w; }
   ((GeneralMatrix&)*this).tDelete(); return sum;
}

Real IdentityMatrix::Trace() const
{
   Real sum = *store * nrows;
   ((GeneralMatrix&)*this).tDelete(); return sum;
}


Real BaseMatrix::Trace() const
{
   REPORT
   MatrixType Diag = MatrixType::Dg; Diag.SetDataLossOK();
   GeneralMatrix* gm = ((BaseMatrix&)*this).Evaluate(Diag);
   Real sum = gm->Trace(); return sum;
}

void LogAndSign::operator*=(Real x)
{
   if (x > 0.0) { log_value += log(x); }
   else if (x < 0.0) { log_value += log(-x); sign = -sign; }
   else sign = 0;
}

void LogAndSign::PowEq(int k)
{
   if (sign)
   {
      log_value *= k;
      if ( (k & 1) == 0 ) sign = 1;
   }
}

Real LogAndSign::Value() const
{
   Tracer et("LogAndSign::Value");
   if (log_value >= FloatingPointPrecision::LnMaximum())
      Throw(OverflowException("Overflow in exponential"));
   return sign * exp(log_value);
}

LogAndSign::LogAndSign(Real f)
{
   if (f == 0.0) { log_value = 0.0; sign = 0; return; }
   else if (f < 0.0) { sign = -1; f = -f; }
   else sign = 1;
   log_value = log(f);
}

LogAndSign DiagonalMatrix::LogDeterminant() const
{
   REPORT
   int i = nrows; LogAndSign sum; Real* s = store;
   while (i--) sum *= *s++;
   ((GeneralMatrix&)*this).tDelete(); return sum;
}

LogAndSign LowerTriangularMatrix::LogDeterminant() const
{
   REPORT
   int i = nrows; LogAndSign sum; Real* s = store; int j = 2;
   // while (i--) { sum *= *s; s += j++; }
   if (i) for(;;) { sum *= *s; if (!(--i)) break; s += j++; }
   ((GeneralMatrix&)*this).tDelete(); return sum;
}

LogAndSign UpperTriangularMatrix::LogDeterminant() const
{
   REPORT
   int i = nrows; LogAndSign sum; Real* s = store;
   while (i) { sum *= *s; s += i--; }
   ((GeneralMatrix&)*this).tDelete(); return sum;
}

LogAndSign IdentityMatrix::LogDeterminant() const
{
   REPORT
   int i = nrows; LogAndSign sum;
   if (i > 0) { sum = *store; sum.PowEq(i); }
   ((GeneralMatrix&)*this).tDelete(); return sum;
}

LogAndSign BaseMatrix::LogDeterminant() const
{
   REPORT GeneralMatrix* gm = ((BaseMatrix&)*this).Evaluate();
   LogAndSign sum = gm->LogDeterminant(); return sum;
}

LogAndSign GeneralMatrix::LogDeterminant() const
{
   REPORT
   Tracer tr("LogDeterminant");
   if (nrows != ncols) Throw(NotSquareException(*this));
   CroutMatrix C(*this); return C.LogDeterminant();
}

LogAndSign CroutMatrix::LogDeterminant() const
{
   REPORT
   if (sing) return 0.0;
   int i = nrows; int dd = i+1; LogAndSign sum; Real* s = store;
   if (i) for(;;)
   {
      sum *= *s;
      if (!(--i)) break;
      s += dd;
   }
   if (!d) sum.ChangeSign(); return sum;

}

Real BaseMatrix::Determinant() const
{
   REPORT
   Tracer tr("Determinant");
   REPORT GeneralMatrix* gm = ((BaseMatrix&)*this).Evaluate();
   LogAndSign ld = gm->LogDeterminant();
   return ld.Value();
}





LinearEquationSolver::LinearEquationSolver(const BaseMatrix& bm)
{
   gm = ( ((BaseMatrix&)bm).Evaluate() )->MakeSolver();
   if (gm==&bm) { REPORT  gm = gm->Image(); }
   // want a copy if  *gm is actually bm
   else { REPORT  gm->Protect(); }
}


#ifdef use_namespace
}
#endif

