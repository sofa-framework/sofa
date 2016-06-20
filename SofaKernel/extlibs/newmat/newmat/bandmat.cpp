//$$ bandmat.cpp                     Band matrix definitions

// Copyright (C) 1991,2,3,4,9: R B Davies

#define WANT_MATH                    // include.h will get math fns

//#define WANT_STREAM

#include "include.h"

#include "newmat.h"
#include "newmatrc.h"

#ifdef use_namespace
namespace NEWMAT {
#endif



#ifdef DO_REPORT
#define REPORT { static ExeCounter ExeCount(__LINE__,10); ++ExeCount; }
#else
#define REPORT {}
#endif

static inline int my_min(int x, int y) { return x < y ? x : y; }
static inline int my_max(int x, int y) { return x > y ? x : y; }


BandMatrix::BandMatrix(const BaseMatrix& M)
{
   REPORT // CheckConversion(M);
   // MatrixConversionCheck mcc;
   GeneralMatrix* gmx=((BaseMatrix&)M).Evaluate(MatrixType::BM);
   GetMatrix(gmx); CornerClear();
}

void BandMatrix::SetParameters(const GeneralMatrix* gmx)
{
   REPORT
   MatrixBandWidth bw = gmx->BandWidth();
   lower = bw.lower; upper = bw.upper;
}

void BandMatrix::ReSize(int n, int lb, int ub)
{
   REPORT
   Tracer tr("BandMatrix::ReSize");
   if (lb<0 || ub<0) Throw(ProgramException("Undefined bandwidth"));
   lower = (lb<=n) ? lb : n-1; upper = (ub<=n) ? ub : n-1;
   GeneralMatrix::ReSize(n,n,n*(lower+1+upper)); CornerClear();
}

// SimpleAddOK shows when we can add etc two matrices by a simple vector add
// and when we can add one matrix into another
// *gm must be the same type as *this
// return 0 if simple add is OK
// return 1 if we can add into *gm only
// return 2 if we can add into *this only
// return 3 if we can't add either way
// For SP this will still be valid if we swap 1 and 2

short BandMatrix::SimpleAddOK(const GeneralMatrix* gm)
{
   const BandMatrix* bm = (const BandMatrix*)gm;
   if (bm->lower == lower && bm->upper == upper) { REPORT return 0; }
   else if (bm->lower >= lower && bm->upper >= upper) { REPORT return 1; }
   else if (bm->lower <= lower && bm->upper <= upper) { REPORT return 2; }
   else { REPORT return 3; }
}

short SymmetricBandMatrix::SimpleAddOK(const GeneralMatrix* gm)
{
   const SymmetricBandMatrix* bm = (const SymmetricBandMatrix*)gm;
   if (bm->lower == lower) { REPORT return 0; }
   else if (bm->lower > lower) { REPORT return 1; }
   else { REPORT return 2; }
}

void UpperBandMatrix::ReSize(int n, int lb, int ub)
{
   REPORT
   if (lb != 0)
   {
      Tracer tr("UpperBandMatrix::ReSize");
      Throw(ProgramException("UpperBandMatrix with non-zero lower band" ));
   }
   BandMatrix::ReSize(n, lb, ub);
}

void LowerBandMatrix::ReSize(int n, int lb, int ub)
{
   REPORT
   if (ub != 0)
   {
      Tracer tr("LowerBandMatrix::ReSize");
      Throw(ProgramException("LowerBandMatrix with non-zero upper band" ));
   }
   BandMatrix::ReSize(n, lb, ub);
}

void BandMatrix::ReSize(const GeneralMatrix& A)
{
   REPORT
   int n = A.Nrows();
   if (n != A.Ncols())
   {
      Tracer tr("BandMatrix::ReSize(GM)");
      Throw(NotSquareException(*this));
   }
   MatrixBandWidth mbw = A.BandWidth();
   ReSize(n, mbw.Lower(), mbw.Upper());
}

bool BandMatrix::SameStorageType(const GeneralMatrix& A) const
{
   if (Type() != A.Type()) { REPORT return false; }
   REPORT
   return BandWidth() == A.BandWidth();
}

void BandMatrix::ReSizeForAdd(const GeneralMatrix& A, const GeneralMatrix& B)
{
   REPORT
   Tracer tr("BandMatrix::ReSizeForAdd");
   MatrixBandWidth A_BW = A.BandWidth(); MatrixBandWidth B_BW = B.BandWidth();
   if ((A_BW.Lower() < 0) | (A_BW.Upper() < 0) | (B_BW.Lower() < 0)
      | (A_BW.Upper() < 0))
         Throw(ProgramException("Can't ReSize to BandMatrix" ));
   // already know A and B are square
   ReSize(A.Nrows(), my_max(A_BW.Lower(), B_BW.Lower()),
      my_max(A_BW.Upper(), B_BW.Upper()));
}

void BandMatrix::ReSizeForSP(const GeneralMatrix& A, const GeneralMatrix& B)
{
   REPORT
   Tracer tr("BandMatrix::ReSizeForSP");
   MatrixBandWidth A_BW = A.BandWidth(); MatrixBandWidth B_BW = B.BandWidth();
   if ((A_BW.Lower() < 0) | (A_BW.Upper() < 0) | (B_BW.Lower() < 0)
      | (A_BW.Upper() < 0))
         Throw(ProgramException("Can't ReSize to BandMatrix" ));
   // already know A and B are square
   ReSize(A.Nrows(), my_min(A_BW.Lower(), B_BW.Lower()),
      my_min(A_BW.Upper(), B_BW.Upper()));
}


void BandMatrix::operator=(const BaseMatrix& X)
{
   REPORT // CheckConversion(X);
   // MatrixConversionCheck mcc;
   Eq(X,MatrixType::BM); CornerClear();
}

void BandMatrix::CornerClear() const
{
   // set unused parts of BandMatrix to zero
   REPORT
   int i = lower; Real* s = store; int bw = lower + 1 + upper;
   while (i)
      { int j = i--; Real* sj = s; s += bw; while (j--) *sj++ = 0.0; }
   i = upper; s = store + storage;
   while (i)
      { int j = i--; Real* sj = s; s -= bw; while (j--) *(--sj) = 0.0; }
}

MatrixBandWidth MatrixBandWidth::operator+(const MatrixBandWidth& bw) const
{
   REPORT
   int l = bw.lower; int u = bw.upper;
   l = (lower < 0 || l < 0) ? -1 : (lower > l) ? lower : l;
   u = (upper < 0 || u < 0) ? -1 : (upper > u) ? upper : u;
   return MatrixBandWidth(l,u);
}

MatrixBandWidth MatrixBandWidth::operator*(const MatrixBandWidth& bw) const
{
   REPORT
   int l = bw.lower; int u = bw.upper;
   l = (lower < 0 || l < 0) ? -1 : lower+l;
   u = (upper < 0 || u < 0) ? -1 : upper+u;
   return MatrixBandWidth(l,u);
}

MatrixBandWidth MatrixBandWidth::minimum(const MatrixBandWidth& bw) const
{
   REPORT
   int l = bw.lower; int u = bw.upper;
   if ((lower >= 0) && ( (l < 0) || (l > lower) )) l = lower;
   if ((upper >= 0) && ( (u < 0) || (u > upper) )) u = upper;
   return MatrixBandWidth(l,u);
}

UpperBandMatrix::UpperBandMatrix(const BaseMatrix& M)
{
   REPORT // CheckConversion(M);
   // MatrixConversionCheck mcc;
   GeneralMatrix* gmx=((BaseMatrix&)M).Evaluate(MatrixType::UB);
   GetMatrix(gmx); CornerClear();
}

void UpperBandMatrix::operator=(const BaseMatrix& X)
{
   REPORT // CheckConversion(X);
   // MatrixConversionCheck mcc;
   Eq(X,MatrixType::UB); CornerClear();
}

LowerBandMatrix::LowerBandMatrix(const BaseMatrix& M)
{
   REPORT // CheckConversion(M);
   // MatrixConversionCheck mcc;
   GeneralMatrix* gmx=((BaseMatrix&)M).Evaluate(MatrixType::LB);
   GetMatrix(gmx); CornerClear();
}

void LowerBandMatrix::operator=(const BaseMatrix& X)
{
   REPORT // CheckConversion(X);
   // MatrixConversionCheck mcc;
   Eq(X,MatrixType::LB); CornerClear();
}

BandLUMatrix::BandLUMatrix(const BaseMatrix& m)
{
   REPORT
   Tracer tr("BandLUMatrix");
   storage2 = 0; store2 = 0;  // in event of exception during build
   GeneralMatrix* gm = ((BaseMatrix&)m).Evaluate(MatrixType::BM);
   m1 = ((BandMatrix*)gm)->lower; m2 = ((BandMatrix*)gm)->upper;
   GetMatrix(gm);
   if (nrows!=ncols) Throw(NotSquareException(*this));
   d = true; sing = false;
   indx = new int [nrows]; MatrixErrorNoSpace(indx);
   MONITOR_INT_NEW("Index (BndLUMat)",nrows,indx)
   storage2 = nrows * m1;
   store2 = new Real [storage2]; MatrixErrorNoSpace(store2);
   MONITOR_REAL_NEW("Make (BandLUMat)",storage2,store2)
   ludcmp();
}

BandLUMatrix::~BandLUMatrix()
{
   REPORT
   MONITOR_INT_DELETE("Index (BndLUMat)",nrows,indx)
   MONITOR_REAL_DELETE("Delete (BndLUMt)",storage2,store2)
   delete [] indx; delete [] store2;
}

MatrixType BandLUMatrix::Type() const { REPORT return MatrixType::BC; }


LogAndSign BandLUMatrix::LogDeterminant() const
{
   REPORT
   if (sing) return 0.0;
   Real* a = store; int w = m1+1+m2; LogAndSign sum; int i = nrows;
   // while (i--) { sum *= *a; a += w; }
   if (i) for (;;) { sum *= *a; if (!(--i)) break; a += w; }
   if (!d) sum.ChangeSign(); return sum;
}

GeneralMatrix* BandMatrix::MakeSolver()
{
   REPORT
   GeneralMatrix* gm = new BandLUMatrix(*this);
   MatrixErrorNoSpace(gm); gm->ReleaseAndDelete(); return gm;
}


void BandLUMatrix::ludcmp()
{
   REPORT
   Real* a = store2; int i = storage2;
   // clear store2 - so unused locations are always zero -
   // required by operator==
   while (i--) *a++ = 0.0;
   a = store;
   i = m1; int j = m2; int k; int n = nrows; int w = m1 + 1 + m2;
   while (i)
   {
      Real* ai = a + i;
      k = ++j; while (k--) *a++ = *ai++;
      k = i--; while (k--) *a++ = 0.0;
   }

   a = store; int l = m1;
   for (k=0; k<n; k++)
   {
      Real x = *a; i = k; Real* aj = a;
      if (l < n) l++;
      for (j=k+1; j<l; j++)
         { aj += w; if (fabs(x) < fabs(*aj)) { x = *aj; i = j; } }
      indx[k] = i;
      if (x==0) { sing = true; return; }
      if (i!=k)
      {
         d = !d; Real* ak = a; Real* ai = store + i * w; j = w;
         while (j--) { x = *ak; *ak++ = *ai; *ai++ = x; }
      }
      aj = a + w; Real* m = store2 + m1 * k;
      for (j=k+1; j<l; j++)
      {
         *m++ = x = *aj / *a; i = w; Real* ak = a;
	 while (--i) { Real* aj1 = aj++; *aj1 = *aj - x * *(++ak); }
         *aj++ = 0.0;
      }
      a += w;
   }
}

void BandLUMatrix::lubksb(Real* B, int mini)
{
   REPORT
   Tracer tr("BandLUMatrix::lubksb");
   if (sing) Throw(SingularException(*this));
   int n = nrows; int l = m1; int w = m1 + 1 + m2;

   for (int k=0; k<n; k++)
   {
      int i = indx[k];
      if (i!=k) { Real x=B[k]; B[k]=B[i]; B[i]=x; }
      if (l<n) l++;
      Real* m = store2 + k*m1; Real* b = B+k; Real* bi = b;
      for (i=k+1; i<l; i++)  *(++bi) -= *m++ * *b;
   }

   l = -m1;
   for (int i = n-1; i>=mini; i--)
   {
      Real* b = B + i; Real* bk = b; Real x = *bk;
      Real* a = store + w*i; Real y = *a;
      int k = l+m1; while (k--) x -=  *(++a) * *(++bk);
      *b = x / y;
      if (l < m2) l++;
   }
}

void BandLUMatrix::Solver(MatrixColX& mcout, const MatrixColX& mcin)
{
   REPORT
   int i = mcin.skip; Real* el = mcin.data-i; Real* el1=el;
   while (i--) *el++ = 0.0;
   el += mcin.storage; i = nrows - mcin.skip - mcin.storage;
   while (i--) *el++ = 0.0;
   lubksb(el1, mcout.skip);
}

// Do we need check for entirely zero output?


void UpperBandMatrix::Solver(MatrixColX& mcout,
   const MatrixColX& mcin)
{
   REPORT
   int i = mcin.skip-mcout.skip; Real* elx = mcin.data-i;
   while (i-- > 0) *elx++ = 0.0;
   int nr = mcin.skip+mcin.storage;
   elx = mcin.data+mcin.storage; Real* el = elx;
   int j = mcout.skip+mcout.storage-nr; i = nr-mcout.skip;
   while (j-- > 0) *elx++ = 0.0;

   Real* Ael = store + (upper+1)*(i-1)+1; j = 0;
   if (i > 0) for(;;)
   {
      elx = el; Real sum = 0.0; int jx = j;
      while (jx--) sum += *(--Ael) * *(--elx);
      elx--; *elx = (*elx - sum) / *(--Ael);
      if (--i <= 0) break;
      if (j<upper) Ael -= upper - (++j); else el--;
   }
}

void LowerBandMatrix::Solver(MatrixColX& mcout,
   const MatrixColX& mcin)
{
   REPORT
   int i = mcin.skip-mcout.skip; Real* elx = mcin.data-i;
   while (i-- > 0) *elx++ = 0.0;
   int nc = mcin.skip; i = nc+mcin.storage; elx = mcin.data+mcin.storage;
   int nr = mcout.skip+mcout.storage; int j = nr-i; i = nr-nc;
   while (j-- > 0) *elx++ = 0.0;

   Real* el = mcin.data; Real* Ael = store + (lower+1)*nc + lower; j = 0;
   if (i > 0) for(;;)
   {
      elx = el; Real sum = 0.0; int jx = j;
      while (jx--) sum += *Ael++ * *elx++;
      *elx = (*elx - sum) / *Ael++;
      if (--i <= 0) break;
      if (j<lower) Ael += lower - (++j); else el++;
   }
}


LogAndSign BandMatrix::LogDeterminant() const
{
   REPORT
   BandLUMatrix C(*this); return C.LogDeterminant();
}

LogAndSign LowerBandMatrix::LogDeterminant() const
{
   REPORT
   int i = nrows; LogAndSign sum; Real* s = store + lower; int j = lower + 1;
//   while (i--) { sum *= *s; s += j; }
   if (i) for (;;) { sum *= *s; if (!(--i)) break; s += j; }
   ((GeneralMatrix&)*this).tDelete(); return sum;
}

LogAndSign UpperBandMatrix::LogDeterminant() const
{
   REPORT
   int i = nrows; LogAndSign sum; Real* s = store; int j = upper + 1;
//   while (i--) { sum *= *s; s += j; }
   if (i) for (;;) { sum *= *s; if (!(--i)) break; s += j; }
   ((GeneralMatrix&)*this).tDelete(); return sum;
}

GeneralMatrix* SymmetricBandMatrix::MakeSolver()
{
   REPORT
   GeneralMatrix* gm = new BandLUMatrix(*this);
   MatrixErrorNoSpace(gm); gm->ReleaseAndDelete(); return gm;
}

SymmetricBandMatrix::SymmetricBandMatrix(const BaseMatrix& M)
{
   REPORT  // CheckConversion(M);
   // MatrixConversionCheck mcc;
   GeneralMatrix* gmx=((BaseMatrix&)M).Evaluate(MatrixType::SB);
   GetMatrix(gmx);
}

GeneralMatrix* SymmetricBandMatrix::Transpose(TransposedMatrix*, MatrixType mt)
{ REPORT  return Evaluate(mt); }

LogAndSign SymmetricBandMatrix::LogDeterminant() const
{
   REPORT
   BandLUMatrix C(*this); return C.LogDeterminant();
}

void SymmetricBandMatrix::SetParameters(const GeneralMatrix* gmx)
{ REPORT lower = gmx->BandWidth().lower; }

void SymmetricBandMatrix::ReSize(int n, int lb)
{
   REPORT
   Tracer tr("SymmetricBandMatrix::ReSize");
   if (lb<0) Throw(ProgramException("Undefined bandwidth"));
   lower = (lb<=n) ? lb : n-1;
   GeneralMatrix::ReSize(n,n,n*(lower+1));
}

void SymmetricBandMatrix::ReSize(const GeneralMatrix& A)
{
   REPORT
   int n = A.Nrows();
   if (n != A.Ncols())
   {
      Tracer tr("SymmetricBandMatrix::ReSize(GM)");
      Throw(NotSquareException(*this));
   }
   MatrixBandWidth mbw = A.BandWidth(); int b = mbw.Lower();
   if (b != mbw.Upper())
   {
      Tracer tr("SymmetricBandMatrix::ReSize(GM)");
      Throw(ProgramException("Upper and lower band-widths not equal"));
   }
   ReSize(n, b);
}

bool SymmetricBandMatrix::SameStorageType(const GeneralMatrix& A) const
{
   if (Type() != A.Type()) { REPORT return false; }
   REPORT
   return BandWidth() == A.BandWidth();
}

void SymmetricBandMatrix::ReSizeForAdd(const GeneralMatrix& A,
   const GeneralMatrix& B)
{
   REPORT
   Tracer tr("SymmetricBandMatrix::ReSizeForAdd");
   MatrixBandWidth A_BW = A.BandWidth(); MatrixBandWidth B_BW = B.BandWidth();
   if ((A_BW.Lower() < 0) | (B_BW.Lower() < 0))
         Throw(ProgramException("Can't ReSize to SymmetricBandMatrix" ));
   // already know A and B are square
   ReSize(A.Nrows(), my_max(A_BW.Lower(), B_BW.Lower()));
}

void SymmetricBandMatrix::ReSizeForSP(const GeneralMatrix& A,
   const GeneralMatrix& B)
{
   REPORT
   Tracer tr("SymmetricBandMatrix::ReSizeForSP");
   MatrixBandWidth A_BW = A.BandWidth(); MatrixBandWidth B_BW = B.BandWidth();
   if ((A_BW.Lower() < 0) | (B_BW.Lower() < 0))
         Throw(ProgramException("Can't ReSize to SymmetricBandMatrix" ));
   // already know A and B are square
   ReSize(A.Nrows(), my_min(A_BW.Lower(), B_BW.Lower()));
}


void SymmetricBandMatrix::operator=(const BaseMatrix& X)
{
   REPORT // CheckConversion(X);
   // MatrixConversionCheck mcc;
   Eq(X,MatrixType::SB);
}

void SymmetricBandMatrix::CornerClear() const
{
   // set unused parts of BandMatrix to zero
   REPORT
   int i = lower; Real* s = store; int bw = lower + 1;
   if (i) for(;;)
   {
      int j = i;
      Real* sj = s;
      while (j--) *sj++ = 0.0;
      if (!(--i)) break;
      s += bw;
   }
}

MatrixBandWidth SymmetricBandMatrix::BandWidth() const
   { REPORT return MatrixBandWidth(lower,lower); }

inline Real square(Real x) { return x*x; }


Real SymmetricBandMatrix::SumSquare() const
{
   REPORT
   CornerClear();
   Real sum1=0.0; Real sum2=0.0; Real* s=store; int i=nrows; int l=lower;
   while (i--)
      { int j = l; while (j--) sum2 += square(*s++); sum1 += square(*s++); }
   ((GeneralMatrix&)*this).tDelete(); return sum1 + 2.0 * sum2;
}

Real SymmetricBandMatrix::SumAbsoluteValue() const
{
   REPORT
   CornerClear();
   Real sum1=0.0; Real sum2=0.0; Real* s=store; int i=nrows; int l=lower;
   while (i--)
      { int j = l; while (j--) sum2 += fabs(*s++); sum1 += fabs(*s++); }
   ((GeneralMatrix&)*this).tDelete(); return sum1 + 2.0 * sum2;
}

Real SymmetricBandMatrix::Sum() const
{
   REPORT
   CornerClear();
   Real sum1=0.0; Real sum2=0.0; Real* s=store; int i=nrows; int l=lower;
   while (i--)
      { int j = l; while (j--) sum2 += *s++; sum1 += *s++; }
   ((GeneralMatrix&)*this).tDelete(); return sum1 + 2.0 * sum2;
}


#ifdef use_namespace
}
#endif

