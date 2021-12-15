//$$ hholder.cpp                                   QR decomposition

// Copyright (C) 1991,2,3,4: R B Davies

#define WANT_MATH

#include "include.h"

#include "newmatap.h"

#ifdef use_namespace
namespace NEWMAT {
#endif

#ifdef DO_REPORT
#define REPORT { static ExeCounter ExeCount(__LINE__,16); ++ExeCount; }
#else
#define REPORT {}
#endif


/*************************** QR decompositions ***************************/

inline Real square(Real x) { return x*x; }

void QRZT(Matrix& X, LowerTriangularMatrix& L)
{
   REPORT
	Tracer et("QZT(1)");
   int n = X.Ncols(); int s = X.Nrows(); L.ReSize(s);
   Real* xi = X.Store(); int k;
   for (int i=0; i<s; i++)
   {
      Real sum = 0.0;
      Real* xi0=xi; k=n; while(k--) { sum += square(*xi++); }
      sum = sqrt(sum);
      L.element(i,i) = sum;
      if (sum==0.0) Throw(SingularException(L));
      Real* xj0=xi0; k=n; while(k--) { *xj0++ /= sum; }
      for (int j=i+1; j<s; j++)
      {
         sum=0.0;
         xi=xi0; Real* xj=xj0; k=n; while(k--) { sum += *xi++ * *xj++; }
         xi=xi0; k=n; while(k--) { *xj0++ -= sum * *xi++; }
         L.element(j,i) = sum;
      }
   }
}

void QRZT(const Matrix& X, Matrix& Y, Matrix& M)
{
   REPORT
   Tracer et("QRZT(2)");
   int n = X.Ncols(); int s = X.Nrows(); int t = Y.Nrows();
   if (Y.Ncols() != n)
      { Throw(ProgramException("Unequal row lengths",X,Y)); }
   M.ReSize(t,s);
   Real* xi = X.Store(); int k;
   for (int i=0; i<s; i++)
   {
      Real* xj0 = Y.Store(); Real* xi0 = xi;
      for (int j=0; j<t; j++)
      {
         Real sum=0.0;
         xi=xi0; Real* xj=xj0; k=n; while(k--) { sum += *xi++ * *xj++; }
         xi=xi0; k=n; while(k--) { *xj0++ -= sum * *xi++; }
         M.element(j,i) = sum;
      }
   }
}

/*
void QRZ(Matrix& X, UpperTriangularMatrix& U)
{
	Tracer et("QRZ(1)");
	int n = X.Nrows(); int s = X.Ncols(); U.ReSize(s);
	Real* xi0 = X.Store(); int k;
	for (int i=0; i<s; i++)
	{
		Real sum = 0.0;
		Real* xi = xi0; k=n; while(k--) { sum += square(*xi); xi+=s; }
		sum = sqrt(sum);
		U.element(i,i) = sum;
		if (sum==0.0) Throw(SingularException(U));
		Real* xj0=xi0; k=n; while(k--) { *xj0 /= sum; xj0+=s; }
		xj0 = xi0;
		for (int j=i+1; j<s; j++)
		{
			sum=0.0;
			xi=xi0; k=n; xj0++; Real* xj=xj0;
			while(k--) { sum += *xi * *xj; xi+=s; xj+=s; }
			xi=xi0; k=n; xj=xj0;
			while(k--) { *xj -= sum * *xi; xj+=s; xi+=s; }
			U.element(i,j) = sum;
		}
		xi0++;
	}
}
*/

void QRZ(Matrix& X, UpperTriangularMatrix& U)
{
   REPORT
   Tracer et("QRZ(1)");
   int n = X.Nrows(); int s = X.Ncols(); U.ReSize(s); U = 0.0;
   Real* xi0 = X.Store(); Real* u0 = U.Store(); Real* u;
   int j, k; int J = s; int i = s;
   while (i--)
   {
      Real* xj0 = xi0; Real* xi = xi0; k = n;
      if (k) for (;;)
      {
         u = u0; Real Xi = *xi; Real* xj = xj0;
         j = J; while(j--) *u++ += Xi * *xj++;
         if (!(--k)) break;
         xi += s; xj0 += s;
      }

      Real sum = sqrt(*u0); *u0 = sum; u = u0+1;
      if (sum == 0.0) Throw(SingularException(U));
      int J1 = J-1; j = J1; while(j--) *u++ /= sum;

      xj0 = xi0; xi = xi0++; k = n;
      if (k) for (;;)
      {
         u = u0+1; Real Xi = *xi; Real* xj = xj0;
         Xi /= sum; *xj++ = Xi;
         j = J1; while(j--) *xj++ -= *u++ * Xi;
         if (!(--k)) break;
	      xi += s; xj0 += s;
      }
      u0 += J--;
   }
}

void QRZ(const Matrix& X, Matrix& Y, Matrix& M)
{
   REPORT
   Tracer et("QRZ(2)");
   int n = X.Nrows(); int s = X.Ncols(); int t = Y.Ncols();
   if (Y.Nrows() != n)
      { Throw(ProgramException("Unequal column lengths",X,Y)); }
   M.ReSize(s,t); M = 0;Real* m0 = M.Store(); Real* m;
   Real* xi0 = X.Store();
   int j, k; int i = s;
   while (i--)
   {
      Real* xj0 = Y.Store(); Real* xi = xi0; k = n;
      if (k) for (;;)
      {
         m = m0; Real Xi = *xi; Real* xj = xj0;
         j = t; while(j--) *m++ += Xi * *xj++;
         if (!(--k)) break;
         xi += s; xj0 += t;
      }

      xj0 = Y.Store(); xi = xi0++; k = n;
      if (k) for (;;)
      {
         m = m0; Real Xi = *xi; Real* xj = xj0;
         j = t; while(j--) *xj++ -= *m++ * Xi;
         if (!(--k)) break;
         xi += s; xj0 += t;
      }
      m0 += t;
   }
}

/*

void QRZ(const Matrix& X, Matrix& Y, Matrix& M)
{
	Tracer et("QRZ(2)");
	int n = X.Nrows(); int s = X.Ncols(); int t = Y.Ncols();
	if (Y.Nrows() != n)
	{ Throw(ProgramException("Unequal column lengths",X,Y)); }
	M.ReSize(s,t);
	Real* xi0 = X.Store(); int k;
	for (int i=0; i<s; i++)
	{
		Real* xj0 = Y.Store();
		for (int j=0; j<t; j++)
		{
			Real sum=0.0;
			Real* xi=xi0; Real* xj=xj0; k=n;
			while(k--) { sum += *xi * *xj; xi+=s; xj+=t; }
			xi=xi0; k=n; xj=xj0++;
			while(k--) { *xj -= sum * *xi; xj+=t; xi+=s; }
			M.element(i,j) = sum;
		}
		xi0++;
	}
}
*/

#ifdef use_namespace
}
#endif

