//$$ newmat.h           definition file for new version of matrix package

// Copyright (C) 1991,2,3,4,7,2000,2002: R B Davies
#ifndef NEWMAT_LIB
#define NEWMAT_LIB 0

#include "include.h"

#include "boolean.h"
#include "myexcept.h"


#ifdef use_namespace
namespace NEWMAT { using namespace RBD_COMMON; }
namespace RBD_LIBRARIES { using namespace NEWMAT; }

namespace NEWMAT {
#endif

//#define DO_REPORT                     // to activate REPORT

#ifdef NO_LONG_NAMES
#define UpperTriangularMatrix UTMatrix
#define LowerTriangularMatrix LTMatrix
#define SymmetricMatrix SMatrix
#define DiagonalMatrix DMatrix
#define BandMatrix BMatrix
#define UpperBandMatrix UBMatrix
#define LowerBandMatrix LBMatrix
#define SymmetricBandMatrix SBMatrix
#define BandLUMatrix BLUMatrix
#endif

#ifndef TEMPS_DESTROYED_QUICKLY_R
#define ReturnMatrix ReturnMatrixX
#else
#define ReturnMatrix ReturnMatrixX&
#endif

// ************************** general utilities ****************************/

class GeneralMatrix;

void MatrixErrorNoSpace(void*);                 // no space handler

class LogAndSign
// Return from LogDeterminant function
//    - value of the log plus the sign (+, - or 0)
{
   Real log_value;
   int sign;
public:
   LogAndSign() { log_value=0.0; sign=1; }
   LogAndSign(Real);
   void operator*=(Real);
   void PowEq(int k);  // raise to power of k
   void ChangeSign() { sign = -sign; }
   Real LogValue() const { return log_value; }
   int Sign() const { return sign; }
   Real Value() const;
   FREE_CHECK(LogAndSign)
};

// the following class is for counting the number of times a piece of code
// is executed. It is used for locating any code not executed by test
// routines. Use turbo GREP locate all places this code is called and
// check which ones are not accessed.
// Somewhat implementation dependent as it relies on "cout" still being
// present when ExeCounter objects are destructed.

#ifdef DO_REPORT

class ExeCounter
{
   int line;                                    // code line number
   int fileid;                                  // file identifier
   long nexe;                                   // number of executions
   static int nreports;                         // number of reports
public:
   ExeCounter(int,int);
   void operator++() { nexe++; }
   ~ExeCounter();                               // prints out reports
};

#endif


// ************************** class MatrixType *****************************/

// Is used for finding the type of a matrix resulting from the binary operations
// +, -, * and identifying what conversions are permissible.
// This class must be updated when new matrix types are added.

class GeneralMatrix;                            // defined later
class BaseMatrix;                               // defined later
class MatrixInput;                              // defined later

class MatrixType
{
public:
   enum Attribute {  Valid     = 1,
                     Diagonal  = 2,             // order of these is important
                     Symmetric = 4,
                     Band      = 8,
                     Lower     = 16,
                     Upper     = 32,
                     LUDeco    = 64,
                     Ones      = 128 };

   enum            { US = 0,
                     UT = Valid + Upper,
                     LT = Valid + Lower,
                     Rt = Valid,
                     Sm = Valid + Symmetric,
                     Dg = Valid + Diagonal + Band + Lower + Upper + Symmetric,
                     Id = Valid + Diagonal + Band + Lower + Upper + Symmetric
                        + Ones,
                     RV = Valid,     //   do not separate out
                     CV = Valid,     //   vectors
                     BM = Valid + Band,
                     UB = Valid + Band + Upper,
                     LB = Valid + Band + Lower,
                     SB = Valid + Band + Symmetric,
                     Ct = Valid + LUDeco,
                     BC = Valid + Band + LUDeco
                   };


   static int nTypes() { return 10; }           // number of different types
					       // exclude Ct, US, BC
public:
   int attribute;
   bool DataLossOK;                            // true if data loss is OK when
                                               // this represents a destination
public:
   MatrixType () : DataLossOK(false) {}
   MatrixType (int i) : attribute(i), DataLossOK(false) {}
   MatrixType (int i, bool dlok) : attribute(i), DataLossOK(dlok) {}
   MatrixType (const MatrixType& mt)
      : attribute(mt.attribute), DataLossOK(mt.DataLossOK) {}
   void operator=(const MatrixType& mt)
      { attribute = mt.attribute; DataLossOK = mt.DataLossOK; }
   void SetDataLossOK() { DataLossOK = true; }
   int operator+() const { return attribute; }
   MatrixType operator+(MatrixType mt) const
      { return MatrixType(attribute & mt.attribute); }
   MatrixType operator*(const MatrixType&) const;
   MatrixType SP(const MatrixType&) const;
   MatrixType KP(const MatrixType&) const;
   MatrixType operator|(const MatrixType& mt) const
      { return MatrixType(attribute & mt.attribute & Valid); }
   MatrixType operator&(const MatrixType& mt) const
      { return MatrixType(attribute & mt.attribute & Valid); }
   bool operator>=(MatrixType mt) const
      { return ( attribute & mt.attribute ) == attribute; }
   bool operator<(MatrixType mt) const         // for MS Visual C++ 4
      { return ( attribute & mt.attribute ) != attribute; }
   bool operator==(MatrixType t) const
      { return (attribute == t.attribute); }
   bool operator!=(MatrixType t) const
      { return (attribute != t.attribute); }
   bool operator!() const { return (attribute & Valid) == 0; }
   MatrixType i() const;                       // type of inverse
   MatrixType t() const;                       // type of transpose
   MatrixType AddEqualEl() const               // Add constant to matrix
      { return MatrixType(attribute & (Valid + Symmetric)); }
   MatrixType MultRHS() const;                 // type for rhs of multiply
   MatrixType sub() const                      // type of submatrix
      { return MatrixType(attribute & Valid); }
   MatrixType ssub() const                     // type of sym submatrix
      { return MatrixType(attribute); }        // not for selection matrix
   GeneralMatrix* New() const;                 // new matrix of given type
   GeneralMatrix* New(int,int,BaseMatrix*) const;
                                               // new matrix of given type
   const char* Value() const;                  // to print type
   friend bool Rectangular(MatrixType a, MatrixType b, MatrixType c);
   friend bool Compare(const MatrixType&, MatrixType&);
                                               // compare and check conv.
   bool IsBand() const { return (attribute & Band) != 0; }
   bool IsDiagonal() const { return (attribute & Diagonal) != 0; }
   bool IsSymmetric() const { return (attribute & Symmetric) != 0; }
   bool CannotConvert() const { return (attribute & LUDeco) != 0; }
                                               // used by operator== 
   FREE_CHECK(MatrixType)
};


// *********************** class MatrixBandWidth ***********************/

class MatrixBandWidth
{
public:
   int lower;
   int upper;
   MatrixBandWidth(const int l, const int u) : lower(l), upper (u) {}
   MatrixBandWidth(const int i) : lower(i), upper(i) {}
   MatrixBandWidth operator+(const MatrixBandWidth&) const;
   MatrixBandWidth operator*(const MatrixBandWidth&) const;
   MatrixBandWidth minimum(const MatrixBandWidth&) const;
   MatrixBandWidth t() const { return MatrixBandWidth(upper,lower); }
   bool operator==(const MatrixBandWidth& bw) const
      { return (lower == bw.lower) && (upper == bw.upper); }
   bool operator!=(const MatrixBandWidth& bw) const { return !operator==(bw); }
   int Upper() const { return upper; }
   int Lower() const { return lower; }
   FREE_CHECK(MatrixBandWidth)
};


// ********************* Array length specifier ************************/

// This class is introduced to avoid constructors such as
//   ColumnVector(int)
// being used for conversions

class ArrayLengthSpecifier
{
   int value;
public:
   int Value() const { return value; }
   ArrayLengthSpecifier(int l) : value(l) {}
};

// ************************* Matrix routines ***************************/


class MatrixRowCol;                             // defined later
class MatrixRow;
class MatrixCol;
class MatrixColX;

class GeneralMatrix;                            // defined later
class AddedMatrix;
class MultipliedMatrix;
class SubtractedMatrix;
class SPMatrix;
class KPMatrix;
class ConcatenatedMatrix;
class StackedMatrix;
class SolvedMatrix;
class ShiftedMatrix;
class NegShiftedMatrix;
class ScaledMatrix;
class TransposedMatrix;
class ReversedMatrix;
class NegatedMatrix;
class InvertedMatrix;
class RowedMatrix;
class ColedMatrix;
class DiagedMatrix;
class MatedMatrix;
class GetSubMatrix;
class ReturnMatrixX;
class Matrix;
class nricMatrix;
class RowVector;
class ColumnVector;
class SymmetricMatrix;
class UpperTriangularMatrix;
class LowerTriangularMatrix;
class DiagonalMatrix;
class CroutMatrix;
class BandMatrix;
class LowerBandMatrix;
class UpperBandMatrix;
class SymmetricBandMatrix;
class LinearEquationSolver;
class GenericMatrix;


#define MatrixTypeUnSp 0
//static MatrixType MatrixTypeUnSp(MatrixType::US);
//						// AT&T needs this

class BaseMatrix : public Janitor               // base of all matrix classes
{
protected:
   virtual int search(const BaseMatrix*) const = 0;
						// count number of times matrix
   						// is referred to

public:
   virtual GeneralMatrix* Evaluate(MatrixType mt=MatrixTypeUnSp) = 0;
						// evaluate temporary
   // for old version of G++
   //   virtual GeneralMatrix* Evaluate(MatrixType mt) = 0;
   //   GeneralMatrix* Evaluate() { return Evaluate(MatrixTypeUnSp); }
#ifndef TEMPS_DESTROYED_QUICKLY
   AddedMatrix operator+(const BaseMatrix&) const;    // results of operations
   MultipliedMatrix operator*(const BaseMatrix&) const;
   SubtractedMatrix operator-(const BaseMatrix&) const;
   ConcatenatedMatrix operator|(const BaseMatrix&) const;
   StackedMatrix operator&(const BaseMatrix&) const;
   ShiftedMatrix operator+(Real) const;
   ScaledMatrix operator*(Real) const;
   ScaledMatrix operator/(Real) const;
   ShiftedMatrix operator-(Real) const;
   TransposedMatrix t() const;
//   TransposedMatrix t;
   NegatedMatrix operator-() const;                   // change sign of elements
   ReversedMatrix Reverse() const;
   InvertedMatrix i() const;
//   InvertedMatrix i;
   RowedMatrix AsRow() const;
   ColedMatrix AsColumn() const;
   DiagedMatrix AsDiagonal() const;
   MatedMatrix AsMatrix(int,int) const;
   GetSubMatrix SubMatrix(int,int,int,int) const;
   GetSubMatrix SymSubMatrix(int,int) const;
   GetSubMatrix Row(int) const;
   GetSubMatrix Rows(int,int) const;
   GetSubMatrix Column(int) const;
   GetSubMatrix Columns(int,int) const;
#else
   AddedMatrix& operator+(const BaseMatrix&) const;    // results of operations
   MultipliedMatrix& operator*(const BaseMatrix&) const;
   SubtractedMatrix& operator-(const BaseMatrix&) const;
   ConcatenatedMatrix& operator|(const BaseMatrix&) const;
   StackedMatrix& operator&(const BaseMatrix&) const;
   ShiftedMatrix& operator+(Real) const;
   ScaledMatrix& operator*(Real) const;
   ScaledMatrix& operator/(Real) const;
   ShiftedMatrix& operator-(Real) const;
   TransposedMatrix& t() const;
//   TransposedMatrix& t;
   NegatedMatrix& operator-() const;                   // change sign of elements
   ReversedMatrix& Reverse() const;
   InvertedMatrix& i() const;
//   InvertedMatrix& i;
   RowedMatrix& AsRow() const;
   ColedMatrix& AsColumn() const;
   DiagedMatrix& AsDiagonal() const;
   MatedMatrix& AsMatrix(int,int) const;
   GetSubMatrix& SubMatrix(int,int,int,int) const;
   GetSubMatrix& SymSubMatrix(int,int) const;
   GetSubMatrix& Row(int) const;
   GetSubMatrix& Rows(int,int) const;
   GetSubMatrix& Column(int) const;
   GetSubMatrix& Columns(int,int) const;
#endif
   Real AsScalar() const;                      // conversion of 1 x 1 matrix
   virtual LogAndSign LogDeterminant() const;
   Real Determinant() const;
   virtual Real SumSquare() const;
   Real NormFrobenius() const;
   virtual Real SumAbsoluteValue() const;
   virtual Real Sum() const;
   virtual Real MaximumAbsoluteValue() const;
   virtual Real MaximumAbsoluteValue1(int& i) const;
   virtual Real MaximumAbsoluteValue2(int& i, int& j) const;
   virtual Real MinimumAbsoluteValue() const;
   virtual Real MinimumAbsoluteValue1(int& i) const;
   virtual Real MinimumAbsoluteValue2(int& i, int& j) const;
   virtual Real Maximum() const;
   virtual Real Maximum1(int& i) const;
   virtual Real Maximum2(int& i, int& j) const;
   virtual Real Minimum() const;
   virtual Real Minimum1(int& i) const;
   virtual Real Minimum2(int& i, int& j) const;
   virtual Real Trace() const;
   Real Norm1() const;
   Real NormInfinity() const;
   virtual MatrixBandWidth BandWidth() const;  // bandwidths of band matrix
   virtual void CleanUp() {}                   // to clear store
   void IEQND() const;                         // called by ineq. ops
//   virtual ReturnMatrix Reverse() const;       // reverse order of elements


//protected:
//   BaseMatrix() : t(this), i(this) {}

   friend class GeneralMatrix;
   friend class Matrix;
   friend class nricMatrix;
   friend class RowVector;
   friend class ColumnVector;
   friend class SymmetricMatrix;
   friend class UpperTriangularMatrix;
   friend class LowerTriangularMatrix;
   friend class DiagonalMatrix;
   friend class CroutMatrix;
   friend class BandMatrix;
   friend class LowerBandMatrix;
   friend class UpperBandMatrix;
   friend class SymmetricBandMatrix;
   friend class AddedMatrix;
   friend class MultipliedMatrix;
   friend class SubtractedMatrix;
   friend class SPMatrix;
   friend class KPMatrix;
   friend class ConcatenatedMatrix;
   friend class StackedMatrix;
   friend class SolvedMatrix;
   friend class ShiftedMatrix;
   friend class NegShiftedMatrix;
   friend class ScaledMatrix;
   friend class TransposedMatrix;
   friend class ReversedMatrix;
   friend class NegatedMatrix;
   friend class InvertedMatrix;
   friend class RowedMatrix;
   friend class ColedMatrix;
   friend class DiagedMatrix;
   friend class MatedMatrix;
   friend class GetSubMatrix;
   friend class ReturnMatrixX;
   friend class LinearEquationSolver;
   friend class GenericMatrix;
   NEW_DELETE(BaseMatrix)
};


// ***************************** working classes **************************/

class GeneralMatrix : public BaseMatrix         // declarable matrix types
{
   virtual GeneralMatrix* Image() const;        // copy of matrix
protected:
   int tag;                                     // shows whether can reuse
   int nrows, ncols;                            // dimensions
   int storage;                                 // total store required
   Real* store;                                 // point to store (0=not set)
   GeneralMatrix();                             // initialise with no store
   GeneralMatrix(ArrayLengthSpecifier);         // constructor getting store
   void Add(GeneralMatrix*, Real);              // sum of GM and Real
   void Add(Real);                              // add Real to this
   void NegAdd(GeneralMatrix*, Real);           // Real - GM
   void NegAdd(Real);                           // this = this - Real
   void Multiply(GeneralMatrix*, Real);         // product of GM and Real
   void Multiply(Real);                         // multiply this by Real
   void Negate(GeneralMatrix*);                 // change sign
   void Negate();                               // change sign
   void ReverseElements();                      // internal reverse of elements
   void ReverseElements(GeneralMatrix*);        // reverse order of elements
   void operator=(Real);                        // set matrix to constant
   Real* GetStore();                            // get store or copy
   GeneralMatrix* BorrowStore(GeneralMatrix*, MatrixType);
                                                // temporarily access store
   void GetMatrix(const GeneralMatrix*);        // used by = and initialise
   void Eq(const BaseMatrix&, MatrixType);      // used by =
   void Eq(const BaseMatrix&, MatrixType, bool);// used by <<
   void Eq2(const BaseMatrix&, MatrixType);     // cut down version of Eq
   int search(const BaseMatrix*) const;
   virtual GeneralMatrix* Transpose(TransposedMatrix*, MatrixType);
   void CheckConversion(const BaseMatrix&);     // check conversion OK
   void ReSize(int, int, int);                  // change dimensions
   virtual short SimpleAddOK(const GeneralMatrix*) { return 0; }
             // see bandmat.cpp for explanation
public:
   GeneralMatrix* Evaluate(MatrixType mt=MatrixTypeUnSp);
   virtual MatrixType Type() const = 0;         // type of a matrix
   int Nrows() const { return nrows; }          // get dimensions
   int Ncols() const { return ncols; }
   int Storage() const { return storage; }
   Real* Store() const { return store; }
   virtual ~GeneralMatrix();                    // delete store if set
   void tDelete();                              // delete if tag permits
   bool reuse();                                // true if tag allows reuse
   void Protect() { tag=-1; }                   // cannot delete or reuse
   int Tag() const { return tag; }
   bool IsZero() const;                         // test matrix has all zeros
   void Release() { tag=1; }                    // del store after next use
   void Release(int t) { tag=t; }               // del store after t accesses
   void ReleaseAndDelete() { tag=0; }           // delete matrix after use
   void operator<<(const Real*);                // assignment from an array
   void operator<<(const BaseMatrix& X)
      { Eq(X,this->Type(),true); }              // = without checking type
   void Inject(const GeneralMatrix&);           // copy stored els only
   void operator+=(const BaseMatrix&);
   void operator-=(const BaseMatrix&);
   void operator*=(const BaseMatrix&);
   void operator|=(const BaseMatrix&);
   void operator&=(const BaseMatrix&);
   void operator+=(Real);
   void operator-=(Real r) { operator+=(-r); }
   void operator*=(Real);
   void operator/=(Real r) { operator*=(1.0/r); }
   virtual GeneralMatrix* MakeSolver();         // for solving
   virtual void Solver(MatrixColX&, const MatrixColX&) {}
   virtual void GetRow(MatrixRowCol&) = 0;      // Get matrix row
   virtual void RestoreRow(MatrixRowCol&) {}    // Restore matrix row
   virtual void NextRow(MatrixRowCol&);         // Go to next row
   virtual void GetCol(MatrixRowCol&) = 0;      // Get matrix col
   virtual void GetCol(MatrixColX&) = 0;        // Get matrix col
   virtual void RestoreCol(MatrixRowCol&) {}    // Restore matrix col
   virtual void RestoreCol(MatrixColX&) {}      // Restore matrix col
   virtual void NextCol(MatrixRowCol&);         // Go to next col
   virtual void NextCol(MatrixColX&);           // Go to next col
   Real SumSquare() const;
   Real SumAbsoluteValue() const;
   Real Sum() const;
   Real MaximumAbsoluteValue1(int& i) const;
   Real MinimumAbsoluteValue1(int& i) const;
   Real Maximum1(int& i) const;
   Real Minimum1(int& i) const;
   Real MaximumAbsoluteValue() const;
   Real MaximumAbsoluteValue2(int& i, int& j) const;
   Real MinimumAbsoluteValue() const;
   Real MinimumAbsoluteValue2(int& i, int& j) const;
   Real Maximum() const;
   Real Maximum2(int& i, int& j) const;
   Real Minimum() const;
   Real Minimum2(int& i, int& j) const;
   LogAndSign LogDeterminant() const;
   virtual bool IsEqual(const GeneralMatrix&) const;
                                                // same type, same values
   void CheckStore() const;                     // check store is non-zero
   virtual void SetParameters(const GeneralMatrix*) {}
                                                // set parameters in GetMatrix
   operator ReturnMatrix() const;               // for building a ReturnMatrix
   ReturnMatrix ForReturn() const;
   virtual bool SameStorageType(const GeneralMatrix& A) const;
   virtual void ReSizeForAdd(const GeneralMatrix& A, const GeneralMatrix& B);
   virtual void ReSizeForSP(const GeneralMatrix& A, const GeneralMatrix& B);
   virtual void ReSize(const GeneralMatrix& A);
   MatrixInput operator<<(Real);                // for loading a list
   MatrixInput operator<<(int f);
//   ReturnMatrix Reverse() const;                // reverse order of elements
   void CleanUp();                              // to clear store

   friend class Matrix;
   friend class nricMatrix;
   friend class SymmetricMatrix;
   friend class UpperTriangularMatrix;
   friend class LowerTriangularMatrix;
   friend class DiagonalMatrix;
   friend class CroutMatrix;
   friend class RowVector;
   friend class ColumnVector;
   friend class BandMatrix;
   friend class LowerBandMatrix;
   friend class UpperBandMatrix;
   friend class SymmetricBandMatrix;
   friend class BaseMatrix;
   friend class AddedMatrix;
   friend class MultipliedMatrix;
   friend class SubtractedMatrix;
   friend class SPMatrix;
   friend class KPMatrix;
   friend class ConcatenatedMatrix;
   friend class StackedMatrix;
   friend class SolvedMatrix;
   friend class ShiftedMatrix;
   friend class NegShiftedMatrix;
   friend class ScaledMatrix;
   friend class TransposedMatrix;
   friend class ReversedMatrix;
   friend class NegatedMatrix;
   friend class InvertedMatrix;
   friend class RowedMatrix;
   friend class ColedMatrix;
   friend class DiagedMatrix;
   friend class MatedMatrix;
   friend class GetSubMatrix;
   friend class ReturnMatrixX;
   friend class LinearEquationSolver;
   friend class GenericMatrix;
   NEW_DELETE(GeneralMatrix)
};



class Matrix : public GeneralMatrix             // usual rectangular matrix
{
   GeneralMatrix* Image() const;                // copy of matrix
public:
   Matrix() {}
   ~Matrix() {}
   Matrix(int, int);                            // standard declaration
   Matrix(const BaseMatrix&);                   // evaluate BaseMatrix
   void operator=(const BaseMatrix&);
   void operator=(Real f) { GeneralMatrix::operator=(f); }
   void operator=(const Matrix& m) { operator=((const BaseMatrix&)m); }
   MatrixType Type() const;
   Real& operator()(int, int);                  // access element
   Real& element(int, int);                     // access element
   Real operator()(int, int) const;            // access element
   Real element(int, int) const;               // access element
#ifdef SETUP_C_SUBSCRIPTS
   Real* operator[](int m) { return store+m*ncols; }
   const Real* operator[](int m) const { return store+m*ncols; }
#endif
   Matrix(const Matrix& gm) : GeneralMatrix() { GetMatrix(&gm); }
   GeneralMatrix* MakeSolver();
   Real Trace() const;
   void GetRow(MatrixRowCol&);
   void GetCol(MatrixRowCol&);
   void GetCol(MatrixColX&);
   void RestoreCol(MatrixRowCol&);
   void RestoreCol(MatrixColX&);
   void NextRow(MatrixRowCol&);
   void NextCol(MatrixRowCol&);
   void NextCol(MatrixColX&);
   virtual void ReSize(int,int);           // change dimensions
      // virtual so we will catch it being used in a vector called as a matrix
   void ReSize(const GeneralMatrix& A);
   Real MaximumAbsoluteValue2(int& i, int& j) const;
   Real MinimumAbsoluteValue2(int& i, int& j) const;
   Real Maximum2(int& i, int& j) const;
   Real Minimum2(int& i, int& j) const;
   friend Real DotProduct(const Matrix& A, const Matrix& B);
   NEW_DELETE(Matrix)
};

class nricMatrix : public Matrix                // for use with Numerical
                                                // Recipes in C
{
   GeneralMatrix* Image() const;                // copy of matrix
   Real** row_pointer;                          // points to rows
   void MakeRowPointer();                       // build rowpointer
   void DeleteRowPointer();
public:
   nricMatrix() {}
   nricMatrix(int m, int n)                     // standard declaration
      :  Matrix(m,n) { MakeRowPointer(); }
   nricMatrix(const BaseMatrix& bm)             // evaluate BaseMatrix
      :  Matrix(bm) { MakeRowPointer(); }
   void operator=(const BaseMatrix& bm)
      { DeleteRowPointer(); Matrix::operator=(bm); MakeRowPointer(); }
   void operator=(Real f) { GeneralMatrix::operator=(f); }
   void operator=(const nricMatrix& m) { operator=((const BaseMatrix&)m); }
   void operator<<(const BaseMatrix& X)
      { DeleteRowPointer(); Eq(X,this->Type(),true); MakeRowPointer(); }
   nricMatrix(const nricMatrix& gm) : Matrix() { GetMatrix(&gm); MakeRowPointer(); }
   void ReSize(int m, int n)               // change dimensions
      { DeleteRowPointer(); Matrix::ReSize(m,n); MakeRowPointer(); }
   void ReSize(const GeneralMatrix& A);
   ~nricMatrix() { DeleteRowPointer(); }
   Real** nric() const { CheckStore(); return row_pointer-1; }
   void CleanUp();                                // to clear store
   NEW_DELETE(nricMatrix)
};

class SymmetricMatrix : public GeneralMatrix
{
   GeneralMatrix* Image() const;                // copy of matrix
public:
   SymmetricMatrix() {}
   ~SymmetricMatrix() {}
   SymmetricMatrix(ArrayLengthSpecifier);
   SymmetricMatrix(const BaseMatrix&);
   void operator=(const BaseMatrix&);
   void operator=(Real f) { GeneralMatrix::operator=(f); }
   void operator=(const SymmetricMatrix& m) { operator=((const BaseMatrix&)m); }
   Real& operator()(int, int);                  // access element
   Real& element(int, int);                     // access element
   Real operator()(int, int) const;             // access element
   Real element(int, int) const;                // access element
#ifdef SETUP_C_SUBSCRIPTS
   Real* operator[](int m) { return store+(m*(m+1))/2; }
   const Real* operator[](int m) const { return store+(m*(m+1))/2; }
#endif
   MatrixType Type() const;
   SymmetricMatrix(const SymmetricMatrix& gm) : GeneralMatrix() { GetMatrix(&gm); }
   Real SumSquare() const;
   Real SumAbsoluteValue() const;
   Real Sum() const;
   Real Trace() const;
   void GetRow(MatrixRowCol&);
   void GetCol(MatrixRowCol&);
   void GetCol(MatrixColX&);
   void RestoreCol(MatrixRowCol&) {}
   void RestoreCol(MatrixColX&);
	GeneralMatrix* Transpose(TransposedMatrix*, MatrixType);
   void ReSize(int);                       // change dimensions
   void ReSize(const GeneralMatrix& A);
   NEW_DELETE(SymmetricMatrix)
};

class UpperTriangularMatrix : public GeneralMatrix
{
   GeneralMatrix* Image() const;                // copy of matrix
public:
   UpperTriangularMatrix() {}
   ~UpperTriangularMatrix() {}
   UpperTriangularMatrix(ArrayLengthSpecifier);
   void operator=(const BaseMatrix&);
   void operator=(const UpperTriangularMatrix& m)
      { operator=((const BaseMatrix&)m); }
   UpperTriangularMatrix(const BaseMatrix&);
   UpperTriangularMatrix(const UpperTriangularMatrix& gm) : GeneralMatrix() { GetMatrix(&gm); }
   void operator=(Real f) { GeneralMatrix::operator=(f); }
   Real& operator()(int, int);                  // access element
   Real& element(int, int);                     // access element
   Real operator()(int, int) const;             // access element
   Real element(int, int) const;                // access element
#ifdef SETUP_C_SUBSCRIPTS
   Real* operator[](int m) { return store+m*ncols-(m*(m+1))/2; }
   const Real* operator[](int m) const { return store+m*ncols-(m*(m+1))/2; }
#endif
   MatrixType Type() const;
   GeneralMatrix* MakeSolver() { return this; } // for solving
   void Solver(MatrixColX&, const MatrixColX&);
   LogAndSign LogDeterminant() const;
   Real Trace() const;
   void GetRow(MatrixRowCol&);
   void GetCol(MatrixRowCol&);
   void GetCol(MatrixColX&);
   void RestoreCol(MatrixRowCol&);
   void RestoreCol(MatrixColX& c) { RestoreCol((MatrixRowCol&)c); }
   void NextRow(MatrixRowCol&);
   void ReSize(int);                       // change dimensions
   void ReSize(const GeneralMatrix& A);
   MatrixBandWidth BandWidth() const;
   NEW_DELETE(UpperTriangularMatrix)
};

class LowerTriangularMatrix : public GeneralMatrix
{
   GeneralMatrix* Image() const;                // copy of matrix
public:
   LowerTriangularMatrix() {}
   ~LowerTriangularMatrix() {}
   LowerTriangularMatrix(ArrayLengthSpecifier);
   LowerTriangularMatrix(const LowerTriangularMatrix& gm) : GeneralMatrix() { GetMatrix(&gm); }
   LowerTriangularMatrix(const BaseMatrix& M);
   void operator=(const BaseMatrix&);
   void operator=(Real f) { GeneralMatrix::operator=(f); }
   void operator=(const LowerTriangularMatrix& m)
      { operator=((const BaseMatrix&)m); }
   Real& operator()(int, int);                  // access element
   Real& element(int, int);                     // access element
   Real operator()(int, int) const;             // access element
   Real element(int, int) const;                // access element
#ifdef SETUP_C_SUBSCRIPTS
   Real* operator[](int m) { return store+(m*(m+1))/2; }
   const Real* operator[](int m) const { return store+(m*(m+1))/2; }
#endif
   MatrixType Type() const;
   GeneralMatrix* MakeSolver() { return this; } // for solving
   void Solver(MatrixColX&, const MatrixColX&);
   LogAndSign LogDeterminant() const;
   Real Trace() const;
   void GetRow(MatrixRowCol&);
   void GetCol(MatrixRowCol&);
   void GetCol(MatrixColX&);
   void RestoreCol(MatrixRowCol&);
   void RestoreCol(MatrixColX& c) { RestoreCol((MatrixRowCol&)c); }
   void NextRow(MatrixRowCol&);
   void ReSize(int);                       // change dimensions
   void ReSize(const GeneralMatrix& A);
   MatrixBandWidth BandWidth() const;
   NEW_DELETE(LowerTriangularMatrix)
};

class DiagonalMatrix : public GeneralMatrix
{
   GeneralMatrix* Image() const;                // copy of matrix
public:
   DiagonalMatrix() {}
   ~DiagonalMatrix() {}
   DiagonalMatrix(ArrayLengthSpecifier);
   DiagonalMatrix(const BaseMatrix&);
   DiagonalMatrix(const DiagonalMatrix& gm) : GeneralMatrix() { GetMatrix(&gm); }
   void operator=(const BaseMatrix&);
   void operator=(Real f) { GeneralMatrix::operator=(f); }
   void operator=(const DiagonalMatrix& m) { operator=((const BaseMatrix&)m); }
   Real& operator()(int, int);                  // access element
   Real& operator()(int);                       // access element
   Real operator()(int, int) const;             // access element
   Real operator()(int) const;
   Real& element(int, int);                     // access element
   Real& element(int);                          // access element
   Real element(int, int) const;                // access element
   Real element(int) const;                     // access element
#ifdef SETUP_C_SUBSCRIPTS
   Real& operator[](int m) { return store[m]; }
   const Real& operator[](int m) const { return store[m]; }
#endif
   MatrixType Type() const;

   LogAndSign LogDeterminant() const;
   Real Trace() const;
   void GetRow(MatrixRowCol&);
   void GetCol(MatrixRowCol&);
   void GetCol(MatrixColX&);
   void NextRow(MatrixRowCol&);
   void NextCol(MatrixRowCol&);
   void NextCol(MatrixColX&);
   GeneralMatrix* MakeSolver() { return this; } // for solving
   void Solver(MatrixColX&, const MatrixColX&);
   GeneralMatrix* Transpose(TransposedMatrix*, MatrixType);
   void ReSize(int);                       // change dimensions
   void ReSize(const GeneralMatrix& A);
   Real* nric() const
      { CheckStore(); return store-1; }         // for use by NRIC
   MatrixBandWidth BandWidth() const;
//   ReturnMatrix Reverse() const;                // reverse order of elements
   NEW_DELETE(DiagonalMatrix)
};

class RowVector : public Matrix
{
   GeneralMatrix* Image() const;                // copy of matrix
public:
   RowVector() { nrows = 1; }
   ~RowVector() {}
   RowVector(ArrayLengthSpecifier n) : Matrix(1,n.Value()) {}
   RowVector(const BaseMatrix&);
   RowVector(const RowVector& gm) : Matrix() { GetMatrix(&gm); }
   void operator=(const BaseMatrix&);
   void operator=(Real f) { GeneralMatrix::operator=(f); }
   void operator=(const RowVector& m) { operator=((const BaseMatrix&)m); }
   Real& operator()(int);                       // access element
   Real& element(int);                          // access element
   Real operator()(int) const;                  // access element
   Real element(int) const;                     // access element
#ifdef SETUP_C_SUBSCRIPTS
   Real& operator[](int m) { return store[m]; }
   const Real& operator[](int m) const { return store[m]; }
#endif
   MatrixType Type() const;
   void GetCol(MatrixRowCol&);
   void GetCol(MatrixColX&);
   void NextCol(MatrixRowCol&);
   void NextCol(MatrixColX&);
   void RestoreCol(MatrixRowCol&) {}
   void RestoreCol(MatrixColX& c);
   GeneralMatrix* Transpose(TransposedMatrix*, MatrixType);
   void ReSize(int);                       // change dimensions
   void ReSize(int,int);                   // in case access is matrix
   void ReSize(const GeneralMatrix& A);
   Real* nric() const
      { CheckStore(); return store-1; }         // for use by NRIC
   void CleanUp();                              // to clear store
   // friend ReturnMatrix GetMatrixRow(Matrix& A, int row);
   NEW_DELETE(RowVector)
};

class ColumnVector : public Matrix
{
   GeneralMatrix* Image() const;                // copy of matrix
public:
   ColumnVector() { ncols = 1; }
   ~ColumnVector() {}
   ColumnVector(ArrayLengthSpecifier n) : Matrix(n.Value(),1) {}
   ColumnVector(const BaseMatrix&);
   ColumnVector(const ColumnVector& gm) : Matrix() { GetMatrix(&gm); }
   void operator=(const BaseMatrix&);
   void operator=(Real f) { GeneralMatrix::operator=(f); }
   void operator=(const ColumnVector& m) { operator=((const BaseMatrix&)m); }
   Real& operator()(int);                       // access element
   Real& element(int);                          // access element
   Real operator()(int) const;                  // access element
   Real element(int) const;                     // access element
#ifdef SETUP_C_SUBSCRIPTS
   Real& operator[](int m) { return store[m]; }
   const Real& operator[](int m) const { return store[m]; }
#endif
   MatrixType Type() const;
   GeneralMatrix* Transpose(TransposedMatrix*, MatrixType);
   void ReSize(int);                       // change dimensions
   void ReSize(int,int);                   // in case access is matrix
   void ReSize(const GeneralMatrix& A);
   Real* nric() const
      { CheckStore(); return store-1; }         // for use by NRIC
   void CleanUp();                              // to clear store
//   ReturnMatrix Reverse() const;                // reverse order of elements
   NEW_DELETE(ColumnVector)
};

class CroutMatrix : public GeneralMatrix        // for LU decomposition
{
   int* indx;
   bool d;
   bool sing;
   void ludcmp();
public:
   CroutMatrix(const BaseMatrix&);
   MatrixType Type() const;
   void lubksb(Real*, int=0);
   ~CroutMatrix();
   GeneralMatrix* MakeSolver() { return this; } // for solving
   LogAndSign LogDeterminant() const;
   void Solver(MatrixColX&, const MatrixColX&);
   void GetRow(MatrixRowCol&);
   void GetCol(MatrixRowCol&);
   void GetCol(MatrixColX& c) { GetCol((MatrixRowCol&)c); }
   void operator=(const BaseMatrix&);
   void operator=(const CroutMatrix& m) { operator=((const BaseMatrix&)m); }
   void CleanUp();                                // to clear store
   bool IsEqual(const GeneralMatrix&) const;
   bool IsSingular() const { return sing; }
   NEW_DELETE(CroutMatrix)
};

// ***************************** band matrices ***************************/

class BandMatrix : public GeneralMatrix         // band matrix
{
   GeneralMatrix* Image() const;                // copy of matrix
protected:
   void CornerClear() const;                    // set unused elements to zero
   short SimpleAddOK(const GeneralMatrix* gm);
public:
   int lower, upper;                            // band widths
   BandMatrix() { lower=0; upper=0; CornerClear(); }
   ~BandMatrix() {}
   BandMatrix(int n,int lb,int ub) { ReSize(n,lb,ub); CornerClear(); }
                                                // standard declaration
   BandMatrix(const BaseMatrix&);               // evaluate BaseMatrix
   void operator=(const BaseMatrix&);
   void operator=(Real f) { GeneralMatrix::operator=(f); }
   void operator=(const BandMatrix& m) { operator=((const BaseMatrix&)m); }
   MatrixType Type() const;
   Real& operator()(int, int);                  // access element
   Real& element(int, int);                     // access element
   Real operator()(int, int) const;             // access element
   Real element(int, int) const;                // access element
#ifdef SETUP_C_SUBSCRIPTS
   Real* operator[](int m) { return store+(upper+lower)*m+lower; }
   const Real* operator[](int m) const { return store+(upper+lower)*m+lower; }
#endif
   BandMatrix(const BandMatrix& gm) : GeneralMatrix() { GetMatrix(&gm); }
   LogAndSign LogDeterminant() const;
   GeneralMatrix* MakeSolver();
   Real Trace() const;
   Real SumSquare() const { CornerClear(); return GeneralMatrix::SumSquare(); }
   Real SumAbsoluteValue() const
      { CornerClear(); return GeneralMatrix::SumAbsoluteValue(); }
   Real Sum() const
      { CornerClear(); return GeneralMatrix::Sum(); }
   Real MaximumAbsoluteValue() const
      { CornerClear(); return GeneralMatrix::MaximumAbsoluteValue(); }
   Real MinimumAbsoluteValue() const
      { int i, j; return GeneralMatrix::MinimumAbsoluteValue2(i, j); }
   Real Maximum() const { int i, j; return GeneralMatrix::Maximum2(i, j); }
   Real Minimum() const { int i, j; return GeneralMatrix::Minimum2(i, j); }
   void GetRow(MatrixRowCol&);
   void GetCol(MatrixRowCol&);
   void GetCol(MatrixColX&);
   void RestoreCol(MatrixRowCol&);
   void RestoreCol(MatrixColX& c) { RestoreCol((MatrixRowCol&)c); }
   void NextRow(MatrixRowCol&);
   virtual void ReSize(int, int, int);             // change dimensions
   void ReSize(const GeneralMatrix& A);
   bool SameStorageType(const GeneralMatrix& A) const;
   void ReSizeForAdd(const GeneralMatrix& A, const GeneralMatrix& B);
   void ReSizeForSP(const GeneralMatrix& A, const GeneralMatrix& B);
   MatrixBandWidth BandWidth() const;
   void SetParameters(const GeneralMatrix*);
   MatrixInput operator<<(Real);                // will give error
   MatrixInput operator<<(int f);
   void operator<<(const Real* r);              // will give error
      // the next is included because Zortech and Borland
      // cannot find the copy in GeneralMatrix
   void operator<<(const BaseMatrix& X) { GeneralMatrix::operator<<(X); }
   NEW_DELETE(BandMatrix)
};

class UpperBandMatrix : public BandMatrix       // upper band matrix
{
   GeneralMatrix* Image() const;                // copy of matrix
public:
   UpperBandMatrix() {}
   ~UpperBandMatrix() {}
   UpperBandMatrix(int n, int ubw)              // standard declaration
      : BandMatrix(n, 0, ubw) {}
   UpperBandMatrix(const BaseMatrix&);          // evaluate BaseMatrix
   void operator=(const BaseMatrix&);
   void operator=(Real f) { GeneralMatrix::operator=(f); }
   void operator=(const UpperBandMatrix& m)
      { operator=((const BaseMatrix&)m); }
   MatrixType Type() const;
   UpperBandMatrix(const UpperBandMatrix& gm) : BandMatrix() { GetMatrix(&gm); }
   GeneralMatrix* MakeSolver() { return this; }
   void Solver(MatrixColX&, const MatrixColX&);
   LogAndSign LogDeterminant() const;
   void ReSize(int, int, int);             // change dimensions
   void ReSize(int n,int ubw)              // change dimensions
      { BandMatrix::ReSize(n,0,ubw); }
   void ReSize(const GeneralMatrix& A) { BandMatrix::ReSize(A); }
   Real& operator()(int, int);
   Real operator()(int, int) const;
   Real& element(int, int);
   Real element(int, int) const;
#ifdef SETUP_C_SUBSCRIPTS
   Real* operator[](int m) { return store+upper*m; }
   const Real* operator[](int m) const { return store+upper*m; }
#endif
   NEW_DELETE(UpperBandMatrix)
};

class LowerBandMatrix : public BandMatrix       // upper band matrix
{
   GeneralMatrix* Image() const;                // copy of matrix
public:
   LowerBandMatrix() {}
   ~LowerBandMatrix() {}
   LowerBandMatrix(int n, int lbw)              // standard declaration
      : BandMatrix(n, lbw, 0) {}
   LowerBandMatrix(const BaseMatrix&);          // evaluate BaseMatrix
   void operator=(const BaseMatrix&);
   void operator=(Real f) { GeneralMatrix::operator=(f); }
   void operator=(const LowerBandMatrix& m)
      { operator=((const BaseMatrix&)m); }
   MatrixType Type() const;
   LowerBandMatrix(const LowerBandMatrix& gm) : BandMatrix() { GetMatrix(&gm); }
   GeneralMatrix* MakeSolver() { return this; }
   void Solver(MatrixColX&, const MatrixColX&);
   LogAndSign LogDeterminant() const;
   void ReSize(int, int, int);             // change dimensions
   void ReSize(int n,int lbw)             // change dimensions
      { BandMatrix::ReSize(n,lbw,0); }
   void ReSize(const GeneralMatrix& A) { BandMatrix::ReSize(A); }
   Real& operator()(int, int);
   Real operator()(int, int) const;
   Real& element(int, int);
   Real element(int, int) const;
#ifdef SETUP_C_SUBSCRIPTS
   Real* operator[](int m) { return store+lower*(m+1); }
   const Real* operator[](int m) const { return store+lower*(m+1); }
#endif
   NEW_DELETE(LowerBandMatrix)
};

class SymmetricBandMatrix : public GeneralMatrix
{
   GeneralMatrix* Image() const;                // copy of matrix
   void CornerClear() const;                    // set unused elements to zero
   short SimpleAddOK(const GeneralMatrix* gm);
public:
   int lower;                                   // lower band width
   SymmetricBandMatrix() { lower=0; CornerClear(); }
   ~SymmetricBandMatrix() {}
   SymmetricBandMatrix(int n, int lb) { ReSize(n,lb); CornerClear(); }
   SymmetricBandMatrix(const BaseMatrix&);
   void operator=(const BaseMatrix&);
   void operator=(Real f) { GeneralMatrix::operator=(f); }
   void operator=(const SymmetricBandMatrix& m)
      { operator=((const BaseMatrix&)m); }
   Real& operator()(int, int);                  // access element
   Real& element(int, int);                     // access element
   Real operator()(int, int) const;             // access element
   Real element(int, int) const;                // access element
#ifdef SETUP_C_SUBSCRIPTS
   Real* operator[](int m) { return store+lower*(m+1); }
   const Real* operator[](int m) const { return store+lower*(m+1); }
#endif
   MatrixType Type() const;
   SymmetricBandMatrix(const SymmetricBandMatrix& gm) : GeneralMatrix() { GetMatrix(&gm); }
   GeneralMatrix* MakeSolver();
   Real SumSquare() const;
   Real SumAbsoluteValue() const;
   Real Sum() const;
   Real MaximumAbsoluteValue() const
      { CornerClear(); return GeneralMatrix::MaximumAbsoluteValue(); }
   Real MinimumAbsoluteValue() const
      { int i, j; return GeneralMatrix::MinimumAbsoluteValue2(i, j); }
   Real Maximum() const { int i, j; return GeneralMatrix::Maximum2(i, j); }
   Real Minimum() const { int i, j; return GeneralMatrix::Minimum2(i, j); }
   Real Trace() const;
   LogAndSign LogDeterminant() const;
   void GetRow(MatrixRowCol&);
   void GetCol(MatrixRowCol&);
   void GetCol(MatrixColX&);
   void RestoreCol(MatrixRowCol&) {}
   void RestoreCol(MatrixColX&);
   GeneralMatrix* Transpose(TransposedMatrix*, MatrixType);
   void ReSize(int,int);                       // change dimensions
   void ReSize(const GeneralMatrix& A);
   bool SameStorageType(const GeneralMatrix& A) const;
   void ReSizeForAdd(const GeneralMatrix& A, const GeneralMatrix& B);
   void ReSizeForSP(const GeneralMatrix& A, const GeneralMatrix& B);
   MatrixBandWidth BandWidth() const;
   void SetParameters(const GeneralMatrix*);
   NEW_DELETE(SymmetricBandMatrix)
};

class BandLUMatrix : public GeneralMatrix
// for LU decomposition of band matrix
{
   int* indx;
   bool d;
   bool sing;                                   // true if singular
   Real* store2;
   int storage2;
   void ludcmp();
   int m1,m2;                                   // lower and upper
public:
   BandLUMatrix(const BaseMatrix&);
   MatrixType Type() const;
   void lubksb(Real*, int=0);
   ~BandLUMatrix();
   GeneralMatrix* MakeSolver() { return this; } // for solving
   LogAndSign LogDeterminant() const;
   void Solver(MatrixColX&, const MatrixColX&);
   void GetRow(MatrixRowCol&);
   void GetCol(MatrixRowCol&);
   void GetCol(MatrixColX& c) { GetCol((MatrixRowCol&)c); }
   void operator=(const BaseMatrix&);
   void operator=(const BandLUMatrix& m) { operator=((const BaseMatrix&)m); }
   void CleanUp();                                // to clear store
   bool IsEqual(const GeneralMatrix&) const;
   bool IsSingular() const { return sing; }
   NEW_DELETE(BandLUMatrix)
};

// ************************** special matrices ****************************

class IdentityMatrix : public GeneralMatrix
{
   GeneralMatrix* Image() const;          // copy of matrix
public:
   IdentityMatrix() {}
   ~IdentityMatrix() {}
   IdentityMatrix(ArrayLengthSpecifier n) : GeneralMatrix(1)
      { nrows = ncols = n.Value(); *store = 1; }
   IdentityMatrix(const IdentityMatrix& gm) : GeneralMatrix() { GetMatrix(&gm); }
   IdentityMatrix(const BaseMatrix&);
   void operator=(const BaseMatrix&);
   void operator=(Real f) { GeneralMatrix::operator=(f); }
   MatrixType Type() const;

   LogAndSign LogDeterminant() const;
   Real Trace() const;
   Real SumSquare() const;
   Real SumAbsoluteValue() const;
   Real Sum() const { return Trace(); }
   void GetRow(MatrixRowCol&);
   void GetCol(MatrixRowCol&);
   void GetCol(MatrixColX&);
   void NextRow(MatrixRowCol&);
   void NextCol(MatrixRowCol&);
   void NextCol(MatrixColX&);
   GeneralMatrix* MakeSolver() { return this; } // for solving
   void Solver(MatrixColX&, const MatrixColX&);
   GeneralMatrix* Transpose(TransposedMatrix*, MatrixType);
   void ReSize(int n);
   void ReSize(const GeneralMatrix& A);
   MatrixBandWidth BandWidth() const;
//   ReturnMatrix Reverse() const;                // reverse order of elements
   NEW_DELETE(IdentityMatrix)
};




// ************************** GenericMatrix class ************************/

class GenericMatrix : public BaseMatrix
{
   GeneralMatrix* gm;
   int search(const BaseMatrix* bm) const;
   friend class BaseMatrix;
public:
   GenericMatrix() : gm(0) {}
   GenericMatrix(const BaseMatrix& bm)
      { gm = ((BaseMatrix&)bm).Evaluate(); gm = gm->Image(); }
   GenericMatrix(const GenericMatrix& bm) : BaseMatrix()
      { gm = bm.gm->Image(); }
   void operator=(const GenericMatrix&);
   void operator=(const BaseMatrix&);
   void operator+=(const BaseMatrix&);
   void operator-=(const BaseMatrix&);
   void operator*=(const BaseMatrix&);
   void operator|=(const BaseMatrix&);
   void operator&=(const BaseMatrix&);
   void operator+=(Real);
   void operator-=(Real r) { operator+=(-r); }
   void operator*=(Real);
   void operator/=(Real r) { operator*=(1.0/r); }
   ~GenericMatrix() { delete gm; }
   void CleanUp() { delete gm; gm = 0; }
   void Release() { gm->Release(); }
   GeneralMatrix* Evaluate(MatrixType = MatrixTypeUnSp);
   MatrixBandWidth BandWidth() const;
   NEW_DELETE(GenericMatrix)
};

// *************************** temporary classes *************************/

class MultipliedMatrix : public BaseMatrix
{
protected:
   // if these union statements cause problems, simply remove them
   // and declare the items individually
   union { const BaseMatrix* bm1; GeneralMatrix* gm1; };
						  // pointers to summands
   union { const BaseMatrix* bm2; GeneralMatrix* gm2; };
   MultipliedMatrix(const BaseMatrix* bm1x, const BaseMatrix* bm2x)
      : bm1(bm1x),bm2(bm2x) {}
   int search(const BaseMatrix*) const;
   friend class BaseMatrix;
   friend class GeneralMatrix;
   friend class GenericMatrix;
public:
   ~MultipliedMatrix() {}
   GeneralMatrix* Evaluate(MatrixType mt=MatrixTypeUnSp);
   MatrixBandWidth BandWidth() const;
   NEW_DELETE(MultipliedMatrix)
};

class AddedMatrix : public MultipliedMatrix
{
protected:
   AddedMatrix(const BaseMatrix* bm1x, const BaseMatrix* bm2x)
      : MultipliedMatrix(bm1x,bm2x) {}

   friend class BaseMatrix;
   friend class GeneralMatrix;
   friend class GenericMatrix;
public:
   ~AddedMatrix() {}
   GeneralMatrix* Evaluate(MatrixType mt=MatrixTypeUnSp);
   MatrixBandWidth BandWidth() const;
   NEW_DELETE(AddedMatrix)
};

class SPMatrix : public AddedMatrix
{
protected:
   SPMatrix(const BaseMatrix* bm1x, const BaseMatrix* bm2x)
      : AddedMatrix(bm1x,bm2x) {}

   friend class BaseMatrix;
   friend class GeneralMatrix;
   friend class GenericMatrix;
public:
   ~SPMatrix() {}
   GeneralMatrix* Evaluate(MatrixType mt=MatrixTypeUnSp);
   MatrixBandWidth BandWidth() const;

#ifndef TEMPS_DESTROYED_QUICKLY
   friend SPMatrix SP(const BaseMatrix&, const BaseMatrix&);
#else
   friend SPMatrix& SP(const BaseMatrix&, const BaseMatrix&);
#endif

   NEW_DELETE(SPMatrix)
};

class KPMatrix : public MultipliedMatrix
{
protected:
   KPMatrix(const BaseMatrix* bm1x, const BaseMatrix* bm2x)
      : MultipliedMatrix(bm1x,bm2x) {}

   friend class BaseMatrix;
   friend class GeneralMatrix;
   friend class GenericMatrix;
public:
   ~KPMatrix() {}
   MatrixBandWidth BandWidth() const;
   GeneralMatrix* Evaluate(MatrixType mt=MatrixTypeUnSp);
#ifndef TEMPS_DESTROYED_QUICKLY
   friend KPMatrix KP(const BaseMatrix&, const BaseMatrix&);
#else
   friend KPMatrix& KP(const BaseMatrix&, const BaseMatrix&);
#endif
   NEW_DELETE(KPMatrix)
};

class ConcatenatedMatrix : public MultipliedMatrix
{
protected:
   ConcatenatedMatrix(const BaseMatrix* bm1x, const BaseMatrix* bm2x)
      : MultipliedMatrix(bm1x,bm2x) {}

   friend class BaseMatrix;
   friend class GeneralMatrix;
   friend class GenericMatrix;
public:
   ~ConcatenatedMatrix() {}
   MatrixBandWidth BandWidth() const;
   GeneralMatrix* Evaluate(MatrixType mt=MatrixTypeUnSp);
   NEW_DELETE(ConcatenatedMatrix)
};

class StackedMatrix : public ConcatenatedMatrix
{
protected:
   StackedMatrix(const BaseMatrix* bm1x, const BaseMatrix* bm2x)
      : ConcatenatedMatrix(bm1x,bm2x) {}

   friend class BaseMatrix;
   friend class GeneralMatrix;
   friend class GenericMatrix;
public:
   ~StackedMatrix() {}
   GeneralMatrix* Evaluate(MatrixType mt=MatrixTypeUnSp);
   NEW_DELETE(StackedMatrix)
};

class SolvedMatrix : public MultipliedMatrix
{
   SolvedMatrix(const BaseMatrix* bm1x, const BaseMatrix* bm2x)
      : MultipliedMatrix(bm1x,bm2x) {}
   friend class BaseMatrix;
   friend class InvertedMatrix;                        // for operator*
public:
   ~SolvedMatrix() {}
   GeneralMatrix* Evaluate(MatrixType mt=MatrixTypeUnSp);
   MatrixBandWidth BandWidth() const;
   NEW_DELETE(SolvedMatrix)
};

class SubtractedMatrix : public AddedMatrix
{
   SubtractedMatrix(const BaseMatrix* bm1x, const BaseMatrix* bm2x)
      : AddedMatrix(bm1x,bm2x) {}
   friend class BaseMatrix;
   friend class GeneralMatrix;
   friend class GenericMatrix;
public:
   ~SubtractedMatrix() {}
   GeneralMatrix* Evaluate(MatrixType mt=MatrixTypeUnSp);
   NEW_DELETE(SubtractedMatrix)
};

class ShiftedMatrix : public BaseMatrix
{
protected:
   union { const BaseMatrix* bm; GeneralMatrix* gm; };
   Real f;
   ShiftedMatrix(const BaseMatrix* bmx, Real fx) : bm(bmx),f(fx) {}
   int search(const BaseMatrix*) const;
   friend class BaseMatrix;
   friend class GeneralMatrix;
   friend class GenericMatrix;
public:
   ~ShiftedMatrix() {}
   GeneralMatrix* Evaluate(MatrixType mt=MatrixTypeUnSp);
#ifndef TEMPS_DESTROYED_QUICKLY
   friend ShiftedMatrix operator+(Real f, const BaseMatrix& BM);
//      { return ShiftedMatrix(&BM, f); }
#endif
   NEW_DELETE(ShiftedMatrix)
};

class NegShiftedMatrix : public ShiftedMatrix
{
protected:
   NegShiftedMatrix(Real fx, const BaseMatrix* bmx) : ShiftedMatrix(bmx,fx) {}
   friend class BaseMatrix;
   friend class GeneralMatrix;
   friend class GenericMatrix;
public:
   ~NegShiftedMatrix() {}
   GeneralMatrix* Evaluate(MatrixType mt=MatrixTypeUnSp);
#ifndef TEMPS_DESTROYED_QUICKLY
   friend NegShiftedMatrix operator-(Real, const BaseMatrix&);
#else
   friend NegShiftedMatrix& operator-(Real, const BaseMatrix&);
#endif
   NEW_DELETE(NegShiftedMatrix)
};

class ScaledMatrix : public ShiftedMatrix
{
   ScaledMatrix(const BaseMatrix* bmx, Real fx) : ShiftedMatrix(bmx,fx) {}
   friend class BaseMatrix;
   friend class GeneralMatrix;
   friend class GenericMatrix;
public:
   ~ScaledMatrix() {}
   GeneralMatrix* Evaluate(MatrixType mt=MatrixTypeUnSp);
   MatrixBandWidth BandWidth() const;
#ifndef TEMPS_DESTROYED_QUICKLY
   friend ScaledMatrix operator*(Real f, const BaseMatrix& BM);
      //{ return ScaledMatrix(&BM, f); }
#endif
   NEW_DELETE(ScaledMatrix)
};

class NegatedMatrix : public BaseMatrix
{
protected:
   union { const BaseMatrix* bm; GeneralMatrix* gm; };
   NegatedMatrix(const BaseMatrix* bmx) : bm(bmx) {}
   int search(const BaseMatrix*) const;
private:
   friend class BaseMatrix;
public:
   ~NegatedMatrix() {}
   GeneralMatrix* Evaluate(MatrixType mt=MatrixTypeUnSp);
   MatrixBandWidth BandWidth() const;
   NEW_DELETE(NegatedMatrix)
};

class TransposedMatrix : public NegatedMatrix
{
   TransposedMatrix(const BaseMatrix* bmx) : NegatedMatrix(bmx) {}
   friend class BaseMatrix;
public:
   ~TransposedMatrix() {}
   GeneralMatrix* Evaluate(MatrixType mt=MatrixTypeUnSp);
   MatrixBandWidth BandWidth() const;
   NEW_DELETE(TransposedMatrix)
};

class ReversedMatrix : public NegatedMatrix
{
   ReversedMatrix(const BaseMatrix* bmx) : NegatedMatrix(bmx) {}
   friend class BaseMatrix;
public:
   ~ReversedMatrix() {}
   GeneralMatrix* Evaluate(MatrixType mt=MatrixTypeUnSp);
   NEW_DELETE(ReversedMatrix)
};

class InvertedMatrix : public NegatedMatrix
{
   InvertedMatrix(const BaseMatrix* bmx) : NegatedMatrix(bmx) {}
public:
   ~InvertedMatrix() {}
#ifndef TEMPS_DESTROYED_QUICKLY
   SolvedMatrix operator*(const BaseMatrix&) const;       // inverse(A) * B
   ScaledMatrix operator*(Real t) const { return BaseMatrix::operator*(t); }
#else
   SolvedMatrix& operator*(const BaseMatrix&);            // inverse(A) * B
   ScaledMatrix& operator*(Real t) const { return BaseMatrix::operator*(t); }
#endif
   friend class BaseMatrix;
   GeneralMatrix* Evaluate(MatrixType mt=MatrixTypeUnSp);
   MatrixBandWidth BandWidth() const;
   NEW_DELETE(InvertedMatrix)
};

class RowedMatrix : public NegatedMatrix
{
   RowedMatrix(const BaseMatrix* bmx) : NegatedMatrix(bmx) {}
   friend class BaseMatrix;
public:
   ~RowedMatrix() {}
   GeneralMatrix* Evaluate(MatrixType mt=MatrixTypeUnSp);
   MatrixBandWidth BandWidth() const;
   NEW_DELETE(RowedMatrix)
};

class ColedMatrix : public NegatedMatrix
{
   ColedMatrix(const BaseMatrix* bmx) : NegatedMatrix(bmx) {}
   friend class BaseMatrix;
public:
   ~ColedMatrix() {}
   GeneralMatrix* Evaluate(MatrixType mt=MatrixTypeUnSp);
   MatrixBandWidth BandWidth() const;
   NEW_DELETE(ColedMatrix)
};

class DiagedMatrix : public NegatedMatrix
{
   DiagedMatrix(const BaseMatrix* bmx) : NegatedMatrix(bmx) {}
   friend class BaseMatrix;
public:
   ~DiagedMatrix() {}
   GeneralMatrix* Evaluate(MatrixType mt=MatrixTypeUnSp);
   MatrixBandWidth BandWidth() const;
   NEW_DELETE(DiagedMatrix)
};

class MatedMatrix : public NegatedMatrix
{
   int nr, nc;
   MatedMatrix(const BaseMatrix* bmx, int nrx, int ncx)
      : NegatedMatrix(bmx), nr(nrx), nc(ncx) {}
   friend class BaseMatrix;
public:
   ~MatedMatrix() {}
   GeneralMatrix* Evaluate(MatrixType mt=MatrixTypeUnSp);
   MatrixBandWidth BandWidth() const;
   NEW_DELETE(MatedMatrix)
};

class ReturnMatrixX : public BaseMatrix    // for matrix return
{
   GeneralMatrix* gm;
   int search(const BaseMatrix*) const;
public:
   ~ReturnMatrixX() {}
   GeneralMatrix* Evaluate(MatrixType mt=MatrixTypeUnSp);
   friend class BaseMatrix;
#ifdef TEMPS_DESTROYED_QUICKLY_R
   ReturnMatrixX(const ReturnMatrixX& tm);
#else
   ReturnMatrixX(const ReturnMatrixX& tm) : BaseMatrix(), gm(tm.gm) {}
#endif
   ReturnMatrixX(const GeneralMatrix* gmx) : BaseMatrix(), gm((GeneralMatrix*&)gmx) {}
//   ReturnMatrixX(GeneralMatrix&);
   MatrixBandWidth BandWidth() const;
   NEW_DELETE(ReturnMatrixX)
};


// ************************** submatrices ******************************/

class GetSubMatrix : public NegatedMatrix
{
   int row_skip;
   int row_number;
   int col_skip;
   int col_number;
   bool IsSym;

   GetSubMatrix
      (const BaseMatrix* bmx, int rs, int rn, int cs, int cn, bool is)
      : NegatedMatrix(bmx),
      row_skip(rs), row_number(rn), col_skip(cs), col_number(cn), IsSym(is) {}
   void SetUpLHS();
   friend class BaseMatrix;
public:
   GetSubMatrix(const GetSubMatrix& g)
      : NegatedMatrix(g.bm), row_skip(g.row_skip), row_number(g.row_number),
      col_skip(g.col_skip), col_number(g.col_number), IsSym(g.IsSym) {}
   ~GetSubMatrix() {}
   GeneralMatrix* Evaluate(MatrixType mt=MatrixTypeUnSp);
   void operator=(const BaseMatrix&);
   void operator+=(const BaseMatrix&);
   void operator-=(const BaseMatrix&);
   void operator=(const GetSubMatrix& m) { operator=((const BaseMatrix&)m); }
   void operator<<(const BaseMatrix&);
   void operator<<(const Real*);                // copy from array
   MatrixInput operator<<(Real);                // for loading a list
   MatrixInput operator<<(int f);
   void operator=(Real);                        // copy from constant
   void operator+=(Real);                       // add constant
   void operator-=(Real r) { operator+=(-r); }  // subtract constant
   void operator*=(Real);                       // multiply by constant
   void operator/=(Real r) { operator*=(1.0/r); } // divide by constant
   void Inject(const GeneralMatrix&);           // copy stored els only
   MatrixBandWidth BandWidth() const;
   NEW_DELETE(GetSubMatrix)
};

// ******************** linear equation solving ****************************/

class LinearEquationSolver : public BaseMatrix
{
   GeneralMatrix* gm;
   int search(const BaseMatrix*) const { return 0; }
   friend class BaseMatrix;
public:
   LinearEquationSolver(const BaseMatrix& bm);
   ~LinearEquationSolver() { delete gm; }
   void CleanUp() { delete gm; } 
   GeneralMatrix* Evaluate(MatrixType) { return gm; }
   // probably should have an error message if MatrixType != UnSp
   NEW_DELETE(LinearEquationSolver)
};

// ************************** matrix input *******************************/

class MatrixInput          // for reading a list of values into a matrix
                           // the difficult part is detecting a mismatch
                           // in the number of elements
{
   int n;                  // number values still to be read
   Real* r;                // pointer to next location to be read to
public:
   MatrixInput(const MatrixInput& mi) : n(mi.n), r(mi.r) {}
   MatrixInput(int nx, Real* rx) : n(nx), r(rx) {}
   ~MatrixInput();
   MatrixInput operator<<(Real);
   MatrixInput operator<<(int f);
   friend class GeneralMatrix;
};



// **************** a very simple integer array class ********************/

// A minimal array class to imitate a C style array but giving dynamic storage
// mostly intended for internal use by newmat

class SimpleIntArray : public Janitor
{
protected:
   int* a;                    // pointer to the array
   int n;                     // length of the array
public:
   SimpleIntArray(int xn);    // build an array length xn
   ~SimpleIntArray();         // return the space to memory
   int& operator[](int i);    // access element of the array - start at 0
   int operator[](int i) const;
			      // access element of constant array
   void operator=(int ai);    // set the array equal to a constant
   void operator=(const SimpleIntArray& b);
			      // copy the elements of an array
   SimpleIntArray(const SimpleIntArray& b);
			      // make a new array equal to an existing one
   int Size() const { return n; }
			      // return the size of the array
   int* Data() { return a; }  // pointer to the data
   const int* Data() const { return a; }
			      // pointer to the data
   void ReSize(int i, bool keep = false);
                              // change length, keep data if keep = true
   void CleanUp() { ReSize(0); }
   NEW_DELETE(SimpleIntArray)
};

// *************************** exceptions ********************************/

class NPDException : public Runtime_error     // Not positive definite
{
public:
   static unsigned long Select;          // for identifying exception
   NPDException(const GeneralMatrix&);
};

class ConvergenceException : public Runtime_error
{
public:
   static unsigned long Select;          // for identifying exception
   ConvergenceException(const GeneralMatrix& A);
   ConvergenceException(const char* c);
};

class SingularException : public Runtime_error
{
public:
   static unsigned long Select;          // for identifying exception
   SingularException(const GeneralMatrix& A);
};

class OverflowException : public Runtime_error
{
public:
   static unsigned long Select;          // for identifying exception
   OverflowException(const char* c);
};

class ProgramException : public Logic_error
{
protected:
   ProgramException();
public:
   static unsigned long Select;          // for identifying exception
   ProgramException(const char* c);
   ProgramException(const char* c, const GeneralMatrix&);
   ProgramException(const char* c, const GeneralMatrix&, const GeneralMatrix&);
   ProgramException(const char* c, MatrixType, MatrixType);
};

class IndexException : public Logic_error
{
public:
   static unsigned long Select;          // for identifying exception
   IndexException(int i, const GeneralMatrix& A);
   IndexException(int i, int j, const GeneralMatrix& A);
   // next two are for access via element function
   IndexException(int i, const GeneralMatrix& A, bool);
   IndexException(int i, int j, const GeneralMatrix& A, bool);
};

class VectorException : public Logic_error    // cannot convert to vector
{
public:
   static unsigned long Select;          // for identifying exception
   VectorException();
   VectorException(const GeneralMatrix& A);
};

class NotSquareException : public Logic_error
{
public:
   static unsigned long Select;          // for identifying exception
   NotSquareException(const GeneralMatrix& A);
};

class SubMatrixDimensionException : public Logic_error
{
public:
   static unsigned long Select;          // for identifying exception
   SubMatrixDimensionException();
};

class IncompatibleDimensionsException : public Logic_error
{
public:
   static unsigned long Select;          // for identifying exception
   IncompatibleDimensionsException();
   IncompatibleDimensionsException(const GeneralMatrix&, const GeneralMatrix&);
};

class NotDefinedException : public Logic_error
{
public:
   static unsigned long Select;          // for identifying exception
   NotDefinedException(const char* op, const char* matrix);
};

class CannotBuildException : public Logic_error
{
public:
   static unsigned long Select;          // for identifying exception
   CannotBuildException(const char* matrix);
};


class InternalException : public Logic_error
{
public:
   static unsigned long Select;          // for identifying exception
   InternalException(const char* c);
};

// ************************ functions ************************************ //

bool operator==(const GeneralMatrix& A, const GeneralMatrix& B);
bool operator==(const BaseMatrix& A, const BaseMatrix& B);
inline bool operator!=(const GeneralMatrix& A, const GeneralMatrix& B)
   { return ! (A==B); }
inline bool operator!=(const BaseMatrix& A, const BaseMatrix& B)
   { return ! (A==B); }

   // inequality operators are dummies included for compatibility
   // with STL. They throw an exception if actually called.
inline bool operator<=(const BaseMatrix& A, const BaseMatrix&)
   { A.IEQND(); return true; }
inline bool operator>=(const BaseMatrix& A, const BaseMatrix&)
   { A.IEQND(); return true; }
inline bool operator<(const BaseMatrix& A, const BaseMatrix&)
   { A.IEQND(); return true; }
inline bool operator>(const BaseMatrix& A, const BaseMatrix&)
   { A.IEQND(); return true; }

bool IsZero(const BaseMatrix& A);


// *********************** friend functions ****************************** //

bool Rectangular(MatrixType a, MatrixType b, MatrixType c);
bool Compare(const MatrixType&, MatrixType&);
Real DotProduct(const Matrix& A, const Matrix& B);
#ifndef TEMPS_DESTROYED_QUICKLY
   SPMatrix SP(const BaseMatrix&, const BaseMatrix&);
   KPMatrix KP(const BaseMatrix&, const BaseMatrix&);
   ShiftedMatrix operator+(Real f, const BaseMatrix& BM);
   NegShiftedMatrix operator-(Real, const BaseMatrix&);
   ScaledMatrix operator*(Real f, const BaseMatrix& BM);
#else
   SPMatrix& SP(const BaseMatrix&, const BaseMatrix&);
   KPMatrix& KP(const BaseMatrix&, const BaseMatrix&);
   NegShiftedMatrix& operator-(Real, const BaseMatrix&);
#endif


// ********************* inline functions ******************************** //

inline LogAndSign LogDeterminant(const BaseMatrix& B)
   { return B.LogDeterminant(); }
inline Real Determinant(const BaseMatrix& B)
   { return B.Determinant(); }
inline Real SumSquare(const BaseMatrix& B) { return B.SumSquare(); }
inline Real NormFrobenius(const BaseMatrix& B) { return B.NormFrobenius(); }
inline Real Trace(const BaseMatrix& B) { return B.Trace(); }
inline Real SumAbsoluteValue(const BaseMatrix& B)
   { return B.SumAbsoluteValue(); }
inline Real Sum(const BaseMatrix& B)
   { return B.Sum(); }
inline Real MaximumAbsoluteValue(const BaseMatrix& B)
   { return B.MaximumAbsoluteValue(); }
inline Real MinimumAbsoluteValue(const BaseMatrix& B)
   { return B.MinimumAbsoluteValue(); }
inline Real Maximum(const BaseMatrix& B) { return B.Maximum(); }
inline Real Minimum(const BaseMatrix& B) { return B.Minimum(); }
inline Real Norm1(const BaseMatrix& B) { return B.Norm1(); }
inline Real Norm1(RowVector& RV) { return RV.MaximumAbsoluteValue(); }
inline Real NormInfinity(const BaseMatrix& B) { return B.NormInfinity(); }
inline Real NormInfinity(ColumnVector& CV)
   { return CV.MaximumAbsoluteValue(); }
inline bool IsZero(const GeneralMatrix& A) { return A.IsZero(); }

#ifdef TEMPS_DESTROYED_QUICKLY
inline ShiftedMatrix& operator+(Real f, const BaseMatrix& BM)
   { return BM + f; }
inline ScaledMatrix& operator*(Real f, const BaseMatrix& BM)
   { return BM * f; }
#endif

// these are moved out of the class definitions because of a problem
// with the Intel 8.1 compiler
#ifndef TEMPS_DESTROYED_QUICKLY
   inline ShiftedMatrix operator+(Real f, const BaseMatrix& BM)
      { return ShiftedMatrix(&BM, f); }
   inline ScaledMatrix operator*(Real f, const BaseMatrix& BM)
      { return ScaledMatrix(&BM, f); }
#endif


inline MatrixInput MatrixInput::operator<<(int f) { return *this << (Real)f; }
inline MatrixInput GeneralMatrix::operator<<(int f) { return *this << (Real)f; }
inline MatrixInput BandMatrix::operator<<(int f) { return *this << (Real)f; }
inline MatrixInput GetSubMatrix::operator<<(int f) { return *this << (Real)f; }



#ifdef use_namespace
}
#endif


#endif

// body file: newmat1.cpp
// body file: newmat2.cpp
// body file: newmat3.cpp
// body file: newmat4.cpp
// body file: newmat5.cpp
// body file: newmat6.cpp
// body file: newmat7.cpp
// body file: newmat8.cpp
// body file: newmatex.cpp
// body file: bandmat.cpp
// body file: submat.cpp







