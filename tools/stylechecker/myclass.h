#ifndef MYCLASS_H__
#define MYCLASS_H__

template<class T>
class Data
{
private:
    T* t;

} ;

template<class T>
class SingleLink
{
private:
    T* m_t;

} ;

template<class T>
class DualLink
{
private:
    T* m_t;

} ;

template<class T>
class OtherType
{
private:
    T* m_t;

} ;

using namespace std;
class MyClass
{
	public:
		MyClass(){}
		int use(){return 0; }
        int publicposition ;

        Data<int> public_datafieldVALID ;
        SingleLink<int> public_singlelinkVALID ;

        DualLink<int> public_duallinkVALID ;
        OtherType<int> public_othertypeVALID ;

	void* public_voidpointerVALID ;
	int* public_intpointerVALID ;

	int othercodingstyleINVALID_;
    private:
        int privatepositionINVALID ;
        int m_privatepositionVALID ;
        Data<int> private_datafieldINVALID ;
        SingleLink<int> private_singlelinkINVALID ;
        DualLink<int> private_duallinkINVALID ;
        OtherType<int> private_othertypeINVALID ;

	void* private_voidpointerINVALID;
	int* private_intpointerINVALID;

	bool invalidBooleanINVALID;
	bool bValidBooleanVALID;
};



#endif // MCLASS_H__
