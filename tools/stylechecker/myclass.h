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

        Data<int> public_datafield ;
        SingleLink<int> public_singlelink ;

        DualLink<int> public_duallink ;
        OtherType<int> public_othertype ;
    private:
        int privateposition ;
        int m_privateposition ;
        Data<int> private_datafield ;
        SingleLink<int> private_singlelink ;
        DualLink<int> private_duallink ;
        OtherType<int> private_othertype ;
};



#endif // MCLASS_H__
