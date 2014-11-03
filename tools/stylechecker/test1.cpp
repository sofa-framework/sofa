#include <iostream>
#include <cstdio>
#include <vector>
#include "test1.h"


using namespace sofa ;

int my_printf(){
	printf("La bonne blague...") ;
	return 0 ; 
}

int MyClass::functionVALID(){
	// This one is not defined in a header..
	return 1 ;
}


int main()
{ 
    // Invalid declaration, we should always initialize
    MyClass* cl2INVALID ;
    cl2INVALID=new MyClass() ;

    // Invalid pod declaration, we should always initialize
    int aINVALID;
    // This one is ok.
    int bVALID=0, cINVALID, dVALID=2;

    for(int iINVALID; iINVALID < 10; iINVALID++)
    {
	    for(int jINVALID; jINVALID < 10; jINVALID++)
	    {
		goto finboucle;
	    }	
    }	
    finboucle: 

    for(int iVALID=10; iVALID< 10; iVALID++)
    {

    }

    printf("Hello world pas bien\n");
    cout << "Hello world bien" << endl ;



    //
    MyClass* clVALID=new MyClass() ;
    clVALID->functionInvalid() ;
}
