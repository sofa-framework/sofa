#ifndef UTILS_EDIT_H
#define UTILS_EDIT_H

namespace {

template<class T>
class editor {
protected:
	typedef	sofa::core::objectmodel::Data<T> data_type;
	data_type& data;
	T* ptr;												// TODO better
public:

	editor( data_type& data ) 
        : data(data) {
	}


    static editor<T> readWrite( data_type& data )
    {
        editor<T> e( data );
        e.ptr = e.data.beginEdit();
        return e;
    }

    static editor<T> writeOnly( data_type& data )
    {
        editor<T> e( data );
        e.ptr = e.data.beginWriteOnly();
        return e;
    }

	~editor() {
		data.endEdit();
	}
		
	T& operator*() const { return *get(); }
	T* get() const { return ptr; }
	T* operator->() const { return get(); }

	void operator=(const T& other) const { 
		*ptr = other;
	}
	
};

template<class T>
editor<T> edit( sofa::core::objectmodel::Data<T>& data) { return editor<T>::readWrite(data); }

template<class T>
editor<T> editOnly( sofa::core::objectmodel::Data<T>& data) { return editor<T>::writeOnly(data); }

} // namespace

#endif
