#ifndef EDIT_H
#define EDIT_H

namespace {

template<class T>
class editor {
	typedef	sofa::core::objectmodel::Data<T> data_type;
	data_type& data;
	T* ptr;												// TODO better
public:
	editor( data_type& data ) 
		: data(data),
		  ptr(data.beginEdit()) {
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
editor<T> edit( sofa::core::objectmodel::Data<T>& data) { return editor<T>(data); }

}

#endif
