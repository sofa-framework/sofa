#ifndef UTILS_FIND_H
#define UTILS_FIND_H



template<class Container>
struct find_traits {
	typedef typename Container::iterator iterator_type;
	typedef typename Container::value_type::second_type data_type;
};
		
template<class Container>
struct find_traits<const Container> {
	typedef typename Container::const_iterator iterator_type;
	typedef const typename Container::value_type::second_type data_type;
};
		
template<class Map>
static typename find_traits<Map>::data_type& find(Map& map, const typename Map::key_type& key) {
	typename find_traits<Map>::iterator_type it = map.find(key); 
	assert( it != map.end() );
			
	return it->second;
}



#endif
