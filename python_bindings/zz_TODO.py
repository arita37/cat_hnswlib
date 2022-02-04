#### Task 2 :
   Add internal mapping Index in binding.cpp


   dictionnary :   HNSW Index 0....10000    ---> map to realid   
        mapid = {   0 : 'myid01',  1:'myid02' , .... }

         
####  Current code is :       
        idxallj, distance = p.knn_query_new(vecti, k=topk, conditions=filters)
   
        idxlist = idxallj[0]      
        realid = [  maprealid[i] for i  in   idxallj   ]  #### Map into realid, in Python side.
        idxall.append(realid)


      
####  New code   
        mapid = {   0 : 'myid01',  1:'myid02' , .... }   #### Python dict
        p.init_index(max_elements=num_elem, ef_construction=512, M=512,   realidmap =  mapid)


        realidxj, distance = p.knn_query_new2(vecti, k=topk, conditions=filters)
   
        realid = realidallj[0]      
        ### No need anymore:    realid = [  maprealid[i] for i  in   idxallj   ]  ,   
        ### Mapping is already done in C++
         
        idxall.append(realid)



      
      
      
      
      
      
      
      
      
      
      
      
      
      




#### What the library does:

   store vector of float into internal Index   
          Index :
             1  -->  catid=100, vector ([1,0.9,34243])
             ....
             n  -->  catid=10, vector ([1,0.9,34243])
            
   knn_query :   (only one condition)      
        input vector =  x0([234,32423,543,35]) , x2([234,32423,543,35]) ,  x2([234,32423,543,35]) , 
        conditions  =  [[(False, 10)]]

      for each input vector xj:
         return all the indices, distance such as
                   dist(xj , Index(i) ) are minimal
                   and catid(Index(i) ) = 10
         
         
   knn_query_new :  (only one vector, many conditions)             
        input vector =  x0([234,32423,543,35]) 
        conditions  =  [[(False, 10)]]  , [[(False, 20)]]  , [[(False, 30)]]  , [[(False, 100)]]  , 

      for each condition ck :
         return all the indices, distance such as
                   dist(x0 , Index(i) ) are minimal
                   and catid(Index(i) ) = ck
            

            
  knn_query_new is the NEW method to add in binding.cpp
            
            
            
#################### Goal
Add a new method     knnQuery_return_numpy_new  here
   
   https://github.com/arita37/cat_hnswlib/blob/loop/python_bindings/bindings.cpp


#### current   
knnQuery_return_numpy  method :
      A list of vectors, one condition

### New method      
knnQuery_return_numpy_new :
        One vector, a list of conditions 
  

List of vector :  [  [ 2,3,43,43], [2,4,5,6] ]
    
One condition : [[ (False, 100)]]    

List of  condition :  [  [[ (False, 100)]]   ,  [[ (False, 10)]]   ,  [[ (False, 200)]]   ,   ]     
    


  
#########  
binding.cpp  : add new method:    knn_query_new
    
   
   
   
    
  PYBIND11_PLUGIN(hnswlib) {
        py::module m("hnswlib");
  
          .def("knn_query_new", &Index<float>::knnQuery_return_numpy_new, 
                                         py::arg("data"),    // only 1 vector
                                         py::arg("k")=1, 
                                         py::arg("num_threads")=-1, 
                                         py::arg("conditions")=std::vector< std::vector<std::vector< hnswlib::tagtype >>  >())   // list of condition
                                         // Triple Vector  [ [[  tagtype  ]]  ]
    
.....
    
    
       py::object knnQuery_return_numpy_new(py::object input, size_t k = 1, int num_threads = -1, hnswlib::condition_t &conditions = {}) {

        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();

    
 
         
....

    #### Always (buffer.ndim == 1)     
    rows = 1;
    features = buffer.shape[0];
         
         
     // Loop over the condition with only  1 input vector      (before only 1 condition and loop over the input vectors)
    if(normalize==false) {
        ParallelFor(0, conditions, 
                    num_threads, 
                    [&](size_t ncondition, size_t threadId) {
                          
                           //  hnswlib::SearchCondition search_condition = hnswlib::SearchCondition(conditions);
                      
                           std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> 
                           result = appr_alg->searchKnn((void *) items.data(row), k,  (void *) conditions);                      
                           while(!result.empty()){    
    
    
    

                               
                               
#####                               
 #### Binding usage in Python ##############################################################
  
  ##############################################################################################
  ##### Current Python version ################################################################
    def hnsw_useremb_get_topk(vecti, genreid=[2323,232,3232], topk=100, dimvect=512, filter_cond='must'):
           ####  genrei: list_siid
           global clientdrant
          
           vecti = np.array([ float(x) for x in  vecti.split(",")] ,  dtype='float32')
           idxall = [] 
                               
           ##### Loop is in python                    
           for gi in genreid :
              condition    = [[ (False, int(gi)) ]]    ### (isexcluded:false, catid) 
              idxallj,_  = clientdrant.p.knn_query( vecti, k = topk, conditions = condition   )
              idxall.append( idxallj[0] )

          return idxall 
        
        

                             
        
#################################################################################### ##########################################                 
    #### New Python version              
    def hnsw_useremb_get_topk(vecti, genreid=[2323,232,3232], topk=100, dimvect=512, filter_cond='must'):
           ####  genrei: list_siid
           global clientdrant
           vecti = np.array([ float(x) for x in  vecti.split(",")] ,  dtype='float32')
           idxall = [] 
                               
           condition_list   =  [   [[ (False, int(10)) ]]     ,  [[ (False, int(200)) ]]    , [[ (False, int(300)) ]]    ]                  

           #### Loop is in C++                  
           idxall  = clientdrant.p.knn_query_new( vecti, k = topk, conditions =  condiion_list   )
           return idxall 
                
        
        
        
        
        
        
