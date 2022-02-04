import hnswlib
import numpy as np
import timeit

def test_query_new(vecti, topk=1, genreid=[10,20]):
    starttime = timeit.default_timer()
    filters = [[[(False,10)]], [[(False,20)]], [[(False,30)]], [[(False,40)]], [[(False,50)]]]
    idxall = []

    idxallj, _ = p.knn_query_new(vecti, k=topk, conditions=filters)
    #print(idxallj)
    realid = [mapid[i] for i in idxallj[0]]

    idxall.append(realid)
    elapsedtime = timeit.default_timer() - starttime
    #print("Time elapsed = ", elapsedtime)
    return idxall, elapsedtime


def test_query_new2(vecti, topk=1, genreid=[10,20]):
    starttime = timeit.default_timer()
    filters = [[[(False,10)]], [[(False,20)]], [[(False,30)]], [[(False,40)]], [[(False,50)]]]
    idxall, _ = p.knn_query_new2(vecti, k=topk, conditions=filters)
    elapsedtime = timeit.default_timer() - starttime
    #print("Time elapsed = ", elapsedtime)
    return idxall, elapsedtime

print("Load data ...")
dim = 512
num_elem = 10000

data = np.float32(np.random.random((num_elem, dim)))
p = hnswlib.Index(space='l2', dim=dim)

mapid = {}
for i in range(num_elem):
    mapid[i] = 'myid' + str(i)

p.init_index(max_elements=num_elem, ef_construction=512, M=512, mapid=mapid)
p.set_num_threads(4)

p.add_items(data)

p.add_tags(list(range(1,num_elem, 5)), 10)
p.add_tags(list(range(2,num_elem, 5)), 20)
p.add_tags(list(range(3,num_elem, 5)), 30)
p.add_tags(list(range(4,num_elem, 5)), 40)
p.add_tags(list(range(5,num_elem, 5)), 50)

vecti = np.float32(list(range(1, dim+1, 1)))
for i in range(dim):
    vecti[i] /= dim

print("Query test ...")
elapsedtimes = []
r = 10
k = 50
for i in range(r):
    #print("new loop ", i)
    result, elapsed = test_query_new2(vecti, topk=k, genreid=[10,20,30,40,50])
    elapsedtimes.append(elapsed)

meantime = np.mean(elapsedtimes)
print("k = ",k, ", loop ", r, " times")
print("New2 Elapsed time (meantime) = ", meantime)



elapsedtimes2 = []
for i in range(r):
    #print("old loop ", i)
    result2, elapsed2 = test_query_new(vecti, topk=k, genreid=[10,20,30,40,50])
    elapsedtimes2.append(elapsed)


for i in range(0,len(result2)):
    assert result[i] == result2[i], 'not same'


meantime = np.mean(elapsedtimes2)
print("\nk = ",k, ", loop ", r, " times")
print("New Elapsed time (meantime) = ", meantime)
