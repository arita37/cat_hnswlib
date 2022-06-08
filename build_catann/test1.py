import catannlib
import numpy as np
import timeit

def log(*s):
    print(*s, flush=True)


def test_query(vecti, topk=1, genreid=[10,20]):
    starttime = timeit.default_timer()
    idxall = []
    log(p.knn_query)
    for gi in genreid:
        filters = [[(False, int(gi))]]
        idxallj, _ = p.knn_query(vecti, k=topk, conditions=filters)
        idxall.append(idxallj[0])
    elapsedtime = timeit.default_timer() - starttime
    #print("Time elapsed = ", elapsedtime)
    return idxall, elapsedtime


def test_query_new(vecti, topk=1, genreid=[10,20]):
    starttime = timeit.default_timer()
    filters = []
    log(p.knn_query_new)
    for gi in genreid:
        filters.append([[(False, int(gi))]])
    idxall, _ = p.knn_query_new(vecti, k=topk, conditions=filters)
    elapsedtime = timeit.default_timer() - starttime
    #print("Time elapsed = ", elapsedtime)
    return idxall, elapsedtime

print("Load data ...")
dim = 512
num_elem = 10000

data = np.float32(np.random.random((num_elem, dim)))
p = catannlib.Index(space='l2', dim=dim)
p.init_index(max_elements=num_elem, ef_construction=512, M=512)
p.set_num_threads(4)

p.add_items(data)

p.add_tags(list(range(1,dim, 5)), 10)
p.add_tags(list(range(2,dim, 5)), 20)
p.add_tags(list(range(3,dim, 5)), 30)
p.add_tags(list(range(4,dim, 5)), 40)
p.add_tags(list(range(5,dim, 5)), 50)

vecti = np.float32(list(range(1,17,1)))
for i in range(16):
    vecti[i] /= 100

print("Query test ...")
elapsedtimes = []
r = 10
k = 50
for i in range(r):
    #print("new loop ", i)
    result, elapsed = test_query_new(vecti, topk=k, genreid=[10,20,30,40,50])
    elapsedtimes.append(elapsed)

meantime = np.mean(elapsedtimes)
print("k = ",k, ", loop ", r, " times")
print("New Elapsed time (meantime) = ", meantime)

elapsedtimes = []
for i in range(r):
    #print("old loop ", i)
    result, elapsed = test_query(vecti, topk=k, genreid=[10,20,30,40,50])
    elapsedtimes.append(elapsed)

meantime = np.mean(elapsedtimes)
print("\nk = ",k, ", loop ", r, " times")
print("Old Elapsed time (meantime) = ", meantime)
