import hnswlib
import numpy as np
import timeit

def test_query_new2(vecti, topk=1, genreid=[10,20]):
    starttime = timeit.default_timer()
    filters = [[[(False,10)]], [[(False,20)]], [[(False,30)]], [[(False,40)]], [[(False,50)]]]
    idxall = []

    for i in range(len(vecti)):
        idxallj, _ = p.knn_query_new2(vecti[i], k=topk, conditions=filters)
        idx = []
        for j in idxallj:
            idx += j
        idxall.append(idx)

    elapsedtime = timeit.default_timer() - starttime
    return idxall, elapsedtime


def test_query_new3(vecti, topk=1, genreid=[10,20]):
    starttime = timeit.default_timer()
    filters = [[[(False,10)]], [[(False,20)]], [[(False,30)]], [[(False,40)]], [[(False,50)]]]
    idxall, _ = p.knn_query_new3(vecti, k=topk, conditions=filters)

    elapsedtime = timeit.default_timer() - starttime
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

vecti = np.float32(np.random.random((1000, 512)))

print("Query test ...")
elapsedtimes = []
r = 5
k = 50
for i in range(r):
    result3, elapsed = test_query_new3(vecti, topk=k, genreid=[10,20,30,40,50])
    elapsedtimes.append(elapsed)

meantime = np.mean(elapsedtimes)
print("k = ",k, ", loop ", r, " times")
print("New3 Elapsed time (meantime) = ", meantime)

elapsedtimes = []
for i in range(r):
    result2, elapsed = test_query_new2(vecti, topk=k, genreid=[10,20,30,40,50])
    elapsedtimes.append(elapsed)

meantime = np.mean(elapsedtimes)
print("\nk = ",k, ", loop ", r, " times")
print("New2 Elapsed time (meantime) = ", meantime)

for i in range(len(result2)):
    for j in range(len(result2[i])):
        if result2[i][j] != result3[i][j]:
            print("not same")
