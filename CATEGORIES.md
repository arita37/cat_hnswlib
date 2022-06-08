# cat_catannlib - Fork of the [catannlib](https://github.com/nmslib/catannlib) with support of categorical filtering.

[Motivation](https://comprehension.ml/posts/categorical-catann/)

New categorical methods:

* `add_tags(labels, tag)` - assign `tag` to the specified `labels`
* `get_tags(label)` - returns list of tags, assigned to the `label`
* `reset_tags()` - drop all tag-related information including additionaly built links
* `index_tagged(tag, m)` - build additional navigation graph among tagged points with `tag`. Ensure connectiviti of conditional search
* `index_cross_tagged(tags, m)` - build additional navigation graph among tagged points with `tags`. Does not create new entrypoints. Useful for creating geo-index and numerical ranges.
* `knn_query(data, k = 1, num_threads = -1,  conditions = [])` - extended with parameret `conditions`. It defines what points to include in search results. Performs traversal starting from the first point which fulfills condition. Example `(A | !B) & C` is represented as  `[[(0, A), (1, B)], [(0, C)]]`, where A, B, C loginal clauses if respective tag is assigned to a point. `[[(0, 55)]]` - means find closest point with tag 55.


## Install

```
pip install --no-binary :all: 'git+https://github.com/generall/cat_catannlib.git#subdirectory=python_bindings' 
```

## Example

```python
import catannlib
import numpy as np
from collections import defaultdict
import tqdm


dim = 50
elements = 10_000
parts_count = 100

catann = catannlib.Index(space='cosine', dim=dim)
catann.init_index(max_elements = elements, ef_construction = 10, M = 16, random_seed=45)

points = np.random.rand(elements, dim)
catann.add_items(points)


# Assign tags by divisibility, for example
tags = defaultdict(list)
for i in range(elements):
    tags[i % parts_count].append(i)

for tag, ids in tqdm.tqdm(tags.items()):
    catann.add_tags(ids, tag)
    catann.index_tagged(tag, m=8)

target = np.float32(np.random.random((1, dim)))
condition = [[(False, 66)]]

# Result will only include points with id % 66 == 0
found_labels, found_dist = catann.knn_query(target, k=10, conditions=condition)
```

# ToDo

* Query planner
  * decide on search strategy depending on the amount of data-points covered by given condition
    * If number of points is small - use full scan
    * If there is large categories - search in them with graph
    * if there is a large number of small or-conditioned categories - search independently and parallel
