
```

1) Need to change the git commit in 

     build_hnsw/recipe/meta.yaml

     Main branch is djoko


2) python tests are located in 
     /build_hnsw/







https://github.dev/arita37/cat_hnswlib/tree/djoko/build_hnsw/test_query_new3.py






https://github.com/arita37/zbuild/runs/4848542134?check_suite_focus=true



INFO :: The inputs making up the hashes for the built packages are as follows:
{
  "hnsw2-0.0.1-py37h9f74b49_0": {
    "recipe": {
      "gxx": "9.4.0",
      "target_platform": "linux-64"
    }
  }
}


####################################################################################
Resource usage summary:

Total time: 0:05:38.9
CPU usage: sys=0:00:00.8, user=0:00:22.6
Maximum memory usage observed: 784.5M
Total disk usage observed (not including envs): 18.9K







Existing build

https://github.com/conda-forge/hnswlib-feedstock/blob/d2495d24bf22e8d1e2aaf8725316e076cbcf6921/.ci_support/linux_64_numpy1.16python3.7.____cpython.yaml


cdt_name:
- cos6
channel_sources:
- conda-forge,defaults
channel_targets:
- conda-forge main
cxx_compiler:
- gxx
cxx_compiler_version:
- '9'
docker_image:
- quay.io/condaforge/linux-anvil-comp7
numpy:
- '1.16'
pin_run_as_build:
  python:
    min_pin: x.x
    max_pin: x.x
python:
- 3.7.* *_cpython
target_platform:
- linux-64
zip_keys:
- - cdt_name
  - docker_image
- - python
  - numpy
© 2022 GitHub, Inc.


```

