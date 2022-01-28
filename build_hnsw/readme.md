
```

This version is working:

    package                    |            build
    ---------------------------|-----------------
    numpy-1.16.6               |   py37h2d18471_3          50 KB
    numpy-base-1.16.6          |   py37hdc34a94_3         3.5 MB
    ------------------------------------------------------------
                                           Total:         3.6 MB

The following NEW packages will be INSTALLED:

  blas               pkgs/main/linux-64::blas-1.0-mkl
  intel-openmp       pkgs/main/linux-64::intel-openmp-2021.4.0-h06a4308_3561
  libgfortran-ng     pkgs/main/linux-64::libgfortran-ng-7.3.0-hdf63c60_0
  mkl                pkgs/main/linux-64::mkl-2021.4.0-h06a4308_640
  mkl-service        pkgs/main/linux-64::mkl-service-2.4.0-py37h7f8727e_0
  mkl_fft            pkgs/main/linux-64::mkl_fft-1.3.1-py37hd3c417c_0
  mkl_random         pkgs/main/linux-64::mkl_random-1.2.2-py37h51133e4_0
  numpy              pkgs/main/linux-64::numpy-1.16.6-py37h2d18471_3
  numpy-base         pkgs/main/linux-64::numpy-base-1.16.6-py37hdc34a94_3
  six                pkgs/main/noarch::six-1.16.0-pyhd3eb1b0_0





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

