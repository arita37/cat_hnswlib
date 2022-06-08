bindings: bindings.cpp
	g++ -O3 -Wall -shared -std=c++14 -fvisibility=hidden -fPIC `python3 -m pybind11 --includes` bindings.cpp -o catannlib.so

