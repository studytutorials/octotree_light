# Common issues

This file contains common issues and problematic code patterns that might be
encountered while developing supereight.



## OpenMP parallel for loops

``` cpp
// Correct, loop variable declared inside the loop
#pragma omp parallel for
for (size_t i = 0; i < 8; ++i)

// Wrong, loop variable declared outside the loop
size_t i;
#pragma omp parallel for
for (i = 0; i < 8; ++i)
```



## Initialization of shared/unique pointers

``` cpp
// Correct, using the std::shared_ptr/std::unique_ptr constructor
x = std::shared_ptr<T>(new T(constructor_parameter));

// Wrong, using std::make_shared/std::make_unique
x = std::make_shared<T>(constructor_parameter);
```

Most C++ implementations seem to ignore the overloaded operator new defined by
Eigen. See [here](https://gitlab.com/libeigen/eigen/-/issues/1049) for details.

