# Documentation
## Data-structures for matrixes
Several data structures are used to save matrixes internally. The main differences between those can be described as:
structures containing a full matrix, structures containing only the values of a sparse-matrix with locs, sparse-matrixes with row-indices. And a special data structure based on the last having split up data over the amount of available Vertices.

All data structures are value-agnostic - as in, can be used with any valid arithmatic type like float and int.

### `Matrix`
Contains a $m * n$ array, containing the complete matrix - including any 0 values.

### `SparseMatrix`
Contains 3 arrays of size $nz$ (amount of non-zeros), containing the value, $m$ and $n$ associated with each non-zero in the Matrix.

### 