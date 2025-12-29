# pDFT

## Env
```shell
conda install -c conda-forge \
  git make cmake ninja \
  openblas gcc clang llvm-openmp \
  libxc xcfun libcint \
  h5py scipy numpy compilers \
  ase toml dftd3-python
```

## Installation
```shell
git clone https://github.com/pyscf/pyscf.git
cd pyscf
git checkout v2.11.0
mkdir -p build
cmake -S pyscf/lib -B build \
  -DENABLE_SMD=ON \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build build -j4
python -m pip install -e . --no-build-isolation
```
python -m pip install sella