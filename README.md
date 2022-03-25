# CPP Neural Net
An unoptimized example of a neural network implementation in C++

## depencencies

Make sure you have a C++ compiler installed no other dependencies are required

## building

to build use make as follows

```
make -j4
```

## running

Make sure you have the data submodule updated before running by doing

```
git submodule update
```

You can run the example by doing the following

```
./build/bin/app
```

You should see the data being loaded, randomly being put into a batch and being used to train the algorithm.
