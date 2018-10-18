# Active Expansion Sampling (AES)
Experiment code associated with our paper:
Chen W, Fuge M. [Active Expansion Sampling for Learning Feasible Domains in an Unbounded Input Space](http://ideal.umd.edu/papers/paper/samo-aes). Structural and Multidisciplinary Optimization, 57(3), 925-945.

Conventional adaptive sampling/active learning:
![Alt text](/straddle.gif)

AES:
![Alt text](/aes.gif)

## Required packages
- numpy
- scipy
- matplotlib
- sklearn
- libact (only for straddle heuristic)
- pyDOE (only for Neighborhood-Voronoi algorithm)

## 2D examples

AES: 
```
python test_2d_aes.py
```

Neighborhood-Voronoi algorithm: 
```
python test_2d_nv.py
```

Straddle heuristic:
```
python test_2d_straddle.py
```

## Higher-dimensional examples

AES:
```
python test_highdim_aes.py
```

Neighborhood-Voronoi algorithm: 
```
python test_highdim_nv.py
```

Straddle heuristic:
```
python test_highdim_straddle.py
```

## License
This code is licensed under the MIT license. Feel free to use all or portions for your research or related projects so long as you provide the following citation information:

Chen W, Fuge M. Active expansion sampling for learning feasible domains in an unbounded input space. Structural and Multidisciplinary Optimization. 2018 Jan 19. doi:10.1007/s00158-017-1894-y.

    @article{chen2018aes,
      author="Chen, Wei
      and Fuge, Mark",
      title="Active expansion sampling for learning feasible domains in an unbounded input space",
      journal="Structural and Multidisciplinary Optimization",
      year="2018",
      month="Jan",
      day="19",
      issn="1615-1488",
      doi="10.1007/s00158-017-1894-y",
      url="https://doi.org/10.1007/s00158-017-1894-y"
      }

## Application: Finding Novel Designs
This paper describes an interesting application of AES:

Chen W, Fuge M. [Beyond the Known: Detecting Novel Feasible Domains Over an Unbounded Design Space](http://ideal.umd.edu/papers/paper/jmd-feasible-designs). ASME. J. Mech. Des. 2017;139(11):111405-111405-10. doi:10.1115/1.4037306.
