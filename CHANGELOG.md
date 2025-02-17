# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


## [0.2.10] - 2024-01-03

### Added

- Added `rustworkx` as a backend for the graph state simulator
  - Only `networkx` backend was available for pattern optimization.
  By setting the `use_rustworkx` option to True while using `Pattern.perform_pauli_measurements()`,
  graphix will run pattern optimization using `rustworkx` (#98)
- Added `.ccx` and `.swap` methods to `graphix.Circuit`.

### Fixed

- Fixed gflow-based graph visualization (#107)

## [0.2.9] - 2023-11-29

### Added

- internal updates of gflow and linear algebra functionalities:
  - A new option `mode` in `gflow.gflow`, specifying whether to obtain all possible maximally delayed gflow or not (#80)
  - New `MatGF2` class that computes elementary operations and Gauss-Jordan elimination on GF2 field, for faster gflow-finding (#80)

### Changed

- Removed `z3-solver` and added `galois` and `sympy` in `requirements.txt` (#80)

### Removed

- Removed `timeout` optional arguments from `gflow.flow` and `gflow.gflow`.

### Fixed

- Bugfix conditional branch in `gflow.gflowaux` (#80)

## [0.2.8] - 2023-11-05

### Added

- Add support for python 3.11

## [0.2.7] - 2023-10-06

### Added

- Visualization tool of resource state for a pattern, with flow or gflow structures (#78)
- Visualize the resource state by calling `Pattern.draw_graph()`
- Tool to extract fusion network from the resource state of a pattern (#87).

### Changed

### Fixed

## [0.2.6] - 2023-09-29

### Added

- `input_nodes` attribute added to the pattern class (#88)
- `leave_input` optional argument to `Pattern.perform_pauli_measurements()` which leaves the input qubits unmeasured during the optimization.

### Changed

- bump networkx version to 3.* (#82)

## [0.2.5] - 2023-08-17

### Added

- Fast alternative to partial trace (`Statevec.remove_qubit`) for a separable (post-measurement) qubit (#73)

### Changed

- `StatevectorBackend` now uses `Statevec.remove_qubit` after each measurement, instead of performing `ptrace` after multiple measurements, for better performance. This keeps the result exactly the same (#73)
- bump dependency versions for docs build (#77)

## [0.2.4] - 2023-07-06

### Added

- Interface to run patterns on the IBMQ devices. (see PR) (#44)

## [0.2.3] - 2023-06-25

### Changed

- Quantum classifier demo (#57) by @Gopal-Dahale

### Changed

- fixed a bug in a code snippet isn docs (#59), as pointed out by @zilkf92
- fixed issue building docs on readthedocs (#61)
- fixed bug in pauli preprocessing routine and graph state simulator (#63)
- Second output of `pattern.pauli_nodes` (`non_pauli_node` list) is now list of nodes, not list of lists (commands).

## [0.2.2] - 2023-05-25

### Added

- Fast pattern standardization and signal shfiting with `pattern.LocalPattern` class (#42), performance report at #43
- Defaulted local pattern method for `graphix.Pattern.standardize()` and `graphix.Pattern.shift_signals()`. Note the resulting pattern is equivalent to the output of original method.
- Automatic selection of appropriate tensor network graph state preparation strategy `graph_prep="auto"` argument for instantiation of `TensorNetworkBackend` (#50)

### Changed

- option `graph_prep="opt"` for `graph_prep` kwarg of `TensorNetworkBackend` (#50) will be deprecated, and will be replaced by `graph_prep="parallel"`, as we identified that `parallel` preparation is not always optimal.

## [0.2.1] - 2023-04-25

### Changed

- Move import path of `generate_from_pattern` from `graphix.gflow` to `grahpix.generator` (#40)
- Rename `Pattern.get_measurement_order` to `Pattern.get_measurement_commands` (#40)
- Modify `Pattern.get_meas_plane` method to work for Clifford-decorated nodes (#40)

### Fixed

- Fix QFT circuits in examples (#38)
- Fix the stability issue of `Pattern.minimize_space` method which sometimes failed to give theoretical minimum space for patterns with flow (#40)

## [0.2.0] - 2023-03-16

### Added

- Fast circuit translation for some types gates and circuits (see PR) (#16)
- Additional required modules: `quimb` and `autoray` for more performant TN backend (#32)

### Changed

- Restructured tensor-network simulator backend for more optimized contraction (#32)
- Modify TN simulator interface to `TensorNetwork` from `MPS` (#32)

### Fixed

- Treatment of isolated node in `perform_pauli_measurements()` method (#36)

## [0.1.2] - 2022-12-21

### Added

- added QAOA demo to documentation and improved readme

### Fixed

- Fix manual input pattern (#11)

## [0.1.1] - 2022-12-19

### Fixed

- nested array error in numpy 1.24 (deprecated from 1.23.*) fixed and numpy version changed in requirements.txt (#7)
- circuit.standardize_and_transpile() error fixed (#9)

## [0.1.0] - 2022-12-15
