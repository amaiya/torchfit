# Changes

Most recent releases are shown at the top. Each release shows:

- **New**: New classes, methods, functions, etc
- **Changed**: Additional parameters, changes to inputs or outputs, etc
- **Fixed**: Bug fixes that don't change documented behaviour


## 0.2.5 (2020-02-12)

### New:
- N/A

### Changed:
- Edits to tutorial

### Fixed:
- removed extra loss computation during validation/testing


## 0.2.4 (2020-02-11)

### New:
- N/A

### Changed:
- fixed image link in README

### Fixed:
- N/A


## 0.2.3 (2020-02-11)

### New:
- N/A

### Changed:
- changed `loss` to `criterion` in `Learner`

### Fixed:
- N/A


## 0.2.2 (2020-02-11)

### New:
- added support for gradient clipping

### Changed:
- re-factored steps in training to support subclassing `Learner` for custom overrides
- added warning about learning rate setting when user-configured scheduler is supplied to `fit`
- `fit` accepts custom learning rate schedulers

### Fixed:
- N/A


## 0.2.1 (2020-02-07)

### New:
- Support lists of inputs (e.g., [text, offsets] for `EmbeddingBag`)
- Added `predict_example` method
- Added text classification example

### Changed:
- N/A

### Fixed:
- N/A


## 0.2.0 (2020-02-07)

### New:
- Added `gpus` parameter to `Learner` to enable multi-gpu training.
- Support lists of inputs (e.g., [text, offsets] for `EmbeddingBag`)

### Changed:
- N/A

### Fixed:
- N/A


## 0.1.0 (2020-02-06)

- first release



