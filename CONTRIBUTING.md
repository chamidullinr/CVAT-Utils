# Contributing to CVAT-Utils

## Creating Package Release

Automatic CI/CD workflows are set using GitHub Actions.
Create new git `tag` to trigger **Build and Create Release** action.

```bash
git tag -a v1.0.0 -m "Changelog:
* First change.
* Second change."
git push origin v1.0.0
```


## Structure of CVAT-Utils

Module structure of the `CVAT-Utils` library.
```
.
└── cvat_utils                                   
    ├── core                              # Main methods for loading CVAT entities like projects, tasks, jobs, and annotations.
    ├── models                            # pydantic models that map CVAT api.
    ├── api_requests                      # Methods for sending HTTP requests to CVAT with credentials.
    ├── postprocess                       # Helper function used by post-processing scripts outside of the library.
    ├── cli                               # Command line interface scripts. 
    └── utils                             # Other helper functions used internally by the library.
```

## Updating to support newer CVAT

Focus on the following steps when updating the `CVAT_Utils` library to support a newer CVAT version.

1. First, update `models` module.
    It should reflect objects returned by CVAT.
    I.e. `pydantic` models should have the same fields as response objects from CVAT.
    The library automatically validates responses and prints warning if the fields do not match.

    The use of `pydantic` models instead of dictionaries simplifies development
    and prevents potential errors like dictionary `KeyError` which would be more difficult to identify.

2. Check if `core` module needs update.
3. Check if `cli/download` script needs update.
