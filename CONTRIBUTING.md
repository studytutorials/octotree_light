## Contributing
Create a pull request against the `devel` branch and add Nils Funk and Sotiris
Papatheodorou as reviewers. Once any potential issues are approved and the pull
request is accepted, it will be rebased on the current `HEAD` of `devel`,
fixing any conflicts and merged. The original pull request branch will be
deleted after merging.

## Notes
- Looked through correct use of `#pragma omp parallel for`:

```
// Correct
#pragma omp parallel for
for (size_t i = 0; i < 8; ++i)

// Wrong
size_t i;
#pragma omp parallel for
for (i = 0; i < 8; ++i)
```