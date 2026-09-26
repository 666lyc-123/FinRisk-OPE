# Release and camera-ready checklist

## Verified in this package

- [x] 15 benchmark tasks, 8 behavior policies, and seeds 7/17/27.
- [x] 50,784 case rows and 2,068,330 full-information reward cells.
- [x] Source ZIP SHA-256 hashes match the frozen manifest.
- [x] Two consecutive full runs produced byte-identical `all_cases.csv` (`533b39112236df54d99c5afd5ff19e17e14b59dea0afff9bd8f85fb8269e4247`).
- [x] Paper tables are generated from the committed reference case file.
- [x] Paper text, tables, and selector counts agree with the reference results.
- [x] Reviewer concerns are addressed: constructed logging behavior is delimited, selectors use equal ACCEPT counts, and transaction costs/market impact are explicit limitations.
- [x] Current camera-ready TeX and `main_preupload.pdf` are synchronized, with the real GitHub URL and the certificate-conditional strict false-safe explanation.
- [x] IEEE two-column conference format, Letter page size, 8 pages including references, and embedded Type 1 fonts.
- [x] Bibliography and cross-references compile without unresolved entries.

## Required after creating the GitHub repository

- [x] MIT license added for original code and documentation; third-party data are excluded from relicensing.
- [x] Real repository URL is already present in `paper_source/main.tex`.
- [ ] Upload the repository contents and confirm the GitHub Actions reproduction workflow passes.
- [ ] Submit the rebuilt PDF to IEEE PDF eXpress through the ICDMW CPS author kit.
- [ ] Upload the PDF eXpress-approved PDF, complete IEEE copyright, and complete the required author registration.

The checked-in PDF is a validated local pre-upload candidate, not the final CPS upload file until it passes PDF eXpress.
