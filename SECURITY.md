# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | ✅        |

## Reporting a Vulnerability

If you discover a security vulnerability in AlphaLOB, please do **NOT** 
open a public GitHub issue, as this could expose the vulnerability before 
it is fixed.

Instead, contact the maintainer directly:

**GitHub:** [@GokavalasaHemanthNaidu](https://github.com/GokavalasaHemanthNaidu)

Please include the following in your report:
- A clear description of the vulnerability
- Steps to reproduce it
- Potential impact or severity assessment

You can expect an acknowledgement within **7 days**.

## Security Practices

This repository follows these security practices:

- No API keys, credentials, or secrets are stored anywhere in this repository
- The `.env` pattern is fully covered in `.gitignore` across all formats
- Runtime-generated database files (`.duckdb`) are gitignored and never committed
- Jupyter notebook outputs are cleared before committing to prevent data leakage
- Dependencies can be audited with `pip audit` or `safety check`
- GitHub secret scanning is enabled on this repository

## Known Safe Declarations

The following files are intentionally public and contain no sensitive information:

- `models/lob_transformer.onnx` — ONNX inference model (required for Hugging Face deployment)
- `requirements.txt` — Python dependencies (standard open-source packages only)
- `Dockerfile` — Container definition (no credentials embedded)
