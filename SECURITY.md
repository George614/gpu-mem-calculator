# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Currently supported versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

The GPU Memory Calculator team takes security vulnerabilities seriously. We appreciate your efforts to responsibly disclose your findings.

### How to Report

If you discover a security vulnerability, please:

1. **DO NOT** open a public GitHub issue
2. Email the maintainers via GitHub with details about the vulnerability
3. Include as much information as possible:
   - Type of vulnerability
   - Affected versions
   - Steps to reproduce
   - Potential impact
   - Suggested fixes (if any)

### What to Expect

- We will acknowledge receipt of your vulnerability report within 48 hours
- We will provide a more detailed response within 7 days
- We will work on fixing the vulnerability and keep you informed of our progress
- Once the vulnerability is fixed, we will publicly disclose it (with credit to you if desired)

### Disclosure Policy

- Security vulnerabilities will be addressed with high priority
- Fixes will be released as soon as possible
- We will coordinate disclosure with you to ensure adequate time for users to update

## Security Best Practices

When using GPU Memory Calculator:

1. **Configuration Files**: Do not store sensitive information in configuration files
2. **Dependencies**: Keep dependencies up to date
3. **Web Interface**: If deploying the web interface publicly, ensure proper authentication and access controls
4. **Input Validation**: The tool validates configuration inputs, but be cautious with user-provided configuration files

## Known Security Considerations

- This tool performs calculations only and does not interact with actual GPUs or training systems
- Configuration files are loaded from the filesystem - ensure proper file permissions
- The web interface is intended for local use or trusted networks

Thank you for helping keep GPU Memory Calculator secure!
