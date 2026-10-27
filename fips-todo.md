# FIPS Compliance TODO

## Overview
This document tracks the necessary changes and validations needed to make model-calling fully FIPS compliant for use in regulated environments.

## Package Dependencies to Validate
- [ ] **httpx**: Verify FIPS-compliant crypto backend
  - Check if httpx[fips] is sufficient
  - May need to enforce specific cryptography package version
  
- [ ] **uvicorn**: Ensure FIPS-compliant SSL/TLS
  - Validate SSL configuration
  - Test with FIPS-compliant certificates

- [ ] **Example Dependencies**:
  - numpy and scikit-learn are only used in examples
  - These now live in the `examples` extra in pyproject.toml
  - Document that examples are not FIPS-compliant

## Container Changes
- [ ] Test Containerfile-FIPS in actual FIPS environment
- [ ] Validate FIPS mode activation
- [ ] Verify crypto policy settings
- [ ] Test certificate handling
- [ ] Document any performance impacts

## Testing Requirements
- [ ] Set up FIPS-compliant test environment
- [ ] Create FIPS compliance test suite
- [ ] Add health checks for FIPS mode
- [ ] Document testing procedures

## Documentation Needs
- [ ] Add FIPS compliance section to README
- [ ] Document FIPS-specific configuration
- [ ] Add troubleshooting guide for FIPS issues
- [ ] Document any functionality limitations in FIPS mode

## OpenShift Integration
- [ ] Test deployment on OpenShift with FIPS enabled
- [ ] Document OpenShift-specific configuration
- [ ] Create deployment templates
- [ ] Add health probes for FIPS status

## Security Validation
- [ ] Audit all cryptographic operations
- [ ] Review third-party dependencies
- [ ] Document compliance requirements
- [ ] Create security policy for FIPS mode

## Performance Testing
- [ ] Benchmark FIPS vs non-FIPS operations
- [ ] Document any performance differences
- [ ] Optimize if needed
