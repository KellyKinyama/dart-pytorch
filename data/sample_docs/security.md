# Security

## Encryption

All customer data is encrypted in transit with TLS 1.3 and at rest with
AES-256. Symmetric encryption keys are stored in AWS KMS and rotated every
90 days. Customers on the Enterprise plan can bring their own KMS key (BYOK)
for tenant-level isolation.

## Authentication

Multi-factor authentication is available to every account and is required for
admins on paid plans. Supported factors are authenticator apps (TOTP), SMS
one-time codes, and hardware security keys (WebAuthn / FIDO2). Single Sign-On
via SAML 2.0 or OIDC is available on the Enterprise plan and requires
configuration by your IT administrator.

## Compliance

We are SOC 2 Type II certified. GDPR and CCPA compliance is standard on every
plan; HIPAA compliance is available for eligible healthcare customers on the
Enterprise plan and requires a signed Business Associate Agreement.

Public trust reports and pen-test summaries are available at trust.example.com.
