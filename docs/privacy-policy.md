# Privacy Policy

*Last updated: March 1, 2026*

---

## 1. Introduction

Beam ("we," "us," or "our") operates a decentralized AI inference network. This Privacy Policy describes how we collect, use, and protect information when you use our platform, website, and related services (collectively, the "Service").

By using the Service, you agree to the collection and use of information as described in this policy.

---

## 2. Information We Collect

### 2.1 Account Information

When you create an account, we may collect:

- Email address
- Username
- Authentication credentials (hashed and salted)

### 2.2 Usage Data

We automatically collect certain information when you use the Service:

- Inference requests and metadata (timestamps, model used, token counts)
- Device information (browser type, operating system)
- IP address and approximate location
- Session duration and interaction patterns

### 2.3 GPU Provider Information

If you contribute GPU resources to the network, we collect:

- Machine fingerprint (hardware-derived hash)
- GPU specifications (model, VRAM, count)
- Node agent version and software metadata
- Uptime and performance metrics
- Transport mode configuration

### 2.4 Payment Information

Payment processing is handled by third-party providers. We do not directly store credit card numbers or bank account details.

---

## 3. How We Use Your Information

We use collected information to:

- Operate and maintain the Service
- Process inference requests and route them through the network
- Manage GPU provider assignments and compute rewards
- Detect and prevent abuse, fraud, and security incidents
- Improve network performance and reliability
- Communicate with you about your account and the Service
- Comply with legal obligations

---

## 4. Inference Data and Privacy

### 4.1 Prompt Visibility and End-to-End Encryption

Beam supports **end-to-end encryption (E2E)** for inference data using X25519 key exchange and AES-256-GCM. When E2E is enabled, prompts and responses are encrypted between the client and the serving node — node operators cannot read your data.

**Without E2E enabled:** Inference prompts are processed by distributed nodes. Nodes that execute transformer blocks may have access to intermediate activations derived from your prompts. We do not guarantee prompt confidentiality against node operators when E2E is not enabled.

We recommend enabling E2E encryption for sensitive conversations.

### 4.2 Transport Modes

Beam offers three transport modes with different privacy characteristics:

| Mode | Privacy Level | Details |
|---|---|---|
| Fast | Standard | Direct TLS connections — lowest latency |
| Secure | Enhanced | TLS with pinned certificates — reduced metadata exposure |
| Onion | Maximum | Tor routing — hides your IP and network identity |

### 4.3 No Logging Policy for Content

We do not log or store the content of your inference prompts or model outputs on our servers beyond what is necessary for request processing. Inference data is transient.

---

## 5. Data Sharing

We do not sell your personal information. We may share information with:

- **GPU providers**: Minimal metadata necessary for inference routing (no personal data)
- **Service providers**: Third-party tools for analytics, payments, and infrastructure
- **Legal authorities**: When required by law, subpoena, or legal process
- **Business transfers**: In connection with a merger, acquisition, or asset sale

---

## 6. Data Security

We implement industry-standard security measures including:

- HMAC-signed authentication for all node communications
- TLS encryption for data in transit
- Hashed and salted credential storage
- Rate limiting and anomaly detection
- Replay attack prevention

---

## 7. Data Retention

- **Account data**: Retained while your account is active and for 30 days after deletion
- **Usage logs**: Retained for up to 90 days for operational purposes
- **Node metrics**: Aggregated and anonymized after 180 days
- **Payment records**: Retained as required by applicable tax and financial regulations

---

## 8. Your Rights

Depending on your jurisdiction, you may have the right to:

- Access the personal data we hold about you
- Correct inaccurate personal data
- Delete your personal data
- Export your data in a portable format
- Opt out of certain data processing activities
- Withdraw consent where processing is based on consent

To exercise these rights, contact us at privacy@openbeam.me.

---

## 9. Cookies and Tracking

We use essential cookies for authentication and session management. We may use analytics tools to understand how the Service is used. You can control cookie preferences through your browser settings.

---

## 10. Children's Privacy

The Service is not directed to individuals under 18 years of age. We do not knowingly collect personal information from children.

---

## 11. International Data Transfers

Your information may be transferred to and processed in countries other than your country of residence. We ensure appropriate safeguards are in place for such transfers.

---

## 12. Changes to This Policy

We may update this Privacy Policy from time to time. We will notify you of material changes by posting the updated policy on our website and updating the "Last updated" date.

---

## 13. Contact Us

If you have questions about this Privacy Policy, contact us at:

- **Email**: privacy@openbeam.me
- **Twitter / X**: [@Beam_open_node](https://x.com/Beam_open_node)
