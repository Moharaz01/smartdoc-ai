# 🔒 GDPR Compliance Document
## SmartDoc AI — Privacy-First AI Document Intelligence System
### Prepared in accordance with UK GDPR and the Data Protection Act 2018

**Document Version:** 1.0  
**Privacy Model:** Privacy by Design and Default (Article 25)  
**Supervisory Authority:** Information Commissioner's Office (ICO), United Kingdom

---

## 1. Executive Summary

SmartDoc AI is designed to the highest privacy standard: **privacy by design and by default** (UK GDPR Article 25). All document processing occurs locally on the user's device. No personal data leaves the system. No third-party processors are involved.

This design approach means that for most use cases, UK GDPR compliance obligations are minimal — because the core GDPR obligations are triggered by the transmission and storage of personal data, and SmartDoc AI avoids both.

---

## 2. Technical Privacy Controls — How the System Works

### Data Flow Audit

The following table traces every piece of data through the SmartDoc system:

| Data | Where It Goes | How Long It Stays | Who Can See It |
|------|--------------|-------------------|----------------|
| Uploaded document text | RAM (local) | Current browser session only | Nobody — never transmitted |
| Text chunks | RAM (local) | Current browser session only | Nobody |
| Chunk embeddings (vectors) | RAM (local, FAISS index) | Current browser session only | Nobody |
| User questions | RAM (local) | Processing duration | Nobody |
| AI-generated answers | Streamlit UI | Current session | The user only |
| Document metadata | Not collected | N/A | N/A |

**Conclusion:** SmartDoc AI, in its current design, does not transmit any personal data to any external party. It does not store any data between sessions. It does not log document content.

### What Happens When You Close the Browser Tab

When the browser session ends:
- All uploaded document text is cleared from RAM
- All embeddings stored in the FAISS index are cleared from RAM
- The chat/Q&A history is cleared
- Nothing is written to disk
- Nothing is sent to any server

This means: if you process a confidential document, there is no residual data anywhere after you close the browser — provided you are running the app locally.

---

## 3. Privacy by Design — Article 25 Compliance

Article 25 of UK GDPR requires that data protection is built into systems from the design phase, not added as an afterthought. SmartDoc AI complies with all seven elements of privacy by design:

| Privacy by Design Principle | How SmartDoc AI Implements It |
|---|---|
| **Proactive, not reactive** | Privacy was the primary design constraint — local-first architecture was chosen before writing any code |
| **Privacy as the default** | No opt-in required for privacy protection — it is the default and only mode of operation |
| **Privacy embedded into design** | Local processing is not a feature — it is the fundamental architecture |
| **Full functionality** | Privacy protection does not reduce functionality — the app is fully featured |
| **End-to-end security** | No transmission = no interception risk; RAM-only = no persistent storage risk |
| **Visibility and transparency** | Privacy page documents exactly how data is handled; user can verify |
| **Respect for user privacy** | User retains complete control — they upload and process their own documents |

---

## 4. Scenarios and GDPR Analysis

### Scenario A: User analyses a publicly available report (e.g., NHS Annual Report)

- **Personal data involved?** Minimal — public report may mention individuals by name in formal roles
- **GDPR applies?** Marginally — public figures in their professional capacity have limited GDPR rights regarding publicly available information
- **Compliance action required?** None
- **Risk level:** Negligible

### Scenario B: Employee analyses their own company's internal business reports

- **Personal data involved?** Possibly — reports may mention employee names, financial figures, business performance
- **GDPR applies?** Yes, if named individuals' personal data is processed
- **Lawful basis?** Legitimate interests (Article 6(1)(f)) — analysing business data for work purposes
- **SmartDoc advantage?** Local processing means data doesn't leave the company's device — significantly reduces compliance burden compared to cloud tools
- **Compliance action required?** Update Records of Processing Activities to note use of local AI tool for document analysis

### Scenario C: Solicitor analyses client legal documents

- **Personal data involved?** Yes — client names, case details, potentially special category data (health, financial, legal proceedings)
- **GDPR applies?** Yes, fully
- **Lawful basis?** Article 6(1)(b) (contract) — processing necessary for legal services contract; Article 9(2)(f) for special category data related to legal claims
- **SmartDoc advantage?** Local processing means no third-party DPA required — critical for legal professional privilege and client confidentiality
- **Compliance action required?** Update ROPA; ensure local machine has appropriate security (encrypted disk, access controls); check SRA (Solicitors Regulation Authority) guidance on AI use

### Scenario D: HR manager analyses employee performance reviews

- **Personal data involved?** Yes — employee personal data, potentially special category data (health conditions mentioned in reviews)
- **GDPR applies?** Yes, fully — employment context attracts stricter protections
- **Lawful basis?** Article 6(1)(b) or (c) — contract of employment or legal obligation
- **Additional requirements?** 
  - Inform employees in employment privacy notice that AI tools may be used for HR document management
  - Ensure analysis is limited to legitimate HR purposes
  - Implement access controls (only HR can use this tool with employee data)
  - Do not use analysis outputs to make automated decisions about employees without human review
- **DPIA required?** Possibly, if large scale or involving special category data
- **SmartDoc advantage?** No third-party processor involved — simplifies the legal framework

### Scenario E: Organisation deploys SmartDoc as an enterprise system on a shared server

This scenario is fundamentally different because:
- The system now processes data from multiple users
- Documents from different users may be stored on the same server
- There may be access control and data isolation requirements

**Additional obligations in this scenario:**
1. **Multi-tenancy isolation:** Ensure one user cannot access another user's documents
2. **Session management:** Each session must be truly isolated — clear data between users
3. **Access logging:** Implement audit logs of who used the system and when (not what they processed)
4. **DPIA:** Required due to large-scale processing
5. **Privacy notice:** All users must be informed of how their documents are handled
6. **Data retention:** Define and enforce session termination — how long does the system wait before clearing RAM?
7. **Server security:** Article 32 — appropriate technical security measures for the server infrastructure

---

## 5. Records of Processing Activities (Article 30)

| Field | Value |
|-------|-------|
| Processing activity | AI-powered document summarisation and Q&A |
| Data controller | The individual or organisation operating the system |
| Processor | None — all processing is local; no third-party processors |
| Purpose | Extracting information and insights from text documents |
| Lawful basis | Varies by use case (see Scenario Analysis above) |
| Categories of data | Depends on documents processed — could be any category |
| Special category data | Possible, depending on document content |
| Retention | Session-only — cleared when browser session ends |
| Security | RAM-only storage, no disk write, no transmission |
| Third-country transfers | None |

---

## 6. Cloud Deployment Considerations

If the SmartDoc AI backend is deployed to a cloud server (as opposed to running purely locally), the privacy model changes:

| Local Deployment | Cloud Deployment |
|---|---|
| Data never leaves user's device | Data is transmitted to the cloud server |
| No third-party processors | Cloud provider becomes a data processor |
| No DPA required | DPA required with cloud provider |
| No data residency concerns | Must choose UK/EU data centre for GDPR compliance |
| No network security concerns | TLS required; server security required |
| No data breach risk from transmission | Transmission and server-side breach risks |

**Recommended approach for cloud deployment:**
1. Use **UK-based cloud regions** (Azure UK South — uksouth; AWS eu-west-2 — London; GCP europe-west2 — London)
2. Sign a Data Processing Agreement with the cloud provider (AWS, Azure, and GCP all provide standard DPAs)
3. Enable encryption at rest and in transit
4. Implement session-based data isolation — ensure no document data persists after session end
5. Add audit logging (access metadata, not document content)
6. Conduct a DPIA before deploying at scale with personal data
7. Update privacy notice to disclose cloud processing

---

## 7. AI Governance Considerations

Beyond GDPR, organisations deploying AI document tools should consider:

### The ICO's AI Guidance

The ICO has published detailed guidance on AI and data protection, including:
- Explaining AI decisions (relevant if summaries or answers are used in decision-making)
- AI auditing framework
- Data minimisation in AI training

### The UK AI Regulation Principles (2023 White Paper)

The UK government's AI regulation principles are implemented by existing regulators:
- **Safety and security:** System should not produce harmful outputs from document processing
- **Transparency:** Users should know they are using AI and what models are involved
- **Fairness:** Summaries should not introduce bias — but AI models can amplify existing biases in language
- **Accountability:** Organisations using the tool are accountable for how outputs are used
- **Contestability:** Users should be able to verify AI outputs against the source document (SmartDoc's source attribution feature enables this)

---

*This document was prepared for portfolio purposes. It is not legal advice. Any organisational deployment of AI document tools should involve a qualified Data Protection Officer and, where appropriate, specialist legal counsel.*
