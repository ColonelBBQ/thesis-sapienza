# Annotation Agreement Report (Human vs LLM)

- Sentences annotated: **2166**
- Human ambiguous: **312** (14.4%)
- LLM ambiguous: **582** (26.9%)

## Binary ambiguity agreement

- **Cohen's kappa: 0.480** (moderate)
- Raw agreement: 82.5%

| | LLM not ambiguous | LLM ambiguous |
|---|---|---|
| **Human not ambiguous** | 1530 (true neg) | 324 (LLM over-flags) |
| **Human ambiguous** | 54 (LLM under-flags) | 258 (true pos) |

## Six-way type agreement (0-5, including 'none')

- **Cohen's kappa: 0.396** (fair)
- Quadratic-weighted kappa: **0.270**

Human rows × LLM columns:

| Human \ LLM | none | lexical | syntactic | semantic | language_error | pragmatic |
|---|---|---|---|---|---|---|
| **none** | 1530 | 116 | 4 | 160 | 12 | 32 |
| **lexical** | 31 | 134 | 5 | 47 | 7 | 8 |
| **syntactic** | 0 | 1 | 6 | 1 | 0 | 0 |
| **semantic** | 15 | 9 | 1 | 1 | 1 | 1 |
| **language_error** | 2 | 1 | 0 | 2 | 23 | 0 |
| **pragmatic** | 6 | 6 | 0 | 2 | 0 | 2 |

## Type agreement (sentences both mark ambiguous)

- n = 258
- **Cohen's kappa: 0.312** (fair)

## Sample disagreements

### LLM over-flags (human: not ambiguous, LLM: ambiguous)

- "The indicative compute requirements for the IT infrastructure is placed at Annexure B, Form V Inbuilt AntiSpam"
  - LLM type: semantic — Sentence fragments make the intended requirement unclear.
- "The bidder shall submit the respective certificates issued by the authorized agency/persons."
  - LLM type: pragmatic — It is unclear which certificates and which authorized issuers are meant.
- "The location of the data (text, audio, video, image files, drawing files, GIS files, pdf, and any compressed d"
  - LLM type: semantic — Long nested list makes the required data location unclear.
- "However, the elapse of the time shall be excused and in no way any facility/services shall be affected/degrade"
  - LLM type: pragmatic — “the elapse of the time” is unclear and obscures the intended requirement.
- "Once the exit process is completed, remove the Department data, content and other assets from the cloud enviro"
  - LLM type: semantic — Sentence is grammatically broken, obscuring the intended destruction requirement.
- "The CSP/Bidder facilities/services need to be certified / compliant to the following standards based on the pr"
  - LLM type: pragmatic — "based on the project requirements" leaves the applicable standards scope unclear.
- "Electronic discovery (e-discovery) is the process of locating, preserving, collecting, processing, reviewing, "
  - LLM type: semantic — Phrase "in the context of or criminal cases" is grammatically malformed.
- "All the above workflows should be cloud agnostic"
  - LLM type: lexical — "cloud agnostic" is a vague qualitative term.

### LLM under-flags (human: ambiguous, LLM: not)

- "information which has been received from a third party who had the right to disclose the aforesaid information"
  - Human type: pragmatic
- "The Bidder & Private Cloud solution OEM has to ensure that entire security process is followed while VM induct"
  - Human type: lexical
- "The Bidder shall be responsible for monitoring and reporting of consumed services"
  - Human type: lexical
- "Dashboards should auto recognize and accommodate change in infrastructure Hardware"
  - Human type: language_error
- "During the period of the contract, all upgrades/updates or requirements in hardware, software, licensing, impl"
  - Human type: semantic
- "A solution shall not be a "point of failure" in the flow of network traffic; failure of one or more of the sol"
  - Human type: lexical
- "Public Cloud must provide unlimited, unmetered data transfer capability between all the cloud’s datacenters an"
  - Human type: semantic
- "The Public cloud platform must support industry security standards and ensure data, resources and users are pr"
  - Human type: lexical

### Type substitution (both ambiguous, disagree on type)

- Human: **lexical** → LLM: **semantic** | "The Internet connectivity should be available to the applications as per the SLA requirements stated"
- Human: **lexical** → LLM: **syntactic** | "Provide support to technical team of NRC / Department or nominated agency for Optimization of resour"
- Human: **syntactic** → LLM: **lexical** | "CSP / Bidder to conduct vulnerability assessment scanning of the Portal/Website for every 1 hour til"
- Human: **semantic** → LLM: **language_error** | "It is the prime responsibility of CSP to ensure continuity of service at all times of the Agreement "
- Human: **language_error** → LLM: **semantic** | "Shall not delete/ purge any data at the end of the without the express approval of Department"
- Human: **lexical** → LLM: **semantic** | "The Bidder & Private Cloud solution OEM has to deliver following use-cases while building the Privat"
- Human: **semantic** → LLM: **lexical** | "User should be able to choose combination of OS, Application, T-shirt size and Platform while deploy"
- Human: **language_error** → LLM: **semantic** | "The VM servers wherever clustering is required for OS level cluster formation the same has to provis"
