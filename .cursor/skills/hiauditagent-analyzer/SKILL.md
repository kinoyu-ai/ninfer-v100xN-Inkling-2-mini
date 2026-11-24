---
name: hiauditagent-analyzer
description: Remote audit for application and smart-contract source. Triggers on hiauditagent scan, hiauditagent analyze, hiauditagent audit, hiauditagent review, or hiauditagent check. hiauditagent_scan uploads the code and returns the backend report. It does not run a local static scan.
---

# HiAuditAgent Analyzer

Comprehensive security analysis for application and smart-contract source, including JavaScript, TypeScript, Python, Go, Rust, COBOL, Java, Solidity, and Solana (Anchor/native).

Combines automated analysis with expert-level manual review. The backend auto-detects the target platform from project files (package.json, Anchor.toml, foundry.toml, etc.) and runs the appropriate analysis pipeline. Modeled on methodologies from Trail of Bits, OpenZeppelin, Spearbit, and Consensys Diligence.

## Analysis Modes

Detect what mode user wants:

### Quick Mode (HiAuditAgent Only)

**Triggers:** "run hiauditagent", "quick scan", "static analysis", "automated check"
**Action:** Run hiauditagent_scan. It uploads the source and returns the remote audit report. Report those findings. Do not run a local static scan.
**Context cost:** Low (~500-2000 tokens)

### Review Mode (Manual Only)

**Triggers:** "review this", "look at this code", "is this safe", "what do you think", "check this function", "what's wrong", "how secure"
**Action:** Read code directly, analyze with reasoning. NO hiauditagent tools.
**Context cost:** Medium (depends on code size)
**Best for:** Business logic, specific functions, code understanding

### Full Audit Mode (Combined) - DEFAULT for "audit"

**Triggers:** "audit", "full review", "thorough analysis", "find all vulnerabilities"
**Action:** Smart combination of HiAuditAgent + manual review (see below)
**Context cost:** Higher but managed

## Quick Scan Workflow

User wants: "quick scan", "run hiauditagent", "static analysis"

```
1. Run hiauditagent_scan
2. Show compact summary
3. Done - no deep dive unless asked
```

Output:
```markdown
## Quick Scan Results

**Target:** ./src
**Summary:** 2 High, 3 Medium, 15 Low, 8 Gas

### High Severity
- [H] reentrancy | Vault.sol:45
- [H] delegatecall-in-loop | Proxy.sol:23

### Medium Severity
- [M] unchecked-transfer | Token.sol:89
- [M] tx-origin | Auth.sol:12
- [M] centralization-risk | Admin.sol:34

*15 Low, 8 Gas findings omitted. Ask for details if needed.*
```

## Full Audit Workflow

User wants: "audit", "full review", "thorough analysis"

### Phase 0: Context and Threat Model (Before Any Analysis)

**This phase is non-negotiable. Never skip it.**

```
1. Read README.md - understand protocol purpose, architecture, trust assumptions
2. Check for known-issues.md, audit/ folder, previous audit reports
3. Read deployment config (foundry.toml, hardhat.config) for chain targets
4. Identify protocol type (DeFi lending, DEX, vault, governance, NFT, bridge)
5. Define initial threat model:
   - Who are the actors? (users, admins, keepers, oracles, external protocols)
   - What are the assets? (ETH, ERC20, LP tokens, NFTs, governance power)
   - What are the trust assumptions? (admin is honest, oracle is fresh, etc.)
   - What are the protocol invariants? (totalSupply == sum of balances, etc.)
```

### Phase 1: Remote Audit

```
Run hiauditagent_scan with the file or directory path.
The tool uploads that source and returns the backend report.
Do not run local static analysis, and do not pass severity or detector filters.
A failed upload is an error, not an empty scan.
```

### Phase 2: Severity-Based Triage

```
High:   → Investigate ALL, no exceptions
Medium: → Investigate ALL, no exceptions
Low:    → Skim for patterns, investigate standouts
Gas/NC: → Skip unless user requests
```

### Phase 3: Deep Verification (High/Med)

For each High/Medium finding:
```
1. Read the hiauditagent_scan report for that detector
2. Read source code at reported location + surrounding context
3. Trace all call paths to the vulnerable code
4. Check for existing mitigations (guards, modifiers, state checks)
5. Assess real-world exploitability:
   - Can an external attacker reach this code path?
   - What preconditions must hold?
   - What is the economic incentive?
6. Verdict: Confirmed / False Positive / Needs PoC
```

### Phase 4: Manual Security Review (Critical)

After static analysis, systematically review for issues HiAuditAgent cannot detect. Use the following checklist structured by vulnerability class:

#### A. Access Control and Authorization
- [ ] Missing or incorrect access modifiers on privileged functions
- [ ] Privilege escalation paths (can a user become admin?)
- [ ] Unsafe delegatecall to user-controlled addresses
- [ ] tx.origin used for authentication
- [ ] Unprotected initializers (can be re-initialized)
- [ ] Missing two-step ownership transfers
- [ ] Timelock bypass vectors on critical operations

#### B. Reentrancy and Cross-Function Reentrancy
- [ ] State changes after external calls (CEI violations)
- [ ] Cross-function reentrancy via shared state
- [ ] Read-only reentrancy (view functions reading stale state during callback)
- [ ] Cross-contract reentrancy through protocol-owned contracts
- [ ] ERC777/ERC1155 callback reentrancy on token receives

#### C. Economic and Financial Logic
- [ ] Flash loan attack vectors (price manipulation, collateral manipulation)
- [ ] Sandwich attack opportunities on swaps or price-dependent operations
- [ ] First depositor / share inflation attacks on ERC4626 vaults
- [ ] Rounding errors that favor attacker (round down on deposit, round up on withdraw)
- [ ] Fee-on-transfer token handling (actual amount received < amount parameter)
- [ ] Rebasing token accounting errors
- [ ] Donation attacks (direct transfer to inflate share price)

#### D. Oracle and Price Feed Security
- [ ] Stale price data (missing freshness checks, heartbeat validation)
- [ ] Oracle manipulation via low-liquidity pools
- [ ] Single oracle dependency without fallback
- [ ] Incorrect decimal handling across different price feeds
- [ ] L2 sequencer downtime impact on oracle freshness
- [ ] TWAP window too short for manipulation resistance

#### E. Integer and Arithmetic Safety
- [ ] Precision loss in division-before-multiplication sequences
- [ ] Unsafe casting between uint sizes (uint256 to uint128, etc.)
- [ ] Unchecked blocks with user-controlled values
- [ ] Multiplication overflow in intermediate calculations
- [ ] Zero-amount edge cases that bypass validation

#### F. State Machine and Logic Errors
- [ ] State transition violations (skipping states, reverting to prior states)
- [ ] Race conditions between related transactions
- [ ] Inconsistent state across multiple storage updates
- [ ] Missing deadline/expiration enforcement
- [ ] Off-by-one errors in boundary checks

#### G. External Integration Risks
- [ ] Unchecked return values from low-level calls
- [ ] Assumptions about external contract behavior (ERC20 compliance, etc.)
- [ ] Hardcoded addresses that may differ across chains
- [ ] Missing slippage protection on DEX interactions
- [ ] Lack of deadline on swap operations

#### H. Denial of Service Vectors
- [ ] Unbounded loops over user-controlled arrays
- [ ] Block gas limit exhaustion from growing data structures
- [ ] Griefing via reverting callbacks (push vs pull patterns)
- [ ] Dust amounts that brick withdrawal or liquidation functions

#### I. Upgrade and Proxy Safety
- [ ] Storage layout collisions between proxy and implementation
- [ ] Uninitialized implementation contracts (vulnerable to selfdestruct)
- [ ] Function selector clashes between proxy and implementation
- [ ] Missing storage gaps in inheritable upgradeable contracts
- [ ] UUPS upgrade authorization bypass

### Phase 5: Invariant Definition and Verification

Define and verify core protocol invariants:

```
For each invariant:
1. State it formally: "X must ALWAYS hold true"
2. Identify operations that could violate it
3. Check if all code paths preserve it
4. Document any exceptions or edge cases
```

Common invariant categories:
- **Accounting:** totalSupply == sum(balances), total deposits >= total withdrawals
- **Authorization:** only admin can call X, users can only access their own data
- **Economic:** shares * price >= deposited value, fees never exceed principal
- **Temporal:** cooldown periods enforced, deadlines cannot be bypassed
- **Conservation:** tokens in == tokens out + fees (no tokens created or destroyed)

### Phase 6: Consolidated Report

Combine HiAuditAgent findings + manual findings into one structured report.

## What HiAuditAgent Catches vs Manual Review

| Issue Type                         | HiAuditAgent | Manual |
|------------------------------------|---------|--------|
| Reentrancy patterns                | Yes     | Yes    |
| Unchecked returns                  | Yes     | Yes    |
| Common vulnerability patterns      | Yes     | Yes    |
| Business logic bugs                | No      | Yes    |
| Economic/flash loan attacks        | No      | Yes    |
| Complex access control chains      | No      | Yes    |
| Cross-contract/cross-function      | No      | Yes    |
| Oracle manipulation                | No      | Yes    |
| Rounding/precision exploits        | No      | Yes    |
| State machine violations           | No      | Yes    |
| Invariant violations               | No      | Yes    |
| Upgrade/proxy storage issues       | No      | Yes    |
| MEV/frontrunning vectors           | No      | Yes    |

**Always do manual review for important audits.**

## Response Labeling

When reporting findings, always attribute the source:

- `[HiAuditAgent]` - Found by static analysis
- `[Manual]` - Found by manual reasoning and code review

Examples:
- "[HiAuditAgent] Found reentrancy in withdraw()"
- "[Manual] Flash loan attack via price oracle manipulation in liquidate()"

## Severity Classification (Aligned with Industry Standards)

### Critical / High
- Direct loss of funds (theft, permanent freezing)
- Unauthorized privilege escalation to protocol control
- Manipulation of core accounting that benefits attacker
- Protocol insolvency scenarios

### Medium
- Conditional fund loss requiring specific circumstances
- Griefing attacks that cost attacker but damage protocol/users
- Governance manipulation with meaningful but limited impact
- Temporary denial of service on critical functions
- Bypassing non-critical security controls

### Low
- Informational security improvements
- Gas inefficiencies with no security impact
- Best practice deviations that don't currently create risk
- Issues requiring admin misconduct to exploit (trusted admin assumption)

## Rationalizations to Reject (Anti-Patterns)

| Rationalization | Why It's Wrong |
|-----------------|----------------|
| "HiAuditAgent found the important stuff" | HiAuditAgent misses business logic, economic attacks, cross-contract issues. Manual review is mandatory. |
| "This code looks standard/safe" | Standard-looking code can have non-standard bugs. Read every line. |
| "I'll skip the math, it's probably fine" | Math bugs are HIGH severity. Precision loss, rounding, overflow - never skip. |
| "The function is too long to analyze" | Long functions = more bugs. Break it down section by section. |
| "No obvious issues, must be clean" | Obvious issues are already fixed. The audit's job is to find non-obvious ones. |
| "I already found some bugs, that's enough" | Your job is to find ALL bugs, not some. Incomplete audit = false sense of security. |
| "Previous audit found no issues here" | Codebases change. Re-audit everything in scope. |
| "The team is experienced, this is probably fine" | Even expert teams miss things. That's why audits exist. |
| "This is just a view function, it's safe" | View functions can be exploited via read-only reentrancy and stale state reads. |
| "This only affects the admin" | Centralization risks and admin key compromise are valid findings. |

## Output Format

### For Quick Scan
```markdown
## HiAuditAgent Scan Results

**Found:** 2 High, 3 Medium, 15 Low, 8 Gas

### Critical (High)
| Detector | Location | Brief |
|----------|----------|-------|
| reentrancy | Vault.sol:45 | External call before state update |

### Important (Medium)
| Detector | Location | Brief |
|----------|----------|-------|
| unchecked-transfer | Token.sol:89 | Return value ignored |

---
*20+ Low/Gas findings available. Use "show low severity" for details.*
```

### For Full Analysis
```markdown
## Security Analysis Report

**Protocol:** [Name] | **Chain:** [Target] | **Commit:** [Hash]
**Methodology:** Automated scan + systematic manual review

### Threat Model Summary
- **Actors:** users, admin (2-of-3 multisig), Chainlink oracle
- **Assets at risk:** User deposits (ETH + ERC20)
- **Trust assumptions:** [Listed]

### Protocol Invariants Verified
- [x] totalShares * pricePerShare == totalAssets (HOLDS)
- [ ] Only authorized liquidators can liquidate (VIOLATED - see H-02)

### Automated Scan (HiAuditAgent)
**Summary:** 2 High, 3 Medium confirmed after triage

#### [H-01] Reentrancy in withdraw() - CONFIRMED (92%)
**Location:** Vault.sol:45
**SWC:** SWC-107 (Reentrancy)
**AI Validation:** External call before state update, no reentrancy guard present
**Issue:** External call before balance update allows recursive withdrawal
**Impact:** Complete vault drainage - loss of all deposited funds
**Preconditions:** Attacker deploys contract implementing receive() callback
**Fix:** Apply CEI pattern - move balance update before external call

#### [M-01] Unchecked Transfer Return - CONFIRMED (85%)
...

### Manual Review Findings

#### [H-02] Flash Loan Price Manipulation in liquidate()
**Location:** Lending.sol:120-150
**Issue:** Single-block TWAP allows price manipulation via flash loan
**Impact:** Attacker triggers liquidation of healthy positions for profit
...

### Summary
| Source | High | Medium | Low |
|--------|------|--------|-----|
| HiAuditAgent | 2 | 3 | 15 |
| Manual | 1 | 2 | 0 |
| **Total** | **3** | **5** | **15** |

### Invariant Verification Results
| Invariant | Status | Notes |
|-----------|--------|-------|
| Accounting conservation | PASS | Verified across all deposit/withdraw paths |
| Access control hierarchy | FAIL | See H-02 |
```

## When NOT to Use HiAuditAgent

If user says:
- "review this function" → Just read the function (Review Mode)
- "is this safe" → Read and reason (Review Mode)
- "what does this do" → Explain without scanning (→ hiauditagent-explainer)

These don't need static analysis - just manual reasoning.

## Context Management Rules

### DO
- Load hiauditagent_scan summary (compact, ~500 tokens)
- Load details ONLY for High/Medium findings
- Read source files as needed for verification
- Skip Low/Gas/NC unless user specifically asks

### DON'T
- Auto-load all finding details at once
- Dump entire hiauditagent output with all locations
- Load code you won't analyze
- Request details for 50+ Low severity findings
- Run hiauditagent_scan when user just wants code review

## Context Budget Guide

| Action | Tokens | When to Use |
|--------|--------|-------------|
| hiauditagent_scan | ~500-2000 | Always OK |
| Read source file | ~1000-5000 | When verifying |

**Total budget for audit:** Stay under 20k tokens for HiAuditAgent-related context, leaving room for code reading and manual review.

## Available MCP Tools

1. **hiauditagent_scan** - Upload source and return the remote audit report
   - `path`: File or directory to upload (optional, defaults to current)
   - There is no local static scan and no severity or detector filter
   - This is the only analysis tool. The backend chooses language, detectors, taint, and OWASP coverage.
