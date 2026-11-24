---
name: hiauditagent-report
description: Audit report writing for smart contract vulnerabilities. Triggers on hiauditagent report, hiauditagent write up, or hiauditagent document.
---

# HiAuditAgent Report Writer

Produces professional audit reports following the standards of Trail of Bits, OpenZeppelin, and Spearbit. Every finding includes precise impact analysis, concrete exploitation scenarios, and actionable remediation with verification criteria.

## When to Activate

- User wants to document a vulnerability
- User asks to write up a finding
- User wants to format for submission (bug bounty, audit contest, client report)

## When NOT to Use

- User is still exploring/validating (→ hiauditagent-validate)
- User wants to find vulnerabilities (→ hiauditagent-analyzer)
- User wants a PoC first (→ hiauditagent-poc)
- Vulnerability hasn't been confirmed yet

## Process

1. **Gather info** - What's the vulnerability? Which contract/function? What severity?
2. **Read code** - Get exact lines and surrounding context
3. **Classify** - Apply severity framework (see below)
4. **Write report to file** - Create structured markdown file
5. **PoC decision** - Auto-include if High severity or already written

**Do NOT** run HiAuditAgent analysis - user already found the bug.

## File Output (CRITICAL)

**ALWAYS write report to a file. NEVER output report content to terminal.**

### File Naming
```
findings/
├── H-01-reentrancy-in-withdraw.md
├── H-02-oracle-manipulation-in-liquidate.md
├── M-01-unchecked-return-value-in-claim.md
├── M-02-missing-deadline-on-swap.md
└── ...
```

**Pattern:** `<SEVERITY>-<NUMBER>-<short-description>.md`

### Multiple Findings
Ask user: "Create separate files per finding, or one combined report?"
- **Separate:** `findings/H-01-xxx.md`, `findings/M-01-yyy.md` (better for submission)
- **Combined:** `findings/audit-report.md` (all findings in one file)

### After Writing
Confirm to user:
```
Report written: findings/H-01-reentrancy-in-withdraw.md
```

## Severity Classification Framework

### Critical / High

**Criteria (any one sufficient):**
- Direct loss of user funds without requiring admin error
- Permanent freezing of funds (>$100k at realistic TVL)
- Unauthorized minting/burning of protocol tokens
- Complete bypass of access control on critical functions
- Protocol insolvency (liabilities exceed assets)
- Theft via manipulation of core accounting (shares, balances, debt)

**Likelihood qualifier:** Exploitable by any external attacker with no special permissions or unlikely preconditions.

### Medium

**Criteria (any one sufficient):**
- Conditional fund loss requiring specific but realistic circumstances
- Griefing attacks that cost attacker but damage protocol/users
- Governance manipulation with meaningful impact
- Temporary denial of service on value-handling functions
- Bypassing non-critical security controls
- Extraction of value below the "Critical" threshold but above dust

**Likelihood qualifier:** Requires specific timing, state, or moderate capital to exploit.

### Low

**Criteria (any one sufficient):**
- Best practice deviations with no current exploit path
- Issues requiring admin misconduct (trusted admin assumption)
- Informational improvements to defense-in-depth
- Edge cases with negligible economic impact (<$100)
- Code quality issues that could become vulnerabilities if code changes

### Informational / Non-Critical

- Style, naming, documentation improvements
- Gas optimizations with no security relevance
- Suggestions for improved testing coverage

## Report Template

```markdown
## [SEVERITY-XX] Title That Describes The Impact

### Summary
One sentence: what's broken and what a successful exploit achieves.

### Vulnerability Detail

**Root Cause:** [The specific code defect - what's wrong at the implementation level]

**Attack Scenario:**
1. Attacker does X (setup/precondition)
2. Attacker calls function Y with parameters Z
3. Due to [root cause], the protocol [incorrect behavior]
4. Attacker extracts/destroys/freezes [value/access/availability]

**Affected Code:**

`path/to/file.sol#L123-L130`

\`\`\`solidity
function withdraw(uint256 amount) external {
    (bool success, ) = msg.sender.call{value: amount}(""); // @audit root cause
    balances[msg.sender] -= amount; // state update after external call
}
\`\`\`

### Impact

**What an attacker achieves:**
- [Concrete impact: "Complete drainage of vault funds"]
- [Quantified if possible: "Up to X ETH at current TVL"]

**Who is affected:**
- [All depositors / specific user class / protocol treasury]

**Preconditions for exploitation:**
- [What must be true for the attack to work]
- [Capital required, timing windows, access needed]

### Likelihood Assessment

| Factor | Assessment |
|--------|-----------|
| Attacker skill required | Low / Medium / High |
| Capital required | None / Moderate / Significant |
| Preconditions | None / Specific state / Rare conditions |
| Detection difficulty | Easy to detect / Hard to detect |
| MEV opportunity | Yes (will be exploited by bots) / No |

### Recommendation

**Primary fix:**

\`\`\`solidity
function withdraw(uint256 amount) external nonReentrant {
    balances[msg.sender] -= amount; // effect before interaction
    (bool success, ) = msg.sender.call{value: amount}("");
    require(success, "Transfer failed");
}
\`\`\`

**Why this fixes it:** [Explain the security property the fix establishes]

**Alternative mitigations** (if primary fix has trade-offs):
- [Alternative approach and its trade-offs]

**Verification criteria:**
- [ ] [How to verify the fix is correct]
- [ ] [What test should pass/fail after the fix]

### References
- [SWC-107: Reentrancy](https://swcregistry.io/docs/SWC-107)
- [Relevant prior incidents, if any]

### PoC
See: `test/Contract.t.sol::test_VulnName_PoC`
```

## Title Conventions

**Good titles describe the IMPACT, not just the vulnerability type:**
- "Reentrancy in `withdraw()` allows complete drainage of vault funds"
- "Missing access control on `setOracle()` enables price manipulation and forced liquidations"
- "Precision loss in share calculation allows first depositor to steal subsequent deposits"
- "Unbounded loop in `distributeRewards()` causes permanent DoS when >100 stakers"

**Bad titles are vague:**
- "Reentrancy vulnerability"
- "Access control issue"
- "Math error"
- "Possible DoS"

## PoC Section Rules

**Default: Link only, not full code.**

```markdown
### PoC
See: `test/Vault.t.sol::test_Reentrancy_PoC`
```

**Why link-only by default:**
- Report tells the story, PoC file proves it
- Pasting 50+ lines of test code bloats report and reduces readability
- PoC file can be run directly; pasted code cannot

**Include inline PoC only if:**
- User explicitly asks for full code in report
- Submission platform requires inline PoC (some bug bounties)
- PoC is very short (<15 lines)

## SWC / CWE References

Include relevant weakness classifications when available:

| Vulnerability | SWC | CWE |
|--------------|-----|-----|
| Reentrancy | SWC-107 | CWE-841 |
| Integer Overflow | SWC-101 | CWE-190 |
| Unchecked Return | SWC-104 | CWE-252 |
| Access Control | SWC-105 | CWE-284 |
| Delegatecall | SWC-112 | CWE-829 |
| tx.origin | SWC-115 | CWE-477 |
| Timestamp Dependence | SWC-116 | CWE-829 |
| DoS with Gas Limit | SWC-128 | CWE-400 |
| Front Running | SWC-114 | CWE-362 |

## Multiple Findings Organization

Number by severity, then by discovery order:
```
H-01, H-02, H-03
M-01, M-02, M-03
L-01, L-02
```

For combined reports, order sections:
1. Executive Summary (scope, methodology, finding counts)
2. Critical/High findings
3. Medium findings
4. Low findings
5. Informational
6. Appendix (full PoC code if requested)

## Rationalizations to Reject

| Rationalization | Why It's Wrong |
|-----------------|----------------|
| "I'll output to terminal so user can review first" | User can review the file. Terminal output gets lost in scrollback. |
| "It's just one finding, doesn't need a file" | Even one finding needs a file for submission/tracking/version control. |
| "User didn't specify a path" | Use `findings/` directory by default. |
| "I'll paste the full PoC for completeness" | Link is complete. Full code bloats report and hurts readability. |
| "The impact is obvious, I don't need to spell it out" | Spell it out. What's obvious to the auditor may not be to the developer. |
| "I'll skip the likelihood assessment" | Likelihood determines actual risk. Severity without likelihood is incomplete. |
| "SWC references aren't necessary" | They help developers understand the vulnerability class and find educational resources. |
| "The fix is obvious" | Include it anyway with explanation of WHY it works. Developers need to understand the security property. |
| "I'll combine similar findings" | Each finding gets its own report unless they share the exact same root cause and fix. |
