---
name: hiauditagent-filter
description: False positive filtering for HiAuditAgent static analysis results. Triggers on hiauditagent filter, hiauditagent triage, or hiauditagent clean report.
---

# HiAuditAgent Filter

Rigorous false positive filtering using multi-factor verification. Each finding is verified against source code, context, mitigations, and real-world exploitability before any verdict is issued. Zero tolerance for lazy dismissals or blind confirmations.

**Context:** This skill filters HIAUDITAGENT's output. For validating your own attack ideas, see hiauditagent-validate.

## When to Activate

- After running HiAuditAgent analysis
- User wants to filter false positives
- User asks to triage/clean the report
- User asks "are these findings real?"

## When NOT to Use

- No HiAuditAgent analysis has been run yet (→ hiauditagent-analyzer first)
- User wants to validate their OWN attack idea (→ hiauditagent-validate)
- User wants deeper manual review (→ hiauditagent-analyzer in Review Mode)

## Filtering Strategy

### Priority Triage

```
┌─────────────────────────────────────────┐
│  HIGH SEVERITY (typically 0-5 issues)   │
│  → Verify ALL - these are critical      │
│  → Multi-factor verification required   │
├─────────────────────────────────────────┤
│  MEDIUM SEVERITY (typically 2-10)       │
│  → Verify ALL - these matter            │
│  → Full code context verification       │
├─────────────────────────────────────────┤
│  LOW SEVERITY (can be many)             │
│  → Sample check if >10 issues           │
│  → Check all if ≤10 issues              │
├─────────────────────────────────────────┤
│  GAS / NC                               │
│  → Skip verification (not security)     │
└─────────────────────────────────────────┘
```

## Two Workflows

### Workflow A: Filter In-Memory (No Report File)

When user just ran `hiauditagent_scan` via MCP:

1. Get findings from hiauditagent_scan output
2. For each High/Medium:
   - Read source code at location
   - Apply multi-factor verification (see below)
3. Report confirmed findings only

### Workflow B: Clean Existing Report File (Context-Efficient)

When report file already exists (user ran hiauditagent with output flag):

1. **Don't read the full report** - it's too large
2. Get summary via `hiauditagent_scan` MCP (small)
3. For each High/Medium finding:
   - Read SOURCE CODE (not the report)
   - Apply multi-factor verification
4. For false positives:
   - Use Edit tool to DELETE that section from report.md
   - Find section header (e.g., `## [H-01] Reentrancy...`)
   - Delete entire section until next `## [` or end

**Why Workflow B is efficient:**
- Full report stays on disk (never in context)
- Only load: summary (~2KB) + source code
- Edit file directly instead of regenerating

## Multi-Factor Verification Process

For each finding to verify, apply ALL of the following checks. A finding is only dismissed as false positive when concrete evidence exists.

### Factor 1: Known Issues Check
```
1. Read README.md for documented limitations or known risks
2. Check known-issues.md, KNOWN_ISSUES.md if they exist
3. Search for audit/ folder with previous findings
4. Look for inline comments: // Known issue, // @audit-known, // WONTFIX
5. If the finding matches a known/accepted risk → KNOWN ISSUE (not reportable)
```

### Factor 2: Source Code Verification
```
1. Read the EXACT code at the reported location (not nearby code)
2. Read at least 20 lines of surrounding context
3. Read the full function containing the flagged code
4. Check function modifiers and their implementations
5. Check inherited contracts for relevant overrides or state
```

### Factor 3: Mitigation Analysis
```
For each potential mitigation, verify it ACTUALLY protects:
1. Is the guard on THIS specific code path? (not just nearby)
2. Does the guard cover ALL entry points to the vulnerable state?
3. Can the guard be bypassed via an alternate call path?
4. Is the guard correctly implemented? (not just present)
5. For reentrancy: does the guard protect cross-function scenarios?
```

### Factor 4: Exploitability Assessment
```
1. Can an external, untrusted caller reach this code?
2. What preconditions are needed? Are they realistic?
3. Is there economic incentive to exploit? (gas cost vs. profit)
4. Can the attack be performed atomically (single transaction)?
5. Are there timing constraints that make exploitation impractical?
6. Would the attack be detected by monitoring before completion?
```

### Factor 5: Cross-Reference Validation
```
1. Does this pattern appear elsewhere in the codebase?
2. Are similar patterns protected differently in other functions?
3. Does fixing this break other functionality?
4. Are there related findings that compound this risk?
```

### Factor 6: Compound Risk Analysis

**Group findings by function/contract and escalate when multiple issues compound:**

```
1. Group all findings by target function and contract
2. For each group with 2+ findings, assess compound risk:
   - "missing signer check" + "arbitrary CPI" on same function = CRITICAL compound
   - "unchecked return" + "reentrancy" on same function = CRITICAL compound
   - "weak access control" + "state manipulation" = HIGH compound
3. Escalate severity when compound risk is identified:
   - Two Medium findings that compound → upgrade to High
   - Medium + High that compound → upgrade to Critical
4. Document the compound chain in the finding output
```

**Example compound escalation:**
```
✓ [COMPOUND] withdraw() has 2 compounding issues:
  → [M-03] Missing zero-check on amount + [M-05] Unchecked external call
  → Combined: Attacker can withdraw 0 amount triggering callback without state change
  → Escalated: Medium → High
```

## Verification Checklists by Detector Type

### Reentrancy Findings
- [ ] Is there actually an external call at the flagged location?
- [ ] Is state modified AFTER the call? (check exact ordering)
- [ ] Is there a reentrancy guard (nonReentrant modifier)?
- [ ] Does the guard cover ALL functions sharing this state?
- [ ] Can the called contract be malicious/untrusted?
- [ ] Is this a read-only reentrancy risk (view functions reading during callback)?
- [ ] ERC777/ERC1155 token callbacks considered?

### Access Control Findings
- [ ] Is the function actually intended to be restricted?
- [ ] Are there modifier checks? Verify modifier implementation.
- [ ] Is the caller validated in a parent function or via inheritance?
- [ ] Can the access control be bypassed through a different entry point?
- [ ] For initializers: is there a proper initialized flag check?
- [ ] Two-step process for critical permission changes?

### Unchecked Return Findings
- [ ] Is the return value actually security-relevant?
- [ ] Is SafeERC20 (safeTransfer, safeApprove) used?
- [ ] Is error handling done at a higher level in the call chain?
- [ ] Does the ERC20 implementation actually return a bool? (USDT edge case)
- [ ] Is this a fire-and-forget by design (e.g., reward distribution where failure is acceptable)?

### Integer/Arithmetic Findings
- [ ] Is unchecked{} block used? Is it intentional?
- [ ] Is the value user-controlled or protocol-controlled?
- [ ] Can overflow/underflow actually cause harm in context?
- [ ] Is precision loss significant (>1 wei for the use case)?
- [ ] Division-before-multiplication ordering issues?

### Centralization/Admin Findings
- [ ] Is the admin a multisig or EOA?
- [ ] Is there a timelock on critical operations?
- [ ] Can the admin rug users (withdraw all funds, set 100% fee)?
- [ ] Are admin powers appropriately scoped?
- [ ] Is this a deployment-time concern vs. ongoing risk?

## Verdict Classification

| Verdict | Criteria | Action |
|---------|----------|--------|
| **CONFIRMED** | Exploitable with concrete attack path | Keep in report, recommend fix |
| **CONFIRMED (Low)** | Real issue but low practical impact | Downgrade severity, keep |
| **FALSE POSITIVE** | Concrete mitigation verified in code | Remove with specific evidence |
| **KNOWN ISSUE** | Documented in README/known-issues | Remove, cite documentation |
| **BY DESIGN** | Intentional per protocol architecture | Remove, note design rationale |
| **NEEDS PoC** | Unclear exploitability, complex interaction | Flag for PoC to confirm/deny |

## Evidence Requirements for Dismissal

**You MUST provide specific code evidence to dismiss any High/Medium finding.**

```
Good dismissal:
"✗ [H-02] Reentrancy in deposit() - FALSE POSITIVE
   Evidence: nonReentrant modifier on L23, covers all state-modifying functions
   Verified: modifier implementation in ReentrancyGuard.sol:L15-L28"

Bad dismissal:
"✗ [H-02] Reentrancy in deposit() - has reentrancy guard"
(WHERE is the guard? WHICH line? Does it cover THIS path?)
```

## Output Format

Keep output minimal but evidenced - one line per finding with key evidence:

```
Filtered 5 findings → 2 confirmed, 2 false positives, 1 known issue
AI Validation: 2 false positives auto-removed before manual triage

✓ [H-01] Reentrancy in withdraw() - CONFIRMED (92%)
  → External call L45 before state update L48, no guard present

✓ [M-03] Access control missing on setFee() - CONFIRMED (87%)
  → No modifier, callable by any address

✗ [H-02] Reentrancy in deposit() - FALSE POSITIVE
  → nonReentrant on L23, verified implementation L15-L28

✗ [M-01] Unchecked return in claim() - FALSE POSITIVE
  → Using SafeERC20.safeTransfer (import L4, usage L67)

⊘ [M-02] Centralization risk on pause() - KNOWN ISSUE
  → Documented in README.md L45: "Admin can pause in emergencies"

Removed 3 sections from report.md
```

**Note on AI Validation confidence scores:** When findings show a percentage (e.g., "92%"), this is the AI validation engine's confidence that the finding is a true positive. The score is computed by the LLM after examining the finding's code context, mitigations, and exploitability. Scores above 80% indicate high confidence; 60-80% means the LLM recommends manual verification.

## After Filtering

Ask user:
- "Found X confirmed issues. Want me to write reports for them?"
- "Want me to add PoCs for High severity findings?"
- "Y findings need PoC verification - want me to test them?"

## Rationalizations to Reject

| Rationalization | Why It's Wrong |
|-----------------|----------------|
| "This detector usually has false positives" | Check THIS instance. Statistical priors don't apply to specific code. |
| "The code looks safe" | READ the code. Verify with line numbers and evidence. |
| "I'll mark as FP without reading source" | ALWAYS read source code before any verdict. Non-negotiable. |
| "SafeERC20 is used so all transfer issues are FP" | Verify SafeERC20 is used at THAT specific location. Imports alone don't prove usage. |
| "This is a known pattern, must be fine" | Known patterns can have implementation bugs. Check THIS implementation. |
| "I'll confirm all High to be safe" | False confirmations waste developer time and erode trust. Verify properly. |
| "The modifier is probably correct" | Read the modifier implementation. "Probably" is not evidence. |
| "I can tell from the function name" | Names lie. Code doesn't. Read the implementation. |
| "This guard exists nearby" | Verify the guard is on THIS exact code path, not just in the same contract. |
