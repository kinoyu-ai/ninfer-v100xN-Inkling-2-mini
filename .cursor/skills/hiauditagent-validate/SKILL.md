---
name: hiauditagent-validate
description: Attack hypothesis validation for smart contracts. Triggers on hiauditagent validate, hiauditagent check attack, or hiauditagent verify.
---

# HiAuditAgent Validate

Rigorous attack hypothesis validation using structured reasoning. Every hypothesis is tested against source code, preconditions, guards, economic viability, and known issues before a verdict is issued. No assumption goes unchecked.

**Context:** This skill validates USER's ideas. For filtering HiAuditAgent output, see hiauditagent-filter.

## When to Activate

- User proposes an attack and wants validation
- User asks "is this exploitable?"
- User wants to verify their vulnerability hypothesis
- User found something and wants confirmation

## When NOT to Use

- User wants automated scanning (→ hiauditagent-analyzer)
- User wants to filter HiAuditAgent results (→ hiauditagent-filter)
- User wants to write a PoC (→ hiauditagent-poc)
- User wants code explained (→ hiauditagent-explainer)

## Validation Framework

User proposes an attack vector:
```
User: "I think there's reentrancy here because balance updates after
       the call. Is this actually exploitable?"
```

### Step 1: Extract and Formalize the Hypothesis

Transform the user's description into a structured hypothesis:

```
Vulnerability type: [reentrancy / access control / oracle manipulation / ...]
Target: [Contract.function()]
Attack flow: [Step-by-step what the attacker would do]
Expected impact: [What the attacker gains / what breaks]
Invariant violated: [Which protocol invariant would this break?]
```

### Step 2: Context Gathering (MANDATORY - Before Code Analysis)

**Read project context to avoid reporting known issues or design decisions.**

1. **README.md** - Understand:
   - Protocol purpose and architecture
   - Trust assumptions explicitly stated
   - Known limitations or accepted risks
   - External dependencies and their trust model

2. **Known issues** - Check for:
   - `known-issues.md`, `KNOWN_ISSUES.md`
   - Issues section in README
   - `audit/` folder with previous findings
   - Inline comments: `// Known issue`, `// @audit-known`, `// WONTFIX`, `// Accepted risk`

3. **Previous audits** - Review:
   - Were similar issues reported and acknowledged?
   - Were similar issues reported and fixed?
   - Has the code changed since the last audit?

**If the proposed attack is a known issue or intentional design:**
```
Verdict: KNOWN ISSUE (not reportable)
Reason: Documented in [source] - "[exact quote]"
```

### Step 3: Source Code Analysis

Read the code systematically:

```
1. Read the TARGET function in full
2. Read ALL modifiers applied to the function (not just names - implementations)
3. Read inherited contracts for relevant overrides or state variables
4. Read functions called BY the target (internal calls, library calls)
5. Read functions that CALL the target (callers can set up attack state)
6. Check storage layout for relevant state variables
7. Use hiauditagent_scan if you need a full program report to map related call paths
```

### Step 4: Attack Path Trace

Walk through the proposed attack step-by-step against the actual code:

```
For each step in the attack:
1. What function does the attacker call?
2. What parameters does the attacker control?
3. What state must exist for this step to succeed?
4. What checks/guards does this step encounter?
5. What state changes occur?
6. Is there a revert condition that would block this step?
7. What is the EVM state after this step? (balances, storage, etc.)
```

### Step 5: Precondition Verification

**Every precondition must be verified as achievable. "Assume X" is not valid.**

| Precondition | Verification Method |
|-------------|-------------------|
| Attacker can call function | Check visibility (external/public) and access modifiers |
| Required state exists | Trace how state reaches the needed value via actual code paths |
| Timing window is available | Check deadline/cooldown enforcement in code |
| Capital is available | Assess if flash loans or reasonable capital suffice |
| Multiple transactions needed | Check if state persists between transactions (frontrunning risk) |
| Specific block state needed | Assess likelihood and attacker control over block params |

### Step 6: Guard Analysis

**Systematically check every protection that could block the attack.**

```
For the target function and all functions in the attack path:

1. ACCESS CONTROL
   - Modifiers: onlyOwner, onlyRole, whenNotPaused, etc.
   - Inline checks: require(msg.sender == X), if (unauthorized) revert
   - Inherited checks from parent contracts

2. REENTRANCY PROTECTION
   - nonReentrant modifier: on this function? On ALL related functions?
   - CEI pattern: is it correctly applied?
   - State-based locks: custom mutex patterns

3. INPUT VALIDATION
   - Parameter bounds checking
   - Zero-value / zero-address checks
   - Overflow protection (Solidity >=0.8 or SafeMath)

4. STATE GUARDS
   - Paused state checks
   - Cooldown/timelock enforcement
   - Balance sufficiency checks
   - Allowance checks for token operations

5. ECONOMIC GUARDS
   - Slippage protection (minAmount parameters)
   - Deadline enforcement (block.timestamp checks)
   - Maximum value caps
   - Fee bounds
```

### Step 7: Economic Viability Assessment

```
Is the attack economically rational?

1. COST to attacker:
   - Gas cost of attack transactions
   - Capital locked during attack (flash loan fees)
   - Risk of failed transaction (lost gas)

2. PROFIT to attacker:
   - Value extracted from protocol/users
   - Is profit > cost?

3. DETECTION risk:
   - Can monitoring detect the attack before completion?
   - Is the attack atomic (single transaction) or multi-step?
   - MEV bots: would they frontrun the attacker?

4. SCALE:
   - Can the attack be repeated?
   - Does TVL make it worthwhile?
   - Does gas price affect viability?
```

### Step 8: Verdict

| Verdict | Criteria | Required Evidence |
|---------|----------|-------------------|
| **CONFIRMED** | Attack works as described with verified preconditions | Full attack trace through actual code, all preconditions verified achievable |
| **CONFIRMED (Modified)** | Attack works but with different parameters or flow than proposed | Modified attack trace, explain deviation from hypothesis |
| **PARTIAL** | Attack works but with significant limitations | Limitations listed with severity reassessment |
| **NOT EXPLOITABLE** | Concrete protection blocks the attack | Specific guard cited with line number and implementation verified |
| **KNOWN ISSUE** | Documented in project as accepted risk | Exact documentation reference quoted |
| **BY DESIGN** | Intentional behavior per protocol architecture | Design rationale referenced |
| **NEEDS PoC** | Theoretically possible but complex interaction makes manual verification insufficient | Specific uncertainty identified, PoC would resolve it |

## Output Format

```markdown
## Attack Validation

**Hypothesis:** [User's proposed attack, formalized]
**Target:** Contract.function()
**Invariant at risk:** [Which protocol invariant would this violate?]

### Context Checked
- [x] README.md reviewed - [relevant trust assumptions noted]
- [x] known-issues.md checked - [result]
- [x] Previous audits reviewed - [result]

### Source Code Analysis
[Key code sections examined with line numbers]

### Attack Path Trace
| Step | Action | State Change | Guards Encountered | Result |
|------|--------|-------------|-------------------|--------|
| 1 | Attacker calls X() | State A → B | none | succeeds |
| 2 | Callback triggers | State B → C | nonReentrant? | blocked/succeeds |
| 3 | ... | ... | ... | ... |

### Preconditions
- [x] Attacker can call function (external, no access modifier)
- [x] Required state achievable (via deposit() on L45)
- [ ] Timing window exists (blocked by cooldown on L67)

### Guards Analysis
- Access control: [checked, result]
- Reentrancy: [checked, result with line numbers]
- Input validation: [checked, result]
- Economic guards: [checked, result]

### Economic Viability
- Attack cost: ~X gas (~$Y at Z gwei)
- Potential profit: ~$W
- Viable: [yes/no/marginal]

### Verdict: [CONFIRMED / PARTIAL / NOT EXPLOITABLE / KNOWN ISSUE / BY DESIGN / NEEDS PoC]

**Reason:** [Specific explanation with code references]
**Evidence:** [Line numbers, modifier implementations, state traces]

### Next Steps (if confirmed)
- Severity: [High/Medium/Low] based on impact and likelihood
- Want me to write a PoC? (→ hiauditagent-poc)
- Want me to format as report? (→ hiauditagent-report)
```

## Common Attack Patterns - Validation Checklist

### Reentrancy
- [ ] External call identified (which type? call, transfer, send, token transfer)
- [ ] State modified AFTER the call (which variables, which lines)
- [ ] No reentrancy guard on function (check modifier list AND implementation)
- [ ] No reentrancy guard on cross-function paths (shared state with other functions)
- [ ] Called contract can be malicious (user-controlled address? token with callbacks?)
- [ ] Attack profitable (is there enough value to extract?)

### Access Control
- [ ] Function visibility allows external calls
- [ ] No modifier restricting access (check ALL modifiers including inherited)
- [ ] No inline access check (msg.sender comparison, role check)
- [ ] The function performs a privileged operation (not just a view)
- [ ] No guard in caller function (if called internally)

### Oracle Manipulation
- [ ] Oracle price used in a value-affecting calculation
- [ ] Price can be manipulated within a single block (spot price, not TWAP)
- [ ] No staleness check or heartbeat validation
- [ ] Manipulation cost < profit from exploitation
- [ ] Flash loan available for required capital

### Flash Loan Attacks
- [ ] Price-dependent operation vulnerable to single-block manipulation
- [ ] No TWAP or time-weighted protection
- [ ] Sufficient liquidity available for flash loan
- [ ] Attack is atomic (completed within flash loan callback)
- [ ] Profit exceeds flash loan fee

### Front-Running / MEV
- [ ] Transaction ordering affects outcome
- [ ] No commit-reveal scheme
- [ ] No deadline protection
- [ ] Value extractable by sandwich (buy before, sell after)
- [ ] Mempool visibility gives attacker advantage

### Precision / Rounding
- [ ] Division before multiplication in calculation chain
- [ ] Rounding direction favors attacker (down on mint, up on burn)
- [ ] Small amounts can trigger zero-result rounding
- [ ] Precision loss accumulates over multiple operations
- [ ] First depositor can exploit rounding via donation

## After Validation

**If CONFIRMED:**
- Offer to write PoC (→ hiauditagent-poc)
- Offer to format as report (→ hiauditagent-report)
- Suggest severity with justification
- Note if the attack affects other functions or contracts

**If NOT EXPLOITABLE:**
- Explain exactly which guard blocks the attack (with line numbers)
- Point to the specific protection mechanism
- Suggest what would make it exploitable (if code changed)
- Note if the protection is fragile (could break with future changes)

**If NEEDS PoC:**
- Explain what's uncertain
- Suggest what the PoC should test specifically
- Offer to write the PoC (→ hiauditagent-poc)

## Rationalizations to Reject

| Rationalization | Why It's Wrong |
|-----------------|----------------|
| "README is probably not important" | README contains trust assumptions and known issues. Skipping it = duplicate work or missed context. |
| "I'll check known issues later" | Check FIRST. Validating a known issue wastes time and undermines credibility. |
| "The code looks vulnerable, must be exploitable" | Code appearance does not equal exploitability. Trace the FULL attack path through actual code. |
| "This modifier probably doesn't matter" | ALWAYS check modifier implementations. "Probably" means you didn't verify. |
| "I'll assume the preconditions are met" | VERIFY every precondition. Unreachable code paths are not vulnerabilities. |
| "Similar to another vuln I've seen" | Each codebase is different. Validate THIS specific instance against THIS code. |
| "The guard exists but might not work" | Read the guard implementation. Either it works or it doesn't - verify with code. |
| "It's theoretically possible" | Theory without verified preconditions is not a finding. Prove achievability or mark NEEDS PoC. |
| "I'll skip the economic analysis" | An unprofitable attack is not a real-world threat. Assess economic viability. |
| "The attack works if we ignore the timelock" | You can't ignore protections. If a timelock blocks it, the attack doesn't work. |
