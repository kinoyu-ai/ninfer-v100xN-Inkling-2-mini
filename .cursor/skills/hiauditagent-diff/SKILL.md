---
name: hiauditagent-diff
description: Security-focused code diff and pull request review for Solidity contracts. Triggers on hiauditagent diff, hiauditagent review pr, hiauditagent review changes, or hiauditagent what changed.
---

# HiAuditAgent Diff

Security-focused review of code changes, pull requests, and diffs for Solidity smart contracts. Analyzes modifications through the lens of vulnerability introduction, invariant preservation, and upgrade safety. Designed to catch security regressions that standard code review misses.

## When to Activate

- User asks to review a PR or code diff
- User asks "is this change safe?"
- User wants a security review of modifications
- User asks about impact of code changes
- User wants to review changes before deployment
- User asks "what security implications does this diff have?"

## When NOT to Use

- User wants a full audit of the entire codebase (→ hiauditagent-analyzer)
- User wants to understand existing code (→ hiauditagent-explainer)
- User wants to validate a specific attack (→ hiauditagent-validate)
- No changes have been made (nothing to diff)

## Available MCP Tools

1. **hiauditagent_scan** - Upload changed source and return the remote audit report
   - `path`: File or directory to upload (optional, defaults to current)
   - Use after identifying changed files. The backend handles language, detectors, and related analysis.

## Process

### Step 1: Identify Changes

```
1. Get the diff:
   - Git diff: read the diff output or PR description
   - File comparison: identify added, modified, and deleted files
2. Categorize changes:
   - New contracts added
   - Existing contracts modified
   - Contracts deleted
   - Configuration/deployment changes
   - Test changes (note what IS and ISN'T tested)
3. Prioritize by risk:
   - Changes to value-handling functions (CRITICAL)
   - Changes to access control (HIGH)
   - Changes to state variables or storage layout (HIGH)
   - Changes to external integrations (MEDIUM)
   - Changes to view/pure functions (LOW)
   - Changes to comments/formatting only (SKIP)
```

### Step 2: Change-by-Change Security Analysis

For each modified function/contract:

```
1. WHAT changed?
   - Lines added, removed, modified
   - New parameters, return values, modifiers
   - New state variables, events, errors

2. WHY did it change?
   - Bug fix (verify it actually fixes the bug without introducing new ones)
   - Feature addition (verify it doesn't break existing invariants)
   - Optimization (verify behavioral equivalence)
   - Refactor (verify no logic changes)

3. WHAT COULD GO WRONG?
   - Does the change alter the function's security properties?
   - Does it change access control (added/removed modifiers)?
   - Does it change state mutation order (CEI violations)?
   - Does it change external call patterns (new reentrancy surface)?
   - Does it change arithmetic (precision, overflow, rounding)?
   - Does it change event emissions (off-chain systems may depend on these)?
```

### Step 3: Impact Radius Analysis

```
For each changed function:
1. Run hiauditagent_scan on the changed files, then trace callers and callees of changedFunction in source
2. Identify all callers and callees
3. Check if the change affects:
   - Functions that call the changed function (do they handle new behavior?)
   - Functions called by the changed function (were they updated too?)
   - Functions sharing state with the changed function (invariant preserved?)
4. Flag any caller/callee NOT updated when it should be
```

### Step 4: Invariant Preservation Check

```
For each protocol invariant:
1. Does the change touch code that maintains this invariant?
2. Is the invariant still preserved after the change?
3. Were new invariants introduced that need to hold?

Common invariants to verify:
- Accounting: totalSupply == sum(balances) still holds?
- Access control: permission checks still present?
- Reentrancy: CEI pattern maintained?
- Economic: fee bounds, slippage protection still enforced?
```

### Step 5: Storage Layout Check (if applicable)

```
If state variables were added, removed, or reordered:
1. Run hiauditagent_scan and review storage-related findings for ModifiedContract; confirm slot changes in source
2. For upgradeable contracts:
   - Verify no existing slots changed
   - Verify new variables append at the end or use gap space
   - Verify storage gaps updated correctly
3. For non-upgradeable:
   - Check packing efficiency
   - Verify initialization of new variables
```

### Step 6: Static Analysis on Changed Files

```
1. Run hiauditagent_scan(path="<changed_file>") for each modified .sol file
2. Compare with known baseline (if available):
   - New findings = potential regressions
   - Resolved findings = improvements
3. Investigate any new High/Medium findings
```

## Security Diff Checklist

### Function Modifications

| Check | What to Verify |
|-------|---------------|
| Visibility changed | Was it intentional? external→public adds internal call path |
| Modifier added/removed | Removed modifier = potential access control loss |
| Parameter added/removed | Callers updated? Default handling correct? |
| Return value changed | Callers handle new return type correctly? |
| Revert conditions changed | Error handling in callers still works? |
| Event emissions changed | Off-chain indexers/monitoring updated? |
| External calls added | New reentrancy surface? Return value checked? |
| State variable access changed | CEI ordering maintained? Race conditions? |
| Math operations changed | Precision, rounding, overflow implications? |

### New Code

| Check | What to Verify |
|-------|---------------|
| New external/public functions | Proper access control? Input validation? |
| New state variables | Initialized? Storage layout safe for upgrades? |
| New external calls | Return value checked? Reentrancy protected? |
| New token interactions | SafeERC20 used? Fee-on-transfer handled? |
| New imports | Trusted source? Correct version? |
| New assembly blocks | Memory safety? Stack safety? |

### Deleted Code

| Check | What to Verify |
|-------|---------------|
| Removed access control | Was it protecting something critical? |
| Removed validation | Was the check actually redundant? |
| Removed event emissions | Did off-chain systems depend on them? |
| Removed functions | Were they called elsewhere? Orphaned references? |
| Removed state variables | Upgrade storage layout impact? |

## Output Format

```markdown
## Security Diff Review

**Scope:** [PR/commit/branch description]
**Files changed:** X modified, Y added, Z deleted
**Risk assessment:** [Critical/High/Medium/Low]

### Change Summary

| File | Change Type | Risk | Key Concern |
|------|-----------|------|-------------|
| Vault.sol | Modified | High | withdraw() logic changed |
| NewToken.sol | Added | Medium | New token with custom transfer |
| OldHelper.sol | Deleted | Low | Verify no remaining callers |

### Detailed Findings

#### [DIFF-01] Removed reentrancy guard from withdraw() - CRITICAL
**File:** Vault.sol L45
**Change:** `nonReentrant` modifier removed from `withdraw()`
**Before:** `function withdraw(uint256 amount) external nonReentrant`
**After:** `function withdraw(uint256 amount) external`
**Risk:** Reentrancy attack now possible. External call on L52 occurs before state update on L55.
**Action:** Restore `nonReentrant` modifier or apply CEI pattern.

#### [DIFF-02] New state variable without storage gap update - MEDIUM
**File:** Vault.sol L15
**Change:** Added `uint256 public newFeeRate` state variable
**Risk:** For upgradeable contract, this shifts storage layout. Storage gap not decremented.
**Action:** Reduce `__gap` size by 1 to maintain compatible layout.

#### [DIFF-03] Safe refactor: error messages updated - INFO
**File:** Vault.sol L30-L35
**Change:** `require` strings replaced with custom errors
**Risk:** None. Behavioral equivalence maintained. Gas improvement.

### Invariant Check

| Invariant | Status | Notes |
|-----------|--------|-------|
| Accounting conservation | OK | deposit/withdraw math unchanged |
| Access control | BROKEN | See DIFF-01 |
| CEI ordering | OK | State updates before external calls |
| Fee bounds | OK | MAX_FEE check still present |

### Impact Radius

Functions affected by changes to withdraw():
- deposit() - shares same state ✓ (not affected)
- claim() - calls _updateRewards() ✓ (not affected)
- liquidate() - reads balance state ⚠ (verify with new flow)

### Recommendation
- BLOCK: Fix DIFF-01 before merge (critical security regression)
- FIX: Address DIFF-02 storage gap (upgrade safety)
- APPROVE: DIFF-03 is safe to merge
```

## Common Dangerous Diff Patterns

### Pattern: Guard Removal

```diff
- function withdraw(uint256 amount) external nonReentrant {
+ function withdraw(uint256 amount) external {
```

**Always flag.** Removing security guards is almost never intentional.

### Pattern: Visibility Escalation

```diff
- function _internalHelper() internal {
+ function _internalHelper() public {
```

**Flag.** Internal-to-public exposes a new attack surface.

### Pattern: CEI Order Change

```diff
  function withdraw(uint256 amount) external {
-     balances[msg.sender] -= amount;
      (bool ok,) = msg.sender.call{value: amount}("");
+     balances[msg.sender] -= amount;
  }
```

**Critical.** State update moved after external call = reentrancy.

### Pattern: Unchecked Block Addition

```diff
- uint256 result = a - b;
+ unchecked { uint256 result = a - b; }
```

**Verify bounds.** Is `a >= b` guaranteed by prior validation?

### Pattern: New External Call

```diff
  function deposit() external payable {
      shares[msg.sender] += calculateShares(msg.value);
+     oracle.notifyDeposit(msg.sender, msg.value);
  }
```

**Flag.** New external call = new reentrancy vector + new dependency.

## Rationalizations to Reject

| Rationalization | Why It's Wrong |
|-----------------|----------------|
| "It's just a small change" | Small changes can have massive security impact. One removed modifier = exploitable. |
| "The tests pass" | Tests may not cover security-relevant edge cases. Reason about correctness independently. |
| "The author knows what they're doing" | Even experts make mistakes. That's why we review. |
| "It's just a refactor, logic didn't change" | Verify. "Just a refactor" frequently introduces subtle behavioral changes. |
| "The removed code was redundant" | Was it? Verify the check is genuinely enforced elsewhere. |
| "I'll focus on the new code only" | Modifications to existing code are often more dangerous than additions. Review everything. |
| "The change is too large to review completely" | Break it down function by function. Large changes NEED thorough review. |
| "It looks like it matches the description" | Read the actual code, not the PR description. Code is truth, descriptions can be wrong. |
