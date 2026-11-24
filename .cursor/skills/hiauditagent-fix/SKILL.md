---
name: hiauditagent-fix
description: Vulnerability remediation and fix implementation for Solidity contracts. Triggers on hiauditagent fix, hiauditagent remediate, hiauditagent patch, or hiauditagent mitigate.
---

# HiAuditAgent Fix

Security-preserving vulnerability remediation that implements correct fixes without introducing new attack surfaces. Every fix is verified against the original vulnerability, tested against the existing suite, and checked for side effects. Fixes solve the root cause, not just the symptom.

## When to Activate

- User has a confirmed vulnerability and wants it fixed
- User asks to implement a remediation
- User asks "how do I fix this?"
- User wants to apply a recommended fix from a report
- User asks to patch a security issue

## When NOT to Use

- Vulnerability hasn't been confirmed yet (→ hiauditagent-validate first)
- User wants to find vulnerabilities (→ hiauditagent-analyzer)
- User wants to write a PoC (→ hiauditagent-poc)
- User wants a report only (→ hiauditagent-report)
- User wants general refactoring without security context (→ hiauditagent-simplify)

## Process

### Step 1: Understand the Vulnerability

```
Before writing any fix:
1. Read the vulnerability description/report in full
2. Read the vulnerable code and surrounding context
3. Understand the ROOT CAUSE (not just the symptom)
4. Understand the ATTACK PATH (how it's exploited)
5. Understand the IMPACT (what the attacker gains)
6. Identify ALL code locations affected (the bug may exist in multiple places)
```

### Step 2: Design the Fix

```
1. Identify the root cause and target it directly
2. Consider multiple fix approaches:
   - Which approach has the smallest attack surface?
   - Which approach is most maintainable?
   - Which approach has the least side effects?
3. Verify the fix addresses ALL instances (not just one location)
4. Check if the fix needs coordination across multiple contracts
5. Ensure the fix doesn't break existing functionality
```

### Step 3: Implement

```
1. Apply the fix using Edit tool
2. One logical change at a time
3. Add comments explaining WHY the fix works (not WHAT it does)
4. Follow existing code style and conventions
5. If adding new imports (e.g., ReentrancyGuard), use the same library version as the project
```

### Step 4: Verify

```
1. Run the full test suite — ALL tests must pass
2. If a PoC exists, run it — it should now FAIL (attack blocked)
3. Review the fix for side effects:
   - Does it change the function's interface? (breaking change for callers)
   - Does it change the function's gas cost significantly?
   - Does it change event emissions?
   - Does it change storage layout? (critical for upgradeable contracts)
4. Run hiauditagent_scan on the fixed file to check for new issues
5. Verify no new linter errors
```

### Step 5: Document

```
1. Summarize what was changed and why
2. Note any behavior changes (even intended ones)
3. Reference the original finding
4. Note if the fix needs corresponding test updates
```

## Fix Patterns by Vulnerability Type

### Reentrancy

**Root cause:** State update after external call.

**Fix approach (in order of preference):**

1. **Apply CEI pattern** (Checks-Effects-Interactions)
```solidity
// BEFORE (vulnerable)
function withdraw(uint256 amount) external {
    require(balances[msg.sender] >= amount, "Insufficient");
    (bool ok,) = msg.sender.call{value: amount}("");
    require(ok, "Transfer failed");
    balances[msg.sender] -= amount;  // state update AFTER call
}

// AFTER (fixed - CEI)
function withdraw(uint256 amount) external {
    require(balances[msg.sender] >= amount, "Insufficient");
    balances[msg.sender] -= amount;  // effect BEFORE interaction
    (bool ok,) = msg.sender.call{value: amount}("");
    require(ok, "Transfer failed");
}
```

2. **Add reentrancy guard** (defense in depth, use WITH CEI)
```solidity
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

contract Vault is ReentrancyGuard {
    function withdraw(uint256 amount) external nonReentrant {
        // ... (still apply CEI pattern even with guard)
    }
}
```

3. **For cross-function reentrancy:** ensure ALL functions sharing state have the guard.

### Missing Access Control

**Root cause:** No permission check on privileged function.

```solidity
// BEFORE (vulnerable)
function setFee(uint256 newFee) external {
    fee = newFee;
}

// AFTER (fixed - with bounds)
function setFee(uint256 newFee) external onlyOwner {
    require(newFee <= MAX_FEE, "Fee exceeds maximum");
    uint256 oldFee = fee;
    fee = newFee;
    emit FeeUpdated(oldFee, newFee);
}
```

**Always add:**
- Access modifier (onlyOwner, onlyRole, etc.)
- Parameter bounds validation
- Event emission for transparency

### Oracle Manipulation

**Root cause:** Using spot price or stale data.

```solidity
// BEFORE (vulnerable - no freshness check)
function getPrice() public view returns (uint256) {
    (, int256 price,,,) = priceFeed.latestRoundData();
    return uint256(price);
}

// AFTER (fixed - full validation)
function getPrice() public view returns (uint256) {
    (
        uint80 roundId,
        int256 price,
        ,
        uint256 updatedAt,
        uint80 answeredInRound
    ) = priceFeed.latestRoundData();

    require(price > 0, "Invalid price");
    require(updatedAt > 0, "Round not complete");
    require(answeredInRound >= roundId, "Stale price");
    require(block.timestamp - updatedAt <= PRICE_STALENESS_THRESHOLD, "Price too old");

    return uint256(price);
}
```

### Unchecked Return Value

**Root cause:** ERC20 transfer return value ignored.

```solidity
// BEFORE (vulnerable)
IERC20(token).transfer(to, amount);

// AFTER (fixed - using SafeERC20)
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

using SafeERC20 for IERC20;

IERC20(token).safeTransfer(to, amount);
```

### Integer Overflow/Precision Loss

**Root cause:** Division before multiplication or unsafe casting.

```solidity
// BEFORE (vulnerable - precision loss)
uint256 reward = totalReward / totalStaked * userStaked;

// AFTER (fixed - multiply first)
uint256 reward = totalReward * userStaked / totalStaked;
```

```solidity
// BEFORE (vulnerable - unsafe downcast)
uint128 amount = uint128(largeValue);  // silent truncation

// AFTER (fixed - checked cast)
require(largeValue <= type(uint128).max, "Value too large");
uint128 amount = uint128(largeValue);
```

### First Depositor / Share Inflation

**Root cause:** Rounding to zero shares for early depositors.

```solidity
// Fix option 1: Virtual shares (like OpenZeppelin ERC4626)
function _convertToShares(uint256 assets) internal view returns (uint256) {
    uint256 supply = totalSupply();
    return (assets * (supply + VIRTUAL_SHARES)) / (totalAssets() + VIRTUAL_ASSETS);
}

// Fix option 2: Minimum deposit requirement
function deposit(uint256 assets) external returns (uint256 shares) {
    shares = _convertToShares(assets);
    require(shares > MIN_SHARES, "Deposit too small");
    // ...
}

// Fix option 3: Dead shares on first deposit
function deposit(uint256 assets) external returns (uint256 shares) {
    if (totalSupply() == 0) {
        shares = assets - DEAD_SHARES;
        _mint(address(0xdead), DEAD_SHARES);  // burn dead shares
    } else {
        shares = _convertToShares(assets);
    }
    // ...
}
```

### Denial of Service (Unbounded Loop)

**Root cause:** Loop iterates over user-controlled array.

```solidity
// BEFORE (vulnerable - unbounded loop)
function distributeRewards() external {
    for (uint256 i = 0; i < stakers.length; i++) {
        _sendReward(stakers[i]);
    }
}

// AFTER (fixed - pull pattern)
mapping(address => uint256) public pendingRewards;

function claimReward() external {
    uint256 reward = pendingRewards[msg.sender];
    require(reward > 0, "No reward");
    pendingRewards[msg.sender] = 0;
    token.safeTransfer(msg.sender, reward);
}
```

## Fix Quality Standards

### 1. Root Cause, Not Symptom

```
BAD: Adding a balance check that coincidentally blocks the attack
GOOD: Fixing the state ordering that enables the attack

Why: Symptom fixes are fragile. Future changes can re-expose the vulnerability.
```

### 2. Minimal and Focused

```
BAD: Refactoring the entire function while fixing a bug
GOOD: Smallest change that resolves the vulnerability

Why: Minimal changes are easier to review, less likely to introduce new bugs,
     and clearly show what the security fix is.
```

### 3. Defense in Depth

```
BAD: Only applying CEI pattern for reentrancy
GOOD: CEI pattern + nonReentrant modifier

Why: Belt and suspenders. If one defense fails, the other catches it.
```

### 4. No Silent Behavior Changes

```
BAD: Fix changes revert message but doesn't mention it
GOOD: Fix is documented, behavior changes are explicit

Why: Off-chain systems may depend on specific revert reasons or event patterns.
```

## What NOT to Do When Fixing

| Anti-Pattern | Why It's Dangerous |
|-------------|-------------------|
| Fix only one instance when the bug exists in multiple places | Other instances remain vulnerable |
| Add `unchecked` to suppress an overflow error | Hides the real bug instead of fixing it |
| Remove a test that now fails because of the fix | The test caught the old behavior; update it to catch the new behavior |
| Change visibility to hide a vulnerable function | The function still exists and may be called internally |
| Add a bandaid check without understanding root cause | The check may be insufficient or bypassable |
| Modify storage layout in an upgradeable contract | Can corrupt state. New variables go at the end or use gap space |
| Change function signatures in deployed contracts | Breaks existing integrations and callers |

## Output Format

```markdown
## Fix Applied: [Vulnerability Title]

**Finding:** [Reference to original finding, e.g., H-01]
**Root Cause:** [One sentence describing the underlying defect]
**Fix Strategy:** [CEI pattern / access control / input validation / etc.]

### Changes

**File:** Vault.sol

1. **L45-L50:** Reordered state update before external call (CEI pattern)
2. **L12:** Added `ReentrancyGuard` import and inheritance
3. **L45:** Added `nonReentrant` modifier to `withdraw()`

### Verification

- [x] All existing tests pass
- [x] PoC `test_Reentrancy_Withdraw_PoC` now fails (attack blocked)
- [x] No new hiauditagent findings introduced
- [x] Storage layout unchanged (not upgradeable)
- [x] No interface changes (backward compatible)

### Side Effects
- Gas cost of withdraw() increased by ~10k due to nonReentrant modifier
- No other behavioral changes
```

## Rationalizations to Reject

| Rationalization | Why It's Wrong |
|-----------------|----------------|
| "I'll fix it and skip testing" | Untested fixes can introduce worse bugs than the original. Always test. |
| "The fix is obvious, no need to understand the root cause" | "Obvious" fixes often miss the root cause and leave residual vulnerability. |
| "I'll fix one instance for now" | If the pattern exists elsewhere, all instances need fixing. Check the entire codebase. |
| "I'll add a comment explaining the risk instead of fixing" | Comments don't prevent exploits. Fix the code. |
| "This fix changes too much, I'll do something simpler" | The correct fix is the one that addresses the root cause. Simpler is not always better. |
| "The tests pass, so the fix is correct" | Tests have gaps. Reason about correctness independent of test results. |
| "I'll refactor while I'm fixing" | Separate concerns. Security fix in one commit, refactor in another. |
| "The gas cost increase isn't worth it" | Security is not optional. The gas cost of a reentrancy guard is negligible vs. the cost of a hack. |
| "I'll remove the old code path entirely" | Removing code can break callers and integrations. Prefer fixing over deleting. |
| "The fix breaks a test, so I'll update the test" | Understand WHY the test broke. If the test was testing correct behavior, your fix may be wrong. |
